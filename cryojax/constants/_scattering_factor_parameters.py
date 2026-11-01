"""
Routines for loading atomic structures.
"""

from typing import Any, Literal

import equinox as eqx
import numpy as np
from jaxtyping import Float, Int

from ._empirical_tables import _CATALASE_TABLE, _EMDB_TABLE


class PengScatteringFactorParameters(eqx.Module, strict=True):
    """Loads electron scattering factor
    parameters from "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996).
    """

    a: Float[np.ndarray, " n_atoms 5"]
    b: Float[np.ndarray, " n_atoms 5"]

    def __init__(self, atomic_numbers: Int[np.ndarray, " n_atoms"]):
        """**Arguments:**

        - `atomic_numbers`:
            An array of atomic numbers.
        """
        parameters = _PENG_PARAMETER_TABLE[atomic_numbers]
        self.a = parameters[:, 0, :]
        self.b = parameters[:, 1, :]


class EmpiricalScatteringFactorParameters(eqx.Module, strict=True):
    """Loads electron scattering factor parameters fit to cryo-EM maps, per atom
    type (an element in its bonded environment), from "Measurement of atomic
    scattering factors by cryoelectron microscopy" by Shtyrov et al. (2026).

    There are two tables: `"catalase"`, 42 types fit to 0.005-0.625 inverse
    angstroms (200-1.60 angstroms), and `"emdb"`, 96 types fit to 0.008-0.544
    inverse angstroms (130-1.84 angstroms); outside its range a table is
    extrapolated. Atoms outside the 20 standard amino acids, and types a table
    lacks, take the factors of `PengScatteringFactorParameters`, as in Shtyrov et
    al. The model must have explicit hydrogens; hydrogens whose positions the fit
    could not fix count toward their neighbours' types but get `a = 0`. Each
    atom's terms are sorted by `b`.

    **References:**

    - Shtyrov et al., PNAS (2026), doi:10.1073/pnas.2528758123.
    - Shtyrov & Murshudov, Zenodo, doi:10.5281/zenodo.17084047. The tables are
        reproduced unchanged under CC BY-SA 4.0.
    """

    a: Float[np.ndarray, " n_atoms 5"]
    b: Float[np.ndarray, " n_atoms 5"]

    def __init__(
        self,
        atomic_numbers: Int[np.ndarray, " n_atoms"],
        atom_chemistry: dict[str, Any],
        table: Literal["catalase", "emdb"] = "catalase",
        b_minimum: float = 0.2,
    ):
        """**Arguments:**

        - `atomic_numbers`:
            An array of atomic numbers.
        - `atom_chemistry`:
            The atoms' chemistry, from [`cryojax.io.read_atoms_from_pdb`][] with
            `loads_chemistry = True`.
        - `table`:
            `"catalase"` or `"emdb"`.
        - `b_minimum`:
            Each `b` is raised to at least this, in angstroms squared, since some
            tabulated `b` are zero.
        """
        if table not in _EMPIRICAL_TABLES:
            raise ValueError(
                "`EmpiricalScatteringFactorParameters` takes `table` equal to "
                f"'catalase' or 'emdb', but got `table = {table!r}`."
            )
        atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        if len(atomic_numbers) != len(atom_chemistry["residue_names"]):
            raise ValueError(
                f"`atomic_numbers` has {len(atomic_numbers)} atoms, but "
                f"`atom_chemistry` has {len(atom_chemistry['residue_names'])}."
            )
        atom_types = _atom_types(atomic_numbers, atom_chemistry)
        tabulated = _EMPIRICAL_TABLES[table]
        is_tabulated = np.isin(atom_types, list(tabulated)) & np.isin(
            atom_chemistry["residue_names"], list(_AMINO_ACIDS)
        )
        parameters = np.empty((len(atomic_numbers), 2, 5))
        if is_tabulated.any():
            parameters[is_tabulated] = [tabulated[t] for t in atom_types[is_tabulated]]
        if not is_tabulated.all():
            check_atomic_numbers_supported(atomic_numbers[~is_tabulated])
            parameters[~is_tabulated] = _PENG_PARAMETER_TABLE[
                atomic_numbers[~is_tabulated]
            ]
        parameters[_find_hidden_hydrogens(atomic_numbers, atom_chemistry)] = 0.0
        order = np.argsort(parameters[:, 1], axis=1, kind="stable")
        self.a = np.take_along_axis(parameters[:, 0], order, axis=1)
        self.b = np.maximum(
            np.take_along_axis(parameters[:, 1], order, axis=1), b_minimum
        )


def _atom_types(atomic_numbers, atom_chemistry):
    neighbours = [[] for _ in range(len(atomic_numbers))]
    for i, j in atom_chemistry["bonds"].tolist():
        neighbours[i].append(int(atomic_numbers[j]))
        neighbours[j].append(int(atomic_numbers[i]))
    for key in ("missing_bonds", "unselected_bonds"):
        for i, atomic_number in atom_chemistry[key].tolist():
            neighbours[i].append(atomic_number)
    atom_types = []
    for i, bonded in enumerate(neighbours):
        if not bonded:
            atom_types.append("")
            continue
        name = (atom_chemistry["residue_names"][i], atom_chemistry["atom_names"][i])
        symbols = "".join(_SYMBOLS[z] for z in sorted(bonded))
        atom_types.append(
            f"{_SYMBOLS[int(atomic_numbers[i])]}({symbols}{_FLAGS.get(name, '')})"
        )
    return np.asarray(atom_types, dtype=str)


def _find_hidden_hydrogens(atomic_numbers, atom_chemistry):
    residue_names = atom_chemistry["residue_names"]
    atom_names = atom_chemistry["atom_names"]
    bonds, missing_bonds = atom_chemistry["bonds"], atom_chemistry["missing_bonds"]
    truncated = np.zeros(len(atomic_numbers), dtype=bool)
    truncated[missing_bonds[missing_bonds[:, 1] > 1, 0]] = True
    hides_hydrogens = np.array(
        [name in _HIDDEN_HYDROGEN_PARENTS for name in zip(residue_names, atom_names)],
        dtype=bool,
    ).reshape(-1) | (truncated & (atom_names != "N"))
    is_hydrogen = atomic_numbers == 1
    unselected_bonds = atom_chemistry["unselected_bonds"]
    orphans = unselected_bonds[is_hydrogen[unselected_bonds[:, 0]], 0]
    if len(orphans) > 0:
        i = int(orphans[0])
        raise ValueError(
            "`EmpiricalScatteringFactorParameters` needs the atom each hydrogen bonds, "
            f"but hydrogen {atom_names[i]} of residue {residue_names[i]} at index {i} "
            "bonds an atom that is not selected."
        )
    for parent in missing_bonds[missing_bonds[:, 1] == 1, 0].tolist():
        if not hides_hydrogens[parent]:
            neighbours = bonds[(bonds == parent).any(axis=1)].ravel()
            found = np.count_nonzero(is_hydrogen[neighbours])
            missing = np.count_nonzero(np.all(missing_bonds == (parent, 1), axis=1))
            raise ValueError(
                "`EmpiricalScatteringFactorParameters` needs every hydrogen of the "
                "CCP4 Monomer Library's protonation, with charged termini at chain "
                f"ends and breaks, but atom {atom_names[parent]} of residue "
                f"{residue_names[parent]} at index {parent} has {found} of its "
                f"{found + missing} hydrogens. They are absent from the model or "
                "protonated differently."
            )
    hidden = np.zeros(len(atomic_numbers), dtype=bool)
    for hydrogen, parent in (bonds.T, bonds[:, ::-1].T):
        hidden[hydrogen[is_hydrogen[hydrogen] & hides_hydrogens[parent]]] = True
    return hidden


_EMPIRICAL_TABLES = {"catalase": _CATALASE_TABLE, "emdb": _EMDB_TABLE}
_AMINO_ACIDS = frozenset(
    (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    )
)
_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
_FLAGS = {
    ("ASP", "OD1"): ", carboxyl",
    ("ASP", "OD2"): ", carboxyl",
    ("GLU", "OE1"): ", carboxyl",
    ("GLU", "OE2"): ", carboxyl",
    ("ASN", "OD1"): ", amide",
    ("GLN", "OE1"): ", amide",
}
# Their hydrogens' positions are not fixed by geometry, so the fit did not draw them.
_HIDDEN_HYDROGEN_PARENTS = {
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
    ("CYS", "SG"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),
}


def check_atomic_numbers_supported(atomic_numbers: Int[np.ndarray, " _"]):
    """Throw an error if `atomic_numbers` contains values not
    supported by cryoJAX electron scattering factor tabulations.
    """
    unique_atomic_numbers = np.unique(atomic_numbers).tolist()
    if not set(unique_atomic_numbers).issubset(set(_SUPPORTED_ATOMIC_NUMBERS)):
        raise ValueError(
            "Found unsupported atomic numbers when running "
            "function `cryojax.constants.check_atomic_numbers_supported`. "
            f"Supported atomic numbers are `{_SUPPORTED_ATOMIC_NUMBERS}`, "
            f"but found values {unique_atomic_numbers}."
        )


_SUPPORTED_ATOMIC_NUMBERS = [
    1,
    6,
    7,
    8,
    9,
    11,
    12,
    15,
    16,
    17,
    19,
    20,
    25,
    26,
    27,
    29,
    30,
]

_PENG_PARAMETER_TABLE = np.full((31, 2, 5), np.nan)

# Peng parameters
_PENG_PARAMETER_TABLE[1] = [
    [0.0349, 0.1201, 0.197, 0.0573, 0.1195],
    [0.5347, 3.5867, 12.3471, 18.9525, 38.6269],
]
_PENG_PARAMETER_TABLE[6] = [
    [0.0893, 0.2563, 0.757, 1.0487, 0.3575],
    [0.2465, 1.71, 6.4094, 18.6113, 50.2523],
]
_PENG_PARAMETER_TABLE[7] = [
    [0.1022, 0.3219, 0.7982, 0.8197, 0.1715],
    [0.2451, 1.7481, 6.1925, 17.3894, 48.1431],
]
_PENG_PARAMETER_TABLE[8] = [
    [0.0974, 0.2921, 0.691, 0.699, 0.2039],
    [0.2067, 1.3815, 4.6943, 12.7105, 32.4726],
]
_PENG_PARAMETER_TABLE[9] = [
    [0.1083, 0.3175, 0.6487, 0.5846, 0.1421],
    [0.2057, 1.3439, 4.2788, 11.3932, 28.7881],
]
_PENG_PARAMETER_TABLE[11] = [
    [0.2142, 0.6853, 0.7692, 1.6589, 1.4482],
    [0.3334, 2.3446, 10.083, 48.3037, 138.27],
]
_PENG_PARAMETER_TABLE[12] = [
    [0.2314, 0.6866, 0.9677, 2.1882, 1.1339],
    [0.3278, 2.272, 10.9241, 39.2898, 101.9748],
]
_PENG_PARAMETER_TABLE[15] = [
    [0.2548, 0.6106, 1.4541, 2.3204, 0.8477],
    [0.2908, 1.874, 8.5176, 24.3434, 63.2996],
]
_PENG_PARAMETER_TABLE[16] = [
    [0.2497, 0.5628, 1.3899, 2.1865, 0.7715],
    [0.2681, 1.6711, 7.0267, 19.5377, 50.3888],
]
_PENG_PARAMETER_TABLE[17] = [
    [0.2443, 0.5397, 1.3919, 2.0197, 0.6621],
    [0.2468, 1.5242, 6.1537, 16.6687, 42.3086],
]
_PENG_PARAMETER_TABLE[19] = [
    [0.4115, 1.4031, 2.2784, 2.6742, 2.2162],
    [0.3703, 3.3874, 13.1029, 68.9592, 194.4329],
]
_PENG_PARAMETER_TABLE[20] = [
    [0.4054, 1.388, 2.1602, 3.7532, 2.2063],
    [0.3499, 3.0991, 11.9608, 53.9353, 142.3892],
]
_PENG_PARAMETER_TABLE[25] = [
    [0.3796, 1.2094, 1.7815, 2.542, 1.5937],
    [0.2699, 2.0455, 7.4726, 31.0604, 91.5622],
]
_PENG_PARAMETER_TABLE[26] = [
    [0.3946, 1.2725, 1.7031, 2.314, 1.4795],
    [0.2717, 2.0443, 7.6007, 29.9714, 86.2265],
]
_PENG_PARAMETER_TABLE[27] = [
    [0.4118, 1.3161, 1.6493, 2.193, 1.283],
    [0.2742, 2.0372, 7.7205, 29.968, 84.9383],
]
_PENG_PARAMETER_TABLE[29] = [
    [0.4314, 1.3208, 1.5236, 1.4671, 0.8562],
    [0.2694, 1.9223, 7.3474, 28.9892, 90.6246],
]
_PENG_PARAMETER_TABLE[30] = [
    [0.4288, 1.2646, 1.4472, 1.8294, 1.0934],
    [0.2593, 1.7998, 6.75, 25.586, 73.5284],
]
