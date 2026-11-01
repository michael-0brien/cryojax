import typing
from collections import Counter
from typing import Any, TypedDict

import mdtraj
import numpy as np
import pandas as pd
from jaxtyping import Float, Int, Shaped

from ._pdb import _load_name_replacement_tables, mmdf_to_atoms, mmdf_to_topology


if hasattr(typing, "GENERATING_DOCUMENTATION"):
    _AtomChemistry = dict[str, Any]  # pyright: ignore[reportAssignmentType]
else:

    class _AtomChemistry(TypedDict):
        residue_names: Shaped[np.ndarray, " n_atoms"]
        atom_names: Shaped[np.ndarray, " n_atoms"]
        bonds: Int[np.ndarray, "n_bonds 2"]
        missing_bonds: Int[np.ndarray, "n_missing 2"]


def mmdf_to_chemistry(
    df: pd.DataFrame,
    *,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    stack_models: bool = False,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"], dict[str, Any]
]:
    """Load atoms as [`cryojax.io.mmdf_to_atoms`][] does, with their chemistry.

    The chemistry is a dictionary of `residue_names`, `atom_names`, `bonds`, the
    covalent bonds between returned atoms as pairs of indices, and `missing_bonds`,
    an `(index, atomic number)` pair for each bond to an atom that is not returned,
    being absent from the model or not selected. Bonds are found on the whole model.

    Only the 20 standard amino acids have bonds, protonated as in the CCP4 Monomer
    Library. Peptide bonds and disulfides are found by distance, and each hydrogen
    bonds to the nearest heavy atom of its residue.

    For the arguments, see [`cryojax.io.read_atoms_from_pdb`][].
    """
    if selection_string != "all" and topology is None:
        topology = mmdf_to_topology(
            df,
            standardizes_names,
            model_index=0 if model_index is None and stack_models else model_index,
        )
    positions, atomic_numbers = mmdf_to_atoms(
        df,
        center=center,
        selection_string=selection_string,
        model_index=model_index,
        stack_models=stack_models,
        standardizes_names=standardizes_names,
        topology=topology,
    )
    if model_index is not None:
        df = df[df["model"] == model_index]
    elif stack_models:
        df = df[df["model"] == df["model"].iloc[0]]
    atom_chemistry = _read_chemistry(df, standardizes_names)
    if selection_string != "all":
        assert topology is not None
        atom_chemistry = _select(
            atom_chemistry,
            topology.select(selection_string),
            df["atomic_number"].to_numpy(),
        )
    return positions, atomic_numbers, atom_chemistry


def _read_chemistry(df: pd.DataFrame, standardizes_names: bool) -> _AtomChemistry:
    residue_names, atom_names = _standard_names(df, standardizes_names)
    atomic_numbers = df["atomic_number"].to_numpy()
    positions = df[["x", "y", "z"]].to_numpy(dtype=float)
    residues: dict[tuple, list[int]] = {}
    for row, key in enumerate(zip(df["model"], df["chain"], df["residue_id"])):
        residues.setdefault(key, []).append(row)
    amino_acids = [
        (key, rows)
        for key, rows in residues.items()
        if residue_names[rows[0]] in _AMINO_ACID_BONDS
    ]
    heavy_atoms = [
        _heavy_atoms(key, rows, residue_names, atom_names, atomic_numbers)
        for key, rows in amino_acids
    ]
    bonds, missing_bonds = [], []
    starts, ends = set(range(len(amino_acids))), set(range(len(amino_acids)))
    for i, j in zip(range(len(amino_acids)), range(1, len(amino_acids))):
        c, n = heavy_atoms[i].get("C"), heavy_atoms[j].get("N")
        if (
            amino_acids[i][0][:2] == amino_acids[j][0][:2]
            and c is not None
            and n is not None
            and np.linalg.norm(positions[c] - positions[n]) < _MAX_PEPTIDE_BOND
        ):
            bonds.append((c, n))
            ends.discard(i)
            starts.discard(j)
    sulfurs = [
        (key[0], atoms["SG"])
        for (key, rows), atoms in zip(amino_acids, heavy_atoms)
        if residue_names[rows[0]] == "CYS" and "SG" in atoms
    ]
    bridged = set()
    for k, (model_1, s_1) in enumerate(sulfurs):
        for model_2, s_2 in sulfurs[k + 1 :]:
            distance = np.linalg.norm(positions[s_1] - positions[s_2])
            if model_1 == model_2 and distance < _MAX_DISULFIDE_BOND:
                bonds.append((s_1, s_2))
                bridged |= {s_1, s_2}
    for i, ((key, rows), atoms) in enumerate(zip(amino_acids, heavy_atoms)):
        residue = residue_names[rows[0]]
        residue_bonds = list(_AMINO_ACID_BONDS[residue])
        if i in ends:
            residue_bonds.append(("C", "OXT"))
        expected_hydrogens = Counter(
            a if b.startswith("H") else b
            for a, b in residue_bonds
            if a.startswith("H") or b.startswith("H")
        )
        if i in starts:
            expected_hydrogens["N"] += 2
        if atoms.get("SG") in bridged:
            expected_hydrogens["SG"] -= 1
        for a, b in residue_bonds:
            if a.startswith("H") or b.startswith("H"):
                continue
            row_a, row_b = atoms.get(a), atoms.get(b)
            if row_a is not None and row_b is not None:
                bonds.append((row_a, row_b))
            elif row_a is not None:
                missing_bonds.append((row_a, _ATOMIC_NUMBERS[b[0]]))
            elif row_b is not None:
                missing_bonds.append((row_b, _ATOMIC_NUMBERS[a[0]]))
        hydrogens = [row for row in rows if atomic_numbers[row] == 1]
        found = Counter()
        if hydrogens:
            if not atoms:
                raise ValueError(
                    f"Found hydrogens but no heavy atoms in {_describe(key, residue)}."
                )
            heavy_rows = np.array(list(atoms.values()))
            distances = np.linalg.norm(
                positions[hydrogens, None] - positions[None, heavy_rows], axis=-1
            )
            parents = heavy_rows[np.argmin(distances, axis=1)]
            bonds.extend(zip(parents.tolist(), hydrogens))
            found = Counter(parents.tolist())
        for name, row in atoms.items():
            if found[row] > expected_hydrogens[name]:
                raise ValueError(
                    f"Found an unexpected hydrogen on atom {name} of "
                    f"{_describe(key, residue)}."
                )
            missing_bonds.extend([(row, 1)] * (expected_hydrogens[name] - found[row]))
    return _AtomChemistry(
        residue_names=residue_names,
        atom_names=atom_names,
        bonds=_as_pairs(bonds, sorts_each=True),
        missing_bonds=_as_pairs(missing_bonds, sorts_each=False),
    )


def _standard_names(df, standardizes_names):
    residue_names = df["residue"].to_numpy(dtype=str)
    atom_names = df["atom"].to_numpy(dtype=str)
    if standardizes_names:
        residue_replacements, atom_replacements = _load_name_replacement_tables()
        residue_names = np.asarray(
            [residue_replacements.get(name, name) for name in residue_names], dtype=str
        )
        atom_names = np.asarray(
            [
                atom_replacements.get(residue, {}).get(name, name)
                for residue, name in zip(residue_names, atom_names)
            ],
            dtype=str,
        )
    return residue_names, np.char.strip(atom_names)


def _heavy_atoms(key, rows, residue_names, atom_names, atomic_numbers):
    residue = residue_names[rows[0]]
    known = {name for bond in _AMINO_ACID_BONDS[residue] for name in bond} | {"OXT"}
    atoms = {}
    for row in rows:
        if atomic_numbers[row] == 1:
            continue
        name = atom_names[row]
        if name not in known:
            raise ValueError(
                f"Found an unknown atom {name} in {_describe(key, residue)}."
            )
        if name in atoms:
            raise ValueError(
                f"Found atom {name} twice in {_describe(key, residue)}; alternate "
                "conformations are not supported, so select one."
            )
        atoms[name] = row
    return atoms


def _select(atom_chemistry, selected, atomic_numbers):
    index = np.full(len(atomic_numbers), -1)
    index[selected] = np.arange(len(selected))
    bonds, missing_bonds = atom_chemistry["bonds"], atom_chemistry["missing_bonds"]
    kept = index[bonds] >= 0
    both, first = kept.all(axis=1), kept[:, 0] & ~kept[:, 1]
    second = kept[:, 1] & ~kept[:, 0]
    cut = np.concatenate(
        [
            np.stack([bonds[first, 0], atomic_numbers[bonds[first, 1]]], axis=-1),
            np.stack([bonds[second, 1], atomic_numbers[bonds[second, 0]]], axis=-1),
            missing_bonds[index[missing_bonds[:, 0]] >= 0],
        ]
    )
    cut[:, 0] = index[cut[:, 0]]
    return _AtomChemistry(
        residue_names=atom_chemistry["residue_names"][selected],
        atom_names=atom_chemistry["atom_names"][selected],
        bonds=_as_pairs(index[bonds[both]], sorts_each=True),
        missing_bonds=_as_pairs(cut, sorts_each=False),
    )


def _as_pairs(pairs, sorts_each):
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if sorts_each:
        pairs = np.sort(pairs, axis=1)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _describe(key, residue):
    model, chain, residue_id = key
    return f"residue {residue} {residue_id} of chain {chain} (model {model})"


_ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
_MAX_PEPTIDE_BOND = 1.5 * 1.341
_MAX_DISULFIDE_BOND = 1.5 * 2.04

# Mid-chain bonds of the standard amino acids, from the CCP4 Monomer Library.
_AMINO_ACID_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "ALA": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB1"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "ARG": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "HD2"),
        ("CD", "HD3"),
        ("CD", "NE"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
        ("N", "CA"),
        ("N", "H"),
        ("NE", "CZ"),
        ("NE", "HE"),
        ("NH1", "HH11"),
        ("NH1", "HH12"),
        ("NH2", "HH21"),
        ("NH2", "HH22"),
    ),
    "ASN": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CG", "ND2"),
        ("CG", "OD1"),
        ("N", "CA"),
        ("N", "H"),
        ("ND2", "HD21"),
        ("ND2", "HD22"),
    ),
    "ASP": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CG", "OD1"),
        ("CG", "OD2"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "CYS": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CB", "SG"),
        ("N", "CA"),
        ("N", "H"),
        ("SG", "HG"),
    ),
    "GLN": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "NE2"),
        ("CD", "OE1"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "H"),
        ("NE2", "HE21"),
        ("NE2", "HE22"),
    ),
    "GLU": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "OE1"),
        ("CD", "OE2"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "GLY": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "HA2"),
        ("CA", "HA3"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "HIS": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD2", "HD2"),
        ("CD2", "NE2"),
        ("CE1", "HE1"),
        ("CE1", "NE2"),
        ("CG", "CD2"),
        ("CG", "ND1"),
        ("N", "CA"),
        ("N", "H"),
        ("ND1", "CE1"),
        ("ND1", "HD1"),
        ("NE2", "HE2"),
    ),
    "ILE": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CB", "HB"),
        ("CD1", "HD11"),
        ("CD1", "HD12"),
        ("CD1", "HD13"),
        ("CG1", "CD1"),
        ("CG1", "HG12"),
        ("CG1", "HG13"),
        ("CG2", "HG21"),
        ("CG2", "HG22"),
        ("CG2", "HG23"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "LEU": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "HD11"),
        ("CD1", "HD12"),
        ("CD1", "HD13"),
        ("CD2", "HD21"),
        ("CD2", "HD22"),
        ("CD2", "HD23"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CG", "HG"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "LYS": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "CE"),
        ("CD", "HD2"),
        ("CD", "HD3"),
        ("CE", "HE2"),
        ("CE", "HE3"),
        ("CE", "NZ"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "H"),
        ("NZ", "HZ1"),
        ("NZ", "HZ2"),
        ("NZ", "HZ3"),
    ),
    "MET": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CE", "HE1"),
        ("CE", "HE2"),
        ("CE", "HE3"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("CG", "SD"),
        ("N", "CA"),
        ("N", "H"),
        ("SD", "CE"),
    ),
    "PHE": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "CE1"),
        ("CD1", "HD1"),
        ("CD2", "CE2"),
        ("CD2", "HD2"),
        ("CE1", "CZ"),
        ("CE1", "HE1"),
        ("CE2", "CZ"),
        ("CE2", "HE2"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CZ", "HZ"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "PRO": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "HD2"),
        ("CD", "HD3"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "CD"),
    ),
    "SER": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CB", "OG"),
        ("N", "CA"),
        ("N", "H"),
        ("OG", "HG"),
    ),
    "THR": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG2"),
        ("CB", "HB"),
        ("CB", "OG1"),
        ("CG2", "HG21"),
        ("CG2", "HG22"),
        ("CG2", "HG23"),
        ("N", "CA"),
        ("N", "H"),
        ("OG1", "HG1"),
    ),
    "TRP": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "HD1"),
        ("CD1", "NE1"),
        ("CD2", "CE2"),
        ("CD2", "CE3"),
        ("CE2", "CZ2"),
        ("CE3", "CZ3"),
        ("CE3", "HE3"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CH2", "HH2"),
        ("CZ2", "CH2"),
        ("CZ2", "HZ2"),
        ("CZ3", "CH2"),
        ("CZ3", "HZ3"),
        ("N", "CA"),
        ("N", "H"),
        ("NE1", "CE2"),
        ("NE1", "HE1"),
    ),
    "TYR": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "CE1"),
        ("CD1", "HD1"),
        ("CD2", "CE2"),
        ("CD2", "HD2"),
        ("CE1", "CZ"),
        ("CE1", "HE1"),
        ("CE2", "CZ"),
        ("CE2", "HE2"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CZ", "OH"),
        ("N", "CA"),
        ("N", "H"),
        ("OH", "HH"),
    ),
    "VAL": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CB", "HB"),
        ("CG1", "HG11"),
        ("CG1", "HG12"),
        ("CG1", "HG13"),
        ("CG2", "HG21"),
        ("CG2", "HG22"),
        ("CG2", "HG23"),
        ("N", "CA"),
        ("N", "H"),
    ),
}
