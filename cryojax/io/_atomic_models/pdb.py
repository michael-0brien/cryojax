"""
Read atomic information from a PDB file. Functions and objects are
adapted from `mdtraj`.
"""

import os
import pathlib
from copy import copy
from typing import Literal, TypedDict, cast, overload
from xml.etree import ElementTree

import jax
import mdtraj
import mmdf
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from mdtraj.core import element as elem
from mdtraj.core.topology import Topology


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_b_factors: Literal[False],
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(  # type: ignore
    filename: str | pathlib.Path,
    *,
    loads_b_factors: Literal[True],
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> (
    tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]
    | tuple[
        Float[np.ndarray, "... n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        Float[np.ndarray, " n_atoms"],
    ]
):
    """Load relevant atomic information for simulating cryo-EM
    images from a PDB or mmCIF file. This function wraps the function
    `read_atoms_from_mmdf`.

    !!! info

        The `selection_string` argument enables usage of
        [`mdtraj`](https://www.mdtraj.org/) atom selection syntax.

    !!! warning

        Using `mdtraj` atom selection requires also passing a `topology`
        or this function will generate one on-the-fly. If `model_index = None`
        and there are multiple models in the PDB/mmCIF, the topology is
        generated *only* using the first model index, yet will be used to
        select atoms across all models.

    **Arguments:**

    - `filename`:
        The name of the PDB/mmCIF file to open.
    - `center`:
        If `True`, center the model so that its center of mass coincides
        with the origin.
    - `loads_b_factors`:
        If `True`, return the B-factors of the atoms.
    - `selection_string`:
        A selection string in `mdtraj`'s format.
    - `model_index`:
        An optional index for grabbing a particular model stored in the PDB. If `None`,
        grab all models, where `atom_positions` has a leading dimension for the model.
    - `standardizes_names`:
        If `True`, non-standard atom names and residue names are standardized.
        If set to `False`, this step is skipped.
    - `topology`:
        If you give a topology as input, the topology won't be parsed from the pdb file
        it saves time if you have to parse a big number of files

    **Returns:**

    A tuple whose first element is a `numpy` array of coordinates containing
    atomic positions, and whose second element is an array of atomic element
    numbers. To be clear,

    ```python
    atom_positons, atom_types = read_atoms_from_pdb(...)
    ```

    !!! info

        If your PDB has multiple models, arrays such as the
        atom positions are loaded with a
        leading dimension for each model. To load a single
        model at index 0,

        ```python
        atom_positons, atom_types = read_atoms_from_pdb(..., model_index=0)
        ```
    """
    # Load `mmdf` dataframe forward the `read_atoms_from_mmdf` method
    df = mmdf.read(pathlib.Path(filename))
    return read_atoms_from_mmdf(
        df,
        loads_b_factors=loads_b_factors,
        center=center,
        selection_string=selection_string,
        model_index=model_index,
        standardizes_names=standardizes_names,
        topology=topology,
    )


@overload
def read_atoms_from_mmdf(
    df: pd.DataFrame,
    *,
    loads_b_factors: Literal[False],
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_mmdf(  # type: ignore
    df: pd.DataFrame,
    *,
    loads_b_factors: Literal[True],
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


@overload
def read_atoms_from_mmdf(
    df: pd.DataFrame,
    *,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


def read_atoms_from_mmdf(
    df: pd.DataFrame,
    *,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> (
    tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]
    | tuple[
        Float[np.ndarray, "... n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        Float[np.ndarray, " n_atoms"],
    ]
):
    """Load relevant atomic information for simulating cryo-EM
    images from a `pandas.DataFrame` loaded from the package
    [`mmdf`](https://github.com/teamtomo/mmdf).

    **Arguments:**

    - `df`:
        The dataframe loaded from or formatted as in
        [`mmdf`](https://github.com/teamtomo/mmdf).

    For documentation of other arguments and return value,
    see the function `read_atoms_from_pdb`.
    ```
    """
    # Load atom info from `mmdf` dataframe
    atom_info = _load_atom_info(
        df,
        model_index=model_index,
        loads_masses=center,
        loads_b_factors=loads_b_factors,
    )
    if selection_string != "all":
        if topology is None:
            topology = make_mdtraj_topology(df, standardizes_names, model_index)
        # Filter atoms and grab positions and identities
        selected_indices = topology.select(selection_string)
        atom_positions = atom_info["positions"][:, selected_indices]
        atom_properties = jax.tree.map(
            lambda arr: arr[:, selected_indices], atom_info["properties"]
        )
    else:
        atom_positions = atom_info["positions"]
        atom_properties = atom_info["properties"]
    atom_types = atom_properties["identities"]
    # Center by mass
    if center:
        atom_masses = cast(np.ndarray, atom_properties["masses"])
        atom_positions = _center_atom_coordinates(atom_positions, atom_masses)
    # Return, optionally with b-factors and without leading dimensions
    # if there is only one structure
    atom_positions = (
        np.squeeze(atom_positions, axis=0)
        if atom_positions.shape[0] == 1
        else atom_positions
    )
    atom_types = (
        np.squeeze(atom_types, axis=0) if atom_types.shape[0] == 1 else atom_types
    )
    if loads_b_factors:
        b_factors = cast(np.ndarray, atom_properties["b_factors"])
        b_factors = (
            np.squeeze(b_factors, axis=0) if b_factors.shape[0] == 1 else b_factors
        )
        return atom_positions, atom_types, b_factors
    else:
        return atom_positions, atom_types


def make_mdtraj_topology(
    df: pd.DataFrame,
    standardizes_names: bool = True,
    model_index: int | None = None,
) -> mdtraj.Topology:
    """Generate an `mdtraj.Topology` using an array of atom
    positions and a `pandas.DataFrame` loaded from the package
    [`mmdf`](https://github.com/teamtomo/mmdf).

    **Arguments:**

    - `df`:
        The dataframe loaded from or formatted as in
        [`mmdf`](https://github.com/teamtomo/mmdf).
    - `standardizes_names`:
        If `True`, non-standard atom names and residue names are
        standardized.
    - `model_index`:
        The model index from which to build the topology. Possible
        indicies are captured in `df["model"]`.

    **Returns:**

    An `mdtraj.Topology` object.
    """
    topology = Topology()
    if standardizes_names:
        residue_name_replacements, atom_name_replacements = (
            _load_name_replacement_tables()
        )
    else:
        residue_name_replacements, atom_name_replacements = {}, {}
    if model_index is None:
        model_index = df["model"].unique().tolist()[0]
    df_at_model = df[df["model"] == model_index]
    for atom_index in range(len(df_at_model)):
        df_at_index = df_at_model.iloc[atom_index]
        chain_id = df_at_index["chain"]
        residue_name = df_at_index["residue"]
        residue_id = df_at_index["residue_id"]
        c = topology.add_chain(chain_id)
        if residue_name in residue_name_replacements and standardizes_names:
            residue_name = residue_name_replacements[residue_name]
        # TODO: is it necessary to have `segment_id`, as is parsed in `mdtraj`?
        r = topology.add_residue(residue_name, c, residue_id, segment_id="")
        if residue_name in atom_name_replacements and standardizes_names:
            atom_replacements = atom_name_replacements[residue_name]
        else:
            atom_replacements = {}
        atom_name = df_at_index["atom"]
        if atom_name in atom_replacements:
            atom_name = atom_replacements[atom_name]
        atom_name = atom_name.strip()
        residue_ids = df_at_model[df_at_model["residue_id"] == residue_id]
        element = _guess_element(
            atom_name,
            residue_name,
            residue_length=len(residue_ids),
        )
        charge = df_at_index["charge"]
        # TODO: ok to remove serial number?
        _ = topology.add_atom(
            atom_name,
            element,  # type: ignore
            r,
            # serial=atom.serial_number,
            formal_charge=charge,
        )
    # Generate bonds
    atom_positions = df_at_model[["x", "y", "z"]].to_numpy()
    topology.create_standard_bonds()
    topology.create_disulfide_bonds(atom_positions.tolist())

    return topology


def read_topology_from_pdb(
    filename: str | pathlib.Path,
    model_index: int | None = None,
    standardizes_names: bool = True,
) -> mdtraj.Topology:
    """Load relevant atomic information for simulating cryo-EM
    images from a PDB or mmCIF file. This function wraps the function
    `read_atoms_from_mmdf`.

    Generate an `mdtraj.Topology` from a PDB or mmCIF file.
    This function wraps the function `make_mdtraj_topology`.

    !!! info
        Since we use `mmdf` to parse the PDB/mmCIF file, the
        atom ordering in some of our functions, e.g., `read_atoms_from_pdb`
        may differ from that of `mdtraj.load`. We recommend using this function
        if you need a topology that is consistent with that of `read_atoms_from_pdb`.

    **Arguments:**

    - `df`:
        The dataframe loaded from or formatted as in
        [`mmdf`](https://github.com/teamtomo/mmdf).
    - `standardizes_names`:
        If `True`, non-standard atom names and residue names are
        standardized.
    - `model_index`:
        The model index from which to build the topology. Possible
        indicies are captured in `df["model"]`.

    **Returns:**

    An `mdtraj.Topology` object.
    """
    df = mmdf.read(pathlib.Path(filename))
    return make_mdtraj_topology(df, standardizes_names, model_index)


def _center_atom_coordinates(atom_positions, atom_masses):
    com_position = (
        np.sum(atom_positions * atom_masses[..., None], axis=1)
        / atom_masses.sum(axis=1)[:, None]
    )
    return atom_positions - com_position[:, None, :]


class _AtomProperties(TypedDict):
    identities: Int[np.ndarray, " N"]
    masses: Float[np.ndarray, " N"] | None
    b_factors: Float[np.ndarray, " N"] | None


class _AtomicModelInfo(TypedDict):
    positions: Float[np.ndarray, "M N 3"]
    properties: _AtomProperties


def _load_atom_info(
    df: pd.DataFrame,
    model_index: int | None,
    loads_b_factors: bool,
    loads_masses: bool,
):
    if df.size == 0:
        raise ValueError(
            "When loading an atomic model using `mmdf`, found that "
            "the dataframe was empty."
        )
    # Load atom info
    if model_index is not None:
        df = df[df["model"] == model_index]
        if df.size == 0:
            raise ValueError(
                f"Found no atoms matching `model_index = {model_index}`. "
                "Model numbers available for indexing are "
                f"{df['model'].unique().tolist()}. "
            )
    model_numbers = df["model"].unique().tolist()
    atom_positions, atomic_numbers, atomic_mass, b_factors = [], [], [], []
    for model_index in model_numbers:
        df_at_index = df[df["model"] == model_index]
        atom_positions.append(df_at_index[["x", "y", "z"]].to_numpy())
        atomic_numbers.append(df_at_index["atomic_number"].to_numpy())
        if loads_masses:
            atomic_mass.append(df_at_index["atomic_weight"].to_numpy())
        if loads_b_factors:
            b_factors.append(df_at_index["b_isotropic"].to_numpy())

    # Gather atom info and return
    properties = _AtomProperties(
        identities=np.asarray(atomic_numbers, dtype=int),
        b_factors=(np.asarray(b_factors, dtype=float) if loads_b_factors else None),
        masses=(np.asarray(atomic_mass, dtype=float) if loads_masses else None),
    )
    atom_info = _AtomicModelInfo(
        positions=np.asarray(atom_positions, dtype=float),
        properties=properties,
    )

    return atom_info


def _guess_element(atom_name, residue_name, residue_length):
    """Try to guess the element name.
    Closely follows `mdtraj.formats.pdb.PDBTrajectoryFile._guess_element`."""
    upper = atom_name.upper()
    if upper.startswith("CL"):
        element = elem.chlorine
    elif upper.startswith("NA"):
        element = elem.sodium
    elif upper.startswith("MG"):
        element = elem.magnesium
    elif upper.startswith("BE"):
        element = elem.beryllium
    elif upper.startswith("LI"):
        element = elem.lithium
    elif upper.startswith("K"):
        element = elem.potassium
    elif upper.startswith("ZN"):
        element = elem.zinc
    elif residue_length == 1 and upper.startswith("CA"):
        element = elem.calcium

    # TJL has edited this. There are a few issues here. First,
    # parsing for the element is non-trivial, so I do my best
    # below. Second, there is additional parsing code in
    # pdbstructure.py, and I am unsure why it doesn't get used
    # here...
    elif residue_length > 1 and upper.startswith("CE"):
        element = elem.carbon  # (probably) not Celenium...
    elif residue_length > 1 and upper.startswith("CD"):
        element = elem.carbon  # (probably) not Cadmium...
    elif residue_name in ["TRP", "ARG", "GLN", "HIS"] and upper.startswith("NE"):
        element = elem.nitrogen  # (probably) not Neon...
    elif residue_name in ["ASN"] and upper.startswith("ND"):
        element = elem.nitrogen  # (probably) not ND...
    elif residue_name == "CYS" and upper.startswith("SG"):
        element = elem.sulfur  # (probably) not SG...
    else:
        try:
            element = elem.get_by_symbol(atom_name[0])
        except KeyError:
            try:
                symbol = (
                    atom_name[0:2].strip().rstrip("AB0123456789").lstrip("0123456789")
                )
                element = elem.get_by_symbol(symbol)
            except KeyError:
                element = None

    return element


def _load_name_replacement_tables():
    """Load the list of atom and residue name replacements.
    Closely follows `mdtraj.formats.pdb.PDBTrajectoryFile._loadNameReplacementTables`.
    """
    tree = ElementTree.parse(
        os.path.join(os.path.dirname(__file__), "pdbNames.xml"),
    )
    # Residue and atom replacements
    residue_name_replacements = {}
    atom_name_replacements = {}
    # ... containers for residues
    all_residues, protein_residues, nucleic_acid_residues = {}, {}, {}
    for residue in tree.getroot().findall("Residue"):
        name = residue.attrib["name"]
        if name == "All":
            _parse_residue(residue, all_residues)
        elif name == "Protein":
            _parse_residue(residue, protein_residues)
        elif name == "Nucleic":
            _parse_residue(residue, nucleic_acid_residues)
    for atom in all_residues:
        protein_residues[atom] = all_residues[atom]
        nucleic_acid_residues[atom] = all_residues[atom]
    for residue in tree.getroot().findall("Residue"):
        name = residue.attrib["name"]
        for id in residue.attrib:
            if id == "name" or id.startswith("alt"):
                residue_name_replacements[residue.attrib[id]] = name
        if "type" not in residue.attrib:
            atoms = copy(all_residues)
        elif residue.attrib["type"] == "Protein":
            atoms = copy(protein_residues)
        elif residue.attrib["type"] == "Nucleic":
            atoms = copy(nucleic_acid_residues)
        else:
            atoms = copy(all_residues)
        _parse_residue(residue, atoms)
        atom_name_replacements[name] = atoms
    return residue_name_replacements, atom_name_replacements


def _parse_residue(residue, map):
    """Closely follows `mdtraj.formats.pdb.PDBTrajectoryFile._parseResidueAtoms`."""
    for atom in residue.findall("Atom"):
        name = atom.attrib["name"]
        for id in atom.attrib:
            map[atom.attrib[id]] = name
