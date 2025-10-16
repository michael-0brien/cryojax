"""
Read atomic information from a PDB file. Functions and objects are
adapted from `mdtraj`.
"""

import os
import pathlib
import warnings
from copy import copy
from typing import Literal, TypedDict, overload
from xml.etree import ElementTree

import jax
import mdtraj
import mmdf
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from mdtraj.core import element as elem
from mdtraj.core.topology import Topology


class AtomProperties(TypedDict):
    masses: Float[np.ndarray, "... n_atoms"]
    b_factors: Float[np.ndarray, "... n_atoms"]
    charges: Float[np.ndarray, "... n_atoms"]


class AtomPropertiesByType(TypedDict):
    masses: tuple[Float[np.ndarray, "... _"], ...]
    b_factors: tuple[Float[np.ndarray, "... _"], ...]
    charges: tuple[Float[np.ndarray, "... _"], ...]


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_properties: Literal[False],
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]: ...


@overload
def read_atoms_from_pdb(  # type: ignore
    filename: str | pathlib.Path,
    *,
    loads_properties: Literal[True],
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"],
    Int[np.ndarray, "... n_atoms"],
    AtomProperties,
]: ...


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_properties: bool = False,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]: ...


def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_properties: bool = False,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> (
    tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]
    | tuple[
        Float[np.ndarray, "... n_atoms 3"],
        Int[np.ndarray, "... n_atoms"],
        AtomProperties | np.ndarray,  # Included for `loads_b_factors=True`
    ]
):
    """Load relevant atomic information for simulating cryo-EM
    images from a PDB or mmCIF file. This function wraps the function
    `mmdf_to_atoms`.

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
    - `loads_properties`:
        If `True`, return a dictionary of the atom properties.
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
    atom_positons, atom_type = read_atoms_from_pdb(...)
    ```

    !!! info

        If your PDB has multiple models, arrays such as the
        atom positions are loaded with a
        leading dimension for each model. To load a single
        model at index 0,

        ```python
        atom_positons, atom_type = read_atoms_from_pdb(..., model_index=0)
        ```
    """
    # Load `mmdf` dataframe forward the `mmdf_to_atoms` method
    df = mmdf.read(pathlib.Path(filename))
    return mmdf_to_atoms(
        df,
        loads_properties=loads_properties,
        loads_b_factors=loads_b_factors,
        center=center,
        selection_string=selection_string,
        model_index=model_index,
        standardizes_names=standardizes_names,
        topology=topology,
    )


@overload
def mmdf_to_atoms(
    df: pd.DataFrame,
    *,
    loads_properties: Literal[False],
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]: ...


@overload
def mmdf_to_atoms(  # type: ignore
    df: pd.DataFrame,
    *,
    loads_properties: Literal[True],
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"],
    Int[np.ndarray, "... n_atoms"],
    AtomProperties,
]: ...


@overload
def mmdf_to_atoms(
    df: pd.DataFrame,
    *,
    loads_properties: bool = False,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]: ...


def mmdf_to_atoms(
    df: pd.DataFrame,
    *,
    loads_properties: bool = False,
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> (
    tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]
    | tuple[
        Float[np.ndarray, "... n_atoms 3"],
        Int[np.ndarray, "... n_atoms"],
        AtomProperties | np.ndarray,
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
    atom_info = _load_atom_info(df, model_index=model_index)
    if selection_string != "all":
        if topology is None:
            topology = mmdf_to_topology(df, standardizes_names, model_index)
        # Filter atoms and grab atomic positions and numbers
        selected_indices = topology.select(selection_string)
        atom_positions = atom_info["positions"][:, selected_indices]
        atom_type = atom_info["numbers"][:, selected_indices]
        atom_properties = jax.tree.map(
            lambda arr: arr[:, selected_indices], atom_info["properties"]
        )
    else:
        atom_positions = atom_info["positions"]
        atom_type = atom_info["numbers"]
        atom_properties = atom_info["properties"]
    # Center by mass
    if center:
        atom_masses = atom_properties["masses"]
        atom_positions = _center_atom_coordinates(atom_positions, atom_masses)
    # Return, without leading dimensions if there is only one structure
    atom_positions = atom_positions[0] if atom_positions.shape[0] == 1 else atom_positions
    atom_type = atom_type[0] if atom_type.shape[0] == 1 else atom_type
    if loads_properties or loads_b_factors:
        # Optionally return atom properties
        atom_properties = jax.tree.map(
            lambda arr: (arr[0] if arr.shape[0] == 1 else arr), atom_properties
        )
        if loads_b_factors:
            warnings.warn(
                "`loads_b_factor` option is deprecated and will be removed in "
                "cryoJAX 0.6.0. Use `loads_properties` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return atom_positions, atom_type, atom_properties["b_factors"]
        else:
            return atom_positions, atom_type, atom_properties
    else:
        return atom_positions, atom_type


def mmdf_to_topology(
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
        element = elem.Element.getByAtomicNumber(df_at_index["atomic_number"])
        charges = df_at_index["charge"]
        # TODO: ok to remove serial number?
        _ = topology.add_atom(
            atom_name,
            element,
            r,
            serial=atom_index,  # atom.serial_number,
            formal_charge=charges,
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
    """Generate an `mdtraj.Topology` from a PDB or mmCIF file.
    This function wraps the function `mmdf_to_topology`.

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
    return mmdf_to_topology(df, standardizes_names, model_index)


@overload
def split_atoms_by_number(
    atom_positions: Float[np.ndarray, "... n_atoms 3"],
    atom_type: Int[np.ndarray, " n_atoms"],
    atom_properties: dict | AtomProperties,
) -> tuple[
    tuple[Float[np.ndarray, "... _ 3"], ...], tuple[int, ...], AtomPropertiesByType
]: ...


@overload
def split_atoms_by_number(
    atom_positions: Float[np.ndarray, "... n_atoms 3"],
    atom_type: Int[np.ndarray, " n_atoms"],
    atom_properties: None,
) -> tuple[tuple[Float[np.ndarray, "... _ 3"], ...], tuple[int, ...]]: ...


def split_atoms_by_number(
    atom_positions: Float[np.ndarray, "... n_atoms 3"],
    atom_type: Int[np.ndarray, " n_atoms"],
    atom_properties: dict | AtomProperties | None = None,
) -> (
    tuple[tuple[Float[np.ndarray, "... _ 3"], ...], tuple[int, ...]]
    | tuple[
        tuple[Float[np.ndarray, "... _ 3"], ...], tuple[int, ...], AtomPropertiesByType
    ]
):
    """Given atom positions and atomic numbers, split
    atom positions into a tuple where each element
    is atom positions for a given atomic number.

    **Arguments:**

    - `atom_positions`:
        An array of atom positions, optionally with a batch
        dimension.
    - `atom_type`:
        Atomic numbers corresponding to `atom_positions`.
    - `atom_properties`:
        Optionally, include a dictionary of atom properties.
        Its arrays must be properties that are quantified by a
        single number and have the same batch dimension as
        `atom_positions`.


    **Returns:**

    A tuple of atom positions for a given atom type, a tuple
    of atom types, and optionally a dictionary with the
    same keys as `atom_properties` but values also a tuple split
    by the atom types.
    """
    pass


def _center_atom_coordinates(atom_positions, atom_masses):
    com_position = (
        np.sum(atom_positions * atom_masses[..., None], axis=1)
        / atom_masses.sum(axis=1)[:, None]
    )
    return atom_positions - com_position[:, None, :]


class _AtomicModelInfo(TypedDict):
    positions: Float[np.ndarray, "M N 3"]
    numbers: Int[np.ndarray, "M N 3"]
    properties: AtomProperties


def _load_atom_info(df: pd.DataFrame, model_index: int | None):
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
    atom_positions, atom_type, atomic_masses, b_factors, charges = [], [], [], [], []
    for model_index in model_numbers:
        df_at_index = df[df["model"] == model_index]
        atom_positions.append(df_at_index[["x", "y", "z"]].to_numpy())
        atom_type.append(df_at_index["atomic_number"].to_numpy())
        atomic_masses.append(df_at_index["atomic_weight"].to_numpy())
        b_factors.append(df_at_index["b_isotropic"].to_numpy())
        charges.append(df_at_index["charge"].to_numpy())

    # Gather atom info and return
    properties = AtomProperties(
        charges=np.asarray(charges, dtype=int),
        b_factors=np.asarray(b_factors, dtype=float),
        masses=np.asarray(atomic_masses, dtype=float),
    )
    atom_info = _AtomicModelInfo(
        positions=np.asarray(atom_positions, dtype=float),
        numbers=np.asarray(atom_type, dtype=int),
        properties=properties,
    )

    return atom_info


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
