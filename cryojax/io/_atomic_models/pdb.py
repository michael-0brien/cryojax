"""
Read atomic information from a PDB file. Functions and objects are
adapted from `mdtraj`.
"""

import os
import pathlib
import typing
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

from ...atom_util import center_atom_positions


if hasattr(typing, "GENERATING_DOCUMENTATION"):
    AtomProperties = dict  # pyright: ignore[reportAssignmentType]
else:

    class AtomProperties(TypedDict):
        masses: Float[np.ndarray, "... n_atoms"]
        b_factors: Float[np.ndarray, "... n_atoms"]
        charges: Float[np.ndarray, "... n_atoms"]


@overload
def read_atoms_from_pdb(
    filename: str | pathlib.Path,
    *,
    loads_properties: Literal[False],
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    stack_models: bool = False,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(  # type: ignore
    filename: str | pathlib.Path,
    *,
    loads_properties: Literal[True],
    loads_b_factors: bool = False,
    center: bool = True,
    selection_string: str = "all",
    model_index: int | None = None,
    stack_models: bool = False,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
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
    stack_models: bool = False,
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
    stack_models: bool = False,
    standardizes_names: bool = True,
    topology: mdtraj.Topology | None = None,
) -> (
    tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, "... n_atoms"]]
    | tuple[
        Float[np.ndarray, "... n_atoms 3"],
        Int[np.ndarray, " n_atoms"],
        AtomProperties | np.ndarray,  # Included for `loads_b_factors=True`
    ]
):
    """Load relevant atomic information for simulating cryo-EM
    images from a PDB or mmCIF file. This function wraps the function
    `mmdf_to_atoms`.

    !!! info

        The `selection_string` argument enables usage of
        [`mdtraj`](https://www.mdtraj.org/) atom selection syntax.

    **Arguments:**

    - `filename`:
        The name of the PDB/PDBx file to open.
    - `center`:
        If `True`, center the model so that its center of mass coincides
        with the origin.
    - `loads_properties`:
        If `True`, return a dictionary of the atom properties.
    - `selection_string`:
        A selection string in `mdtraj`'s format.
    - `model_index`:
        An optional index for grabbing a particular model stored in the PDB. If `None`,
        grab all models, where `atom_positions` has a leading dimension for the model
        if `stack_models = True` or concatenates all models if `stack_models = False`.
    - `stack_models`:
        If `True`, `model_index = None`, and there are multiple models in the PDB,
        assume that each model is of the same protein and return atom positions
        and properties with a stacked leading dimension.
    - `standardizes_names`:
        If `True`, non-standard atom names and residue names are standardized.
        If set to `False`, this step is skipped.
    - `topology`:
        If `None`, use the function `mmdf_to_topology` to build a topology
        on-the-fly. If `stack_models = True`, `model_index = None`, and there
        are multiple models in the PDB, use the first model index to build the
        topology.

    **Returns:**

    A tuple whose first element is a `numpy` array of coordinates containing
    atomic positions, and whose second element is an array of atomic
    numbers. To be clear,

    ```python
    atom_positons, atomic_numbers = read_atoms_from_pdb(...)
    ```

    !!! info

        If your PDB has multiple models, arrays such as the
        atom positions are loaded with a
        leading dimension for each model. To load a single
        model at index 0,

        ```python
        atom_positons, atomic_numbers = read_atoms_from_pdb(..., model_index=0)
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
        stack_models=stack_models,
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
    stack_models: bool = False,
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
    stack_models: bool = False,
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
    stack_models: bool = False,
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
    stack_models: bool = False,
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
    atom_info = _load_atom_info(df, model_index=model_index, stack_models=stack_models)
    if selection_string != "all":
        if topology is None:
            if model_index is None and stack_models:
                topology = mmdf_to_topology(df, standardizes_names, model_index=0)
            else:
                topology = mmdf_to_topology(
                    df, standardizes_names, model_index=model_index
                )
        # Filter atoms and grab atomic positions and numbers
        selected_indices = topology.select(selection_string)
        atom_positions = atom_info["positions"][:, selected_indices]
        atomic_numbers = atom_info["numbers"][selected_indices]
        atom_properties = jax.tree.map(
            lambda arr: arr[:, selected_indices], atom_info["properties"]
        )
    else:
        atom_positions = atom_info["positions"]
        atomic_numbers = atom_info["numbers"]
        atom_properties = atom_info["properties"]
    # Center by mass
    if center:
        atom_masses = atom_properties["masses"]
        atom_positions = center_atom_positions(atom_positions, atom_masses, atom_axis=1)
    # Return, without leading dimensions if there is only one structure
    atom_positions = atom_positions[0] if atom_positions.shape[0] == 1 else atom_positions
    if loads_properties or loads_b_factors:
        # Optionally return atom properties
        atom_properties = jax.tree.map(
            lambda arr: (arr[0] if arr.shape[0] == 1 else arr), atom_properties
        )
        if loads_b_factors:
            warnings.warn(
                "`loads_b_factor` option is deprecated and will be removed in "
                "cryoJAX 0.6.0. Use `loads_properties` instead.",
                category=FutureWarning,
                stacklevel=2,
            )
            return atom_positions, atomic_numbers, atom_properties["b_factors"]
        else:
            return atom_positions, atomic_numbers, atom_properties
    else:
        return atom_positions, atomic_numbers


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
        indices are captured in `df["model"]`. If `None`, use all
        models to build a single topology.

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
    df_at_model = df if model_index is None else df[df["model"] == model_index]
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
        Since we use `mmdf` to parse the PDB/PDBx file, the
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
        indices are captured in `df["model"]`.

    **Returns:**

    An `mdtraj.Topology` object.
    """
    df = mmdf.read(pathlib.Path(filename))
    return mmdf_to_topology(df, standardizes_names, model_index)


class _AtomicModelInfo(TypedDict):
    positions: Float[np.ndarray, "M N 3"]
    numbers: Int[np.ndarray, "M N 3"]
    properties: AtomProperties


def _load_atom_info(df: pd.DataFrame, model_index: int | None, stack_models: bool):
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
    if model_index is None and stack_models:
        model_numbers = df["model"].unique().tolist()
    else:
        model_numbers = [None]
    atom_positions, atomic_numbers = [], []
    atom_masses, b_factors, charges = [], [], []
    for index in model_numbers:
        df_at_model = df if index is None else df[df["model"] == index]
        atom_positions.append(df_at_model[["x", "y", "z"]].to_numpy())
        atomic_numbers.append(df_at_model["atomic_number"].to_numpy())
        atom_masses.append(df_at_model["atomic_weight"].to_numpy())
        b_factors.append(df_at_model["b_isotropic"].to_numpy())
        charges.append(df_at_model["charge"].to_numpy())

    if len(atomic_numbers) > 1:
        if not all(np.array_equal(arr, atomic_numbers[0]) for arr in atomic_numbers):
            raise ValueError(
                "Tried to load multiple models with `stack_models = True`, "
                "but found that atomic numbers across different models "
                "were different. Only use `stack_models = True` if "
                "each model represents the same protein."
            )

    # Gather atom info and return
    properties = AtomProperties(
        charges=np.asarray(charges, dtype=int),
        b_factors=np.asarray(b_factors, dtype=float),
        masses=np.asarray(atom_masses, dtype=float),
    )
    atom_info = _AtomicModelInfo(
        positions=np.asarray(atom_positions, dtype=float),
        numbers=np.asarray(atomic_numbers[0], dtype=int),
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
