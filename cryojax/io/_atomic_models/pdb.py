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
    """Read atomic information from a PDB file. This object
    wraps the `cryojax.io.AtomicModelReader` class into a function
    interface to accomodate most use cases in cryo-EM.

    **Arguments:**

    - `filename_or_url`:
        The name of the PDB/mmCIF file to open. Can be a URL.
    - `center`:
        If `True`, center the model so that its center of mass coincides
        with the origin.
    - `loads_b_factors`:
        If `True`, return the B-factors of the atoms.
    - `selection_string`:
        A selection string in `mdtraj`'s format. See `mdtraj` for documentation.
    - `model_index`:
        An optional index for grabbing a particular model stored in the PDB. If `None`,
        grab all models, where `atom_positions` has a leading dimension for the model.
    - `standardizes_names`:
        If `True`, non-standard atom names and residue names are standardized to conform
        with the current PDB format version. If set to `False`, this step is skipped.
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

        If your PDB has multiple models, `atom_positions` by
        default with a leading dimension that indexes each model.
        On the other hand, `atom_types` (and `b_factors`, if loaded)
        do not have this leading dimension and are constant across
        models.
    """
    # Load `mmdf` dataframe
    df = mmdf.read(pathlib.Path(filename))
    atom_info, topology = _load_atom_info(
        df,
        standardizes_names=standardizes_names,
        topology=topology,
        model_index=model_index,
        loads_masses=center,
        loads_b_factors=loads_b_factors,
    )
    # Filter atoms and grab positions and identities
    selected_indices = topology.select(selection_string)
    atom_positions = atom_info["positions"][:, selected_indices]
    atom_properties = jax.tree.map(
        lambda arr: arr[selected_indices], atom_info["properties"]
    )
    atom_types = atom_properties["identities"]
    # Center by mass
    if center:
        atom_masses = cast(np.ndarray, atom_properties["masses"])
        atom_positions = _center_atom_coordinates(atom_positions, atom_masses)
    # Return, optionality with b-factors and without a leading dimension for the
    # positions if there is only one structure
    if atom_positions.shape[0] == 1:
        atom_positions = np.squeeze(atom_positions, axis=0)
    if loads_b_factors:
        b_factors = cast(np.ndarray, atom_properties["b_factors"])
        return atom_positions, atom_types, b_factors
    else:
        return atom_positions, atom_types


def _center_atom_coordinates(atom_positions, atom_masses):
    com_position = np.transpose(atom_positions, axes=[0, 2, 1]).dot(
        atom_masses / atom_masses.sum()
    )
    return atom_positions - com_position[:, None, :]


class AtomProperties(TypedDict):
    identities: Int[np.ndarray, " N"]
    masses: Float[np.ndarray, " N"] | None
    b_factors: Float[np.ndarray, " N"] | None


class AtomicModelInfo(TypedDict):
    positions: Float[np.ndarray, "M N 3"]
    properties: AtomProperties


def _make_topology(
    df: pd.DataFrame,
    atom_positions: list,
    standardizes_names: bool,
    model_index: int | None,
) -> Topology:
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
        chain_id = df_at_model["chain"].iloc[atom_index]
        residue_name = df_at_model["residue"].iloc[atom_index]
        residue_id = df_at_model["residue_id"].iloc[atom_index]
        c = topology.add_chain(chain_id)
        if residue_name in residue_name_replacements and standardizes_names:
            residue_name = residue_name_replacements[residue_name]
        # TODO: is it necessary to have `segment_id`, as is parsed in `mdtraj`?
        r = topology.add_residue(residue_name, c, residue_id, segment_id="")
        if residue_name in atom_name_replacements and standardizes_names:
            atom_replacements = atom_name_replacements[residue_name]
        else:
            atom_replacements = {}
        atom_name = df_at_model["atom"].iloc[atom_index]
        if atom_name in atom_replacements:
            atom_name = atom_replacements[atom_name]
        atom_name = atom_name.strip()
        residue_ids = df_at_model[df_at_model["residue_id"] == residue_id]
        element = _guess_element(
            atom_name,
            residue_name,
            residue_length=len(residue_ids),
        )
        charge = df_at_model["charge"].iloc[atom_index]
        # TODO: ok to remove serial number?
        _ = topology.add_atom(
            atom_name,
            element,  # type: ignore
            r,
            # serial=atom.serial_number,
            formal_charge=charge,
        )

    topology.create_standard_bonds()
    topology.create_disulfide_bonds(atom_positions[0])

    return topology


def _load_atom_info(
    df: pd.DataFrame,
    topology: Topology | None,
    model_index: int | None,
    standardizes_names: bool,
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
    atom_positions = []
    atomic_numbers, atomic_mass, b_factors = None, None, None
    for index, model_index in enumerate(model_numbers):
        df_at_index = df[df["model"] == model_index]
        atom_positions.append(df_at_index[["x", "y", "z"]].to_numpy())
        if index == 0:
            # Assume atom properties don't change between models
            atomic_numbers = df_at_index["atomic_number"].to_numpy()
            if loads_masses:
                atomic_mass = df_at_index["atomic_weight"].to_numpy()
            if loads_b_factors:
                b_factors = df_at_index["b_isotropic"].to_numpy()
    assert atomic_numbers is not None
    if loads_masses:
        assert atomic_mass is not None
    if loads_b_factors:
        assert b_factors is not None

    # Load the topology if None is given
    if topology is None:
        topology = _make_topology(df, atom_positions, standardizes_names, model_index)

    # Gather atom info and return
    properties = AtomProperties(
        identities=np.asarray(atomic_numbers, dtype=int),
        b_factors=(b_factors if loads_b_factors else None),
        masses=(atomic_mass if loads_masses else None),
    )
    atom_info = AtomicModelInfo(
        positions=np.asarray(atom_positions, dtype=float),
        properties=properties,
    )

    return atom_info, topology


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
