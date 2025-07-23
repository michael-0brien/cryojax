"""
Read atomic information from a PDB file. Functions and objects are
adapted from `mdtraj`.
"""

import bz2
import gzip
import io
import os
import pathlib
from copy import copy
from io import StringIO
from typing import Literal, Optional, TypedDict, cast, overload
from urllib.parse import urlparse, uses_netloc, uses_params, uses_relative
from urllib.request import urlopen
from xml.etree import ElementTree

import jax
import mdtraj
import numpy as np
from jaxtyping import Float, Int
from mdtraj.core import element as elem
from mdtraj.core.topology import Topology
from mdtraj.formats.pdb.pdbstructure import PdbStructure


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    loads_b_factors: Literal[False],
    center: bool = True,
    *,
    selection_string: str = "all",
    model_index: Optional[int] = None,
    standardizes_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> tuple[Float[np.ndarray, "... n_atoms 3"], Int[np.ndarray, " n_atoms"]]: ...


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    loads_b_factors: Literal[True],
    center: bool = True,
    *,
    selection_string: str = "all",
    model_index: Optional[int] = None,
    standardizes_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
) -> tuple[
    Float[np.ndarray, "... n_atoms 3"],
    Int[np.ndarray, " n_atoms"],
    Float[np.ndarray, " n_atoms"],
]: ...


@overload
def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    loads_b_factors: bool = False,
    center: bool = True,
    *,
    selection_string: str,
    model_index: Optional[int] = None,
    standardizes_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
): ...


def read_atoms_from_pdb(
    filename_or_url: str | pathlib.Path,
    loads_b_factors: bool = False,
    center: bool = True,
    *,
    selection_string: str = "all",
    model_index: Optional[int] = None,
    standardizes_names: bool = True,
    topology: Optional[mdtraj.Topology] = None,
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
    atom_positons, atom_identities = read_atoms_from_pdb(...)
    ```

    !!! info

        If your PDB has multiple models, `atom_positions` by
        default with a leading dimension that indexes each model.
        On the other hand, `atom_identities` (and `b_factors`, if loaded)
        do not have this leading dimension and are constant across
        models.
    """
    # Load all atoms
    with AtomicModelFile(filename_or_url) as pdb_file:
        # Read file
        atom_info, topology = pdb_file.read_atoms(
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
    atom_identities = atom_properties["identities"]
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
        return atom_positions, atom_identities, b_factors
    else:
        return atom_positions, atom_identities


def _center_atom_coordinates(atom_positions, atom_masses):
    com_position = np.transpose(atom_positions, axes=[0, 2, 1]).dot(
        atom_masses / atom_masses.sum()
    )
    return atom_positions - com_position[:, None, :]


class AtomProperties(TypedDict):
    """A struct for the info of individual atoms.

    **Attributes:**

    - `identities`: The atomic numbers of all of the atoms.
    - `masses`: The mass of each atom.
    - `b_factors`: The B-factors of all of the atoms.
    """

    identities: Int[np.ndarray, " N"]
    masses: Optional[Float[np.ndarray, " N"]]
    b_factors: Optional[Float[np.ndarray, " N"]]


class AtomicModelInfo(TypedDict):
    """A struct for the info of individual atoms.

    **Keys:**

    - `positions`:
        The cartesian coordinates of all of the atoms.
    - `properties`:
        The static properties of the individual atom, i.e.
        that do not change between frames.
    """

    positions: Float[np.ndarray, "M N 3"]
    properties: AtomProperties


class AtomicModelFile:
    """A PDB file loader that loads the necessary information for
    cryo-EM. This object is based on the `PDBTrajectoryFile`
    from `mdtraj`.
    """

    def __init__(self, filename_or_url: str | pathlib.Path):
        """**Arguments:**

        - `filename_or_url`:
            The name of the PDB file to open. Can be a URL.
        """
        # Set field that says we are reading the file
        self._is_open = True
        # Setup I/O
        if _is_url(filename_or_url):
            filename_or_url = str(filename_or_url)
            file_obj = urlopen(filename_or_url)
            if filename_or_url.lower().endswith(".gz"):
                file_obj = gzip.GzipFile(fileobj=file_obj)
            self._file = StringIO(file_obj.read().decode("utf-8"))

        else:
            filename_or_url = pathlib.Path(filename_or_url)
            _validate_pdb_file(filename_or_url)
            self._file = _open_maybe_zipped(filename_or_url, "r")
        # Create the PDB structure via `mdtraj`
        self._pdb_structure = PdbStructure(self._file, load_all_models=True)

    @property
    def pdb_structure(self) -> PdbStructure:
        return self._pdb_structure

    def read_atoms(
        self,
        *,
        standardizes_names: bool = True,
        topology: Optional[Topology] = None,
        model_index: Optional[int] = None,
        loads_b_factors: bool = True,
        loads_masses: bool = True,
    ) -> tuple[AtomicModelInfo, Topology]:
        """Load properties from the PDB reader.

        **Arguments:**

        - `standardizes_names`:
            If `True`, non-standard atom names and residue names are standardized to
            conform with the current PDB format version. If `False`, this step is skipped.
        - `topology`:
            If the `topology` is passed as input, it won't be parsed from the PDB file.
            This saves time if you have to parse a big number of files.

        **Returns:**

        A tuple of the `AtomicModelInfo` dataclass and the `mdtraj.Topology`.
        """
        atom_info, topology = _load_atom_info(
            self._pdb_structure,
            topology,
            model_index,
            standardizes_names,
            loads_b_factors,
            loads_masses,
        )

        return atom_info, topology

    @property
    def is_open(self):
        return self._is_open

    def close(self):
        """Close the PDB file"""
        if self._is_open:
            if hasattr(self, "_file"):
                self._file.close()
        self._is_open = False

    def __del__(self):
        self.close()
        del self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _make_topology(
    pdb: PdbStructure,
    atom_positions: list,
    standardizes_names: bool,
    model_index: Optional[int],
) -> Topology:
    topology = Topology()
    if standardizes_names:
        residue_name_replacements, atom_name_replacements = (
            _load_name_replacement_tables()
        )
    else:
        residue_name_replacements, atom_name_replacements = {}, {}
    atom_by_number = {}
    if model_index is None:
        model_index = 0
    model = pdb.models_by_number[model_index]
    for chain in model.iter_chains():
        c = topology.add_chain(chain.chain_id)
        for residue in chain.iter_residues():
            residue_name = residue.get_name()
            if residue_name in residue_name_replacements and standardizes_names:
                residue_name = residue_name_replacements[residue_name]
            r = topology.add_residue(residue_name, c, residue.number, residue.segment_id)
            if residue_name in atom_name_replacements and standardizes_names:
                atom_replacements = atom_name_replacements[residue_name]
            else:
                atom_replacements = {}
            for atom in residue.atoms:
                atom_name = atom.get_name()
                if atom_name in atom_replacements:
                    atom_name = atom_replacements[atom_name]
                atom_name = atom_name.strip()
                element = atom.element
                if element is None:
                    element = _guess_element(atom_name, residue.name, len(residue))

                new_atom = topology.add_atom(
                    atom_name,
                    element,
                    r,
                    serial=atom.serial_number,
                    formal_charge=atom.formal_charge,
                )
                atom_by_number[atom.serial_number] = new_atom

    topology.create_standard_bonds()
    topology.create_disulfide_bonds(atom_positions[0])

    return topology


def _load_atom_info(
    pdb: PdbStructure,
    topology: Optional[Topology],
    model_index: Optional[int],
    standardizes_names: bool,
    loads_b_factors: bool,
    loads_masses: bool,
):
    # Load atom info
    if model_index is None:
        # ... with multiple models
        n_models = len(pdb.model_numbers())
        temp = dict(
            positions=[*([] for _ in range(n_models))],
            identities=[],
            b_factors=[],
            masses=[],
        )
        for index, model_index in enumerate(pdb.model_numbers()):  # type: ignore
            model = pdb.models_by_number[model_index]
            for chain in model.iter_chains():
                for residue in chain.iter_residues():
                    for atom in residue.atoms:
                        # ... make sure this is read in angstroms?
                        temp["positions"][index].append(atom.get_position())
                        if index == 0:
                            # Assume atom properties don't change between models
                            temp["identities"].append(atom.element.atomic_number)
                            if loads_masses:
                                temp["masses"].append(atom.element.mass)
                            if loads_b_factors:
                                temp["b_factors"].append(atom.get_temperature_factor())
    else:
        # ... with a model at one index
        n_models = 1
        temp = dict(positions=[[]], identities=[], b_factors=[], masses=[])
        try:
            model = pdb.models_by_number[model_index]
        except Exception as err:
            raise ValueError(
                "Caught exception indexing atomic model with "
                f"index {model_index}. Found that the PDB "
                f"contained model numbers {pdb.model_numbers()} "
                f"available for indexing. Traceback was:\n{err}"
            )
        for chain in model.iter_chains():
            for residue in chain.iter_residues():
                for atom in residue.atoms:
                    # ... make sure this is read in angstroms?
                    temp["positions"][0].append(atom.get_position())
                    # Assume atom properties don't change between models
                    temp["identities"].append(atom.element.atomic_number)
                    if loads_masses:
                        temp["masses"].append(atom.element.mass)
                    if loads_b_factors:
                        temp["b_factors"].append(atom.get_temperature_factor())

    # Load the topology if None is given
    if topology is None:
        topology = _make_topology(pdb, temp["positions"], standardizes_names, model_index)

    # Gather atom info and return
    properties = AtomProperties(
        identities=np.asarray(temp["identities"], dtype=int),
        b_factors=(
            np.asarray(temp["b_factors"], dtype=float) if loads_b_factors else None
        ),
        masses=(np.asarray(temp["masses"], dtype=float) if loads_masses else None),
    )
    atom_info = AtomicModelInfo(
        positions=np.asarray(temp["positions"], dtype=float),
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


def _validate_pdb_file(filename):
    if filename.suffixes not in [[".pdb"], [".pdb", ".gz"]]:  # , [".cif"]]:
        raise ValueError(
            "PDB filename must have suffix `.pdb` or `.pdb.gz`"  # , or 'cif'. "
            f"Got filename {filename}."
        )


def _is_url(url):
    """Check to see if a URL has a valid protocol.
    Originally pandas/io.common.py, sourced from `mdtraj`.
    Copyright 2014 Pandas Developers, used under the BSD licence.
    """
    try:
        return urlparse(url).scheme in _VALID_URLS
    except (AttributeError, TypeError):
        return False


def _open_maybe_zipped(filename, mode, force_overwrite=True):
    """Open a file in text (not binary) mode, transparently handling
    .gz or .bz2 compresssion, with utf-8 encoding.
    Closely follows `mdtraj.utils.open_maybe_zipped`.
    """
    _, extension = os.path.splitext(str(filename).lower())
    if mode == "r":
        if extension == ".gz":
            with gzip.GzipFile(filename, "r") as gz_f:
                return StringIO(gz_f.read().decode("utf-8"))
        elif extension == ".bz2":
            with bz2.BZ2File(filename, "r") as bz2_f:
                return StringIO(bz2_f.read().decode("utf-8"))
        else:
            return open(filename)
    elif mode == "w":
        if os.path.exists(filename) and not force_overwrite:
            raise OSError('"%s" already exists' % filename)
        if extension == ".gz":
            binary_fh = gzip.GzipFile(filename, "wb")
            return io.TextIOWrapper(binary_fh, encoding="utf-8")
        elif extension == ".bz2":
            binary_fh = bz2.BZ2File(filename, "wb")
            return io.TextIOWrapper(binary_fh, encoding="utf-8")
        else:
            return open(filename, "w")
    else:
        raise Exception(f"Internal error opening file {filename}. Invalid mode {mode}.")
