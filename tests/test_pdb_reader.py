import os

import numpy as np
import pytest
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.io import read_atoms_from_pdb


@pytest.fixture
def pdb_multiple_structures_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao_assembly.pdb")


def test_read_structure(sample_pdb_path):
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="protein and not element H",
        loads_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_types.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_types.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_structure_no_b_factors(sample_pdb_path):
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="protein and not element H",
        loads_b_factors=False,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atom_types.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_pdb_multiple_structures(pdb_multiple_structures_path):
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_b_factors=True,
        selection_string="all",
    )
    assert atom_positions.ndim == 3
    assert atom_positions.shape[0] == 10
    assert atom_positions.shape[2] == 3
    assert atom_positions.shape[1] == 138
    assert atom_types.shape == b_factors.shape == atom_positions.shape[0:2]


def test_read_pdb_at_structure(pdb_multiple_structures_path):
    atom_positions, atom_types = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_b_factors=False,
        selection_string="all",
        model_index=1,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atom_types.shape[0]
    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 138


def test_read_cif(sample_cif_path):
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_cif_path,
        center=True,
        selection_string="all",
        model_index=None,
        loads_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_types.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_types.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 3222


def test_center_waterbox(sample_waterbox_pdb):
    atom_positions, _ = read_atoms_from_pdb(
        sample_waterbox_pdb,
        center=True,
        selection_string="all",
    )

    assert not np.isnan(atom_positions).any(), "Centering resulted in positions with NaNs"
