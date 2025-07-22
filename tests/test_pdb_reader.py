import os

import numpy as np
import pytest
from jax import config
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.io import read_atoms_from_pdb

config.update("jax_enable_x64", True)


@pytest.fixture
def pdb_multiple_structures_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao_assembly.pdb")


def test_read_structure(sample_pdb_path):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="protein and not element H",
        loads_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_identities.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_structure_no_b_factors(sample_pdb_path):
    atom_positions, atom_identities = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="protein and not element H",
        loads_b_factors=False,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_pdb_multiple_structures(pdb_multiple_structures_path):
    atom_positions, atom_identities = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_b_factors=False,
        selection_string="all",
    )
    assert atom_positions.ndim == 3
    assert atom_positions.shape[0] == 10
    assert atom_positions.shape[1] == atom_identities.shape[0]
    assert atom_positions.shape[2] == 3
    assert atom_positions.shape[1] == 138


def test_read_pdb_at_structure(pdb_multiple_structures_path):
    atom_positions, atom_identities = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_b_factors=False,
        selection_string="all",
        model_index=1,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atom_identities.shape[0]
    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 138


# def test_read_full_assembly_pdb(sample_pdb_path_assembly):
#     atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
#         sample_pdb_path_assembly,
#         center=True,
#         selection_string="all",
#         is_assembly=True,
#         i_model=None,
#         loads_b_factors=True,
#     )

#     assert atom_positions.ndim == 2
#     assert atom_identities.shape == b_factors.shape
#     assert atom_positions.shape[0] == atom_identities.shape[0]

#     assert atom_positions.shape[1] == 3
#     assert atom_positions.shape[0] == 1380


# def test_read_cif(sample_cif_path):
#     atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
#         sample_cif_path,
#         center=True,
#         selection_string="all",
#         is_assembly=False,
#         i_model=None,
#         loads_b_factors=True,
#     )

#     assert atom_positions.ndim == 2
#     assert atom_identities.shape == b_factors.shape
#     assert atom_positions.shape[0] == atom_identities.shape[0]

#     assert atom_positions.shape[1] == 3
#     assert atom_positions.shape[0] == 3222


def test_read_from_url(sample_pdb_url):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_url,
        center=True,
        selection_string="all",
        loads_b_factors=True,
    )

    assert atom_positions.ndim == 2
    assert atom_identities.shape == b_factors.shape
    assert atom_positions.shape[0] == atom_identities.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 1973


def test_center_waterbox(sample_waterbox_pdb):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_waterbox_pdb,
        center=True,
        selection_string="all",
        loads_b_factors=True,
    )

    assert not np.isnan(atom_positions).any(), "Centering resulted in positions with NaNs"
