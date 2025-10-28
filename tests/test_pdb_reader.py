import os

import jax
import mdtraj
import mmdf
import numpy as np
import pandas as pd
import pytest
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.io import mmdf_to_atoms, read_atoms_from_pdb


@pytest.fixture
def pdb_multiple_structures_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao_assembly.pdb")


def test_read_structure(sample_pdb_path):
    atom_positions, atomic_numbers, atom_properties = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="protein and not element H",
        loads_properties=True,
    )

    assert atom_positions.ndim == 2
    assert jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda z: z.shape == atomic_numbers.shape, atom_properties),
    )
    assert atom_positions.shape[0] == atomic_numbers.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_structure_no_properties(sample_pdb_path):
    atom_positions, atomic_numbers = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="protein and not element H",
        loads_properties=False,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atomic_numbers.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 77


def test_read_pdb_multiple_structures_stack(pdb_multiple_structures_path):
    atom_positions, atomic_numbers, atom_properties = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_properties=True,
        selection_string="all",
        model_index=None,
        stack_models=True,
    )
    assert atom_positions.ndim == 3
    assert atom_positions.shape[0] == 10
    assert atom_positions.shape[2] == 3
    assert atom_positions.shape[1] == 138
    assert jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda z: z.shape == atom_positions.shape[0:2], atom_properties),
    )
    assert atomic_numbers.shape == (138,)


def test_read_pdb_multiple_structures_flat(pdb_multiple_structures_path):
    atom_positions, atomic_numbers, atom_properties = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_properties=True,
        selection_string="all",
        model_index=None,
        stack_models=False,
    )
    assert atom_positions.ndim == 2
    assert atomic_numbers.shape == (10 * 138,)
    assert atom_positions.shape == (10 * 138, 3)
    assert jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda z: z.shape == atomic_numbers.shape, atom_properties),
    )


def test_bad_read_pdb_multiple_structures(pdb_multiple_structures_path):
    df = mmdf.read(pdb_multiple_structures_path)
    last_row = df.iloc[[-1]]
    df = pd.concat([df, last_row], ignore_index=True)
    with pytest.raises(ValueError):
        _ = mmdf_to_atoms(
            df,
            center=True,
            loads_properties=True,
            selection_string="all",
            model_index=None,
            stack_models=True,
        )


def test_read_pdb_at_structure(pdb_multiple_structures_path):
    atom_positions, atomic_numbers = read_atoms_from_pdb(
        pdb_multiple_structures_path,
        center=True,
        loads_b_factors=False,
        selection_string="all",
        model_index=1,
    )

    assert atom_positions.ndim == 2
    assert atom_positions.shape[0] == atomic_numbers.shape[0]
    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 138


def test_read_cif(sample_cif_path):
    atom_positions, atomic_numbers, atom_properties = read_atoms_from_pdb(
        sample_cif_path,
        center=True,
        selection_string="all",
        model_index=None,
        loads_properties=True,
    )

    assert atom_positions.ndim == 2
    assert jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda z: z.shape == atomic_numbers.shape, atom_properties),
    )
    assert atom_positions.shape[0] == atomic_numbers.shape[0]

    assert atom_positions.shape[1] == 3
    assert atom_positions.shape[0] == 3222


def test_center_waterbox(sample_waterbox_pdb):
    atom_positions, _ = read_atoms_from_pdb(
        sample_waterbox_pdb,
        center=True,
        selection_string="all",
    )

    assert not np.isnan(atom_positions).any(), "Centering resulted in positions with NaNs"


@pytest.mark.parametrize("pdbfile", ["sample_pdb_path", "sample_waterbox_pdb"])
@pytest.mark.parametrize("center", [True, False])
def test_consistency_with_mdtraj(pdbfile, center, request):
    pdbfile = request.getfixturevalue(pdbfile)
    atom_positions, _ = read_atoms_from_pdb(
        pdbfile,
        center=center,
        selection_string="all",
    )

    import warnings

    warnings.filterwarnings("ignore")

    traj = mdtraj.load(pdbfile)
    if center:
        traj.center_coordinates(mass_weighted=True)

    mdtraj_positions = traj.xyz[0] * 10.0  # convert to Angstrom

    distance_matrix = np.linalg.norm(
        atom_positions[:, None, :] - mdtraj_positions[None, :, :], axis=-1
    )
    np.testing.assert_allclose(distance_matrix.min(axis=1), 0.0, atol=1e-4, rtol=1e-4)
    unique_min_indices = np.unique(distance_matrix.argmin(axis=1))
    assert unique_min_indices.shape[0] == atom_positions.shape[0], (
        "Atom position loading is not consistent with MDTraj up to permutation"
    )
