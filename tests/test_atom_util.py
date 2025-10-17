from typing import cast

import jax
import mdtraj
import mmdf
import numpy as np
import pytest
from cryojax.atom_util import center_atom_positions, split_atoms_by_element
from cryojax.io import mmdf_to_atoms, mmdf_to_topology, read_atoms_from_pdb


@pytest.mark.parametrize("mass_weighted, atom_axis", ([True, 0], [True, 1], [False, 0]))
def test_center_positions(sample_pdb_path, mass_weighted, atom_axis):
    # No batch dimension
    df = mmdf.read(sample_pdb_path)
    topology = mmdf_to_topology(df)
    atom_positions, _, atom_properties = mmdf_to_atoms(
        df, topology=topology, loads_properties=True
    )
    if atom_axis > 0:
        atom_positions, atom_properties = jax.tree.map(
            lambda x: np.expand_dims(x, axis=tuple(range(atom_axis))),
            (atom_positions, atom_properties),
        )
    # MDTraj centering
    traj = mdtraj.Trajectory(atom_positions / 10, topology)
    traj.center_coordinates(mass_weighted=mass_weighted)
    # cryoJAX centering
    cx_xyz = center_atom_positions(
        atom_positions,
        atom_masses=(atom_properties["masses"] if mass_weighted else None),
        atom_axis=atom_axis,
    )
    traj_xyz = np.expand_dims(cast(np.ndarray, traj.xyz)[0], axis=tuple(range(atom_axis)))

    np.testing.assert_allclose(cx_xyz, traj_xyz * 10, atol=1e-5)


@pytest.mark.parametrize("atom_axis", [1, 2])
def test_atom_splitting(sample_pdb_path, atom_axis):
    # No batch dimension
    atom_positions, atom_elements = read_atoms_from_pdb(
        sample_pdb_path, selection_string="element C || element O || element N"
    )
    positions_by_element, elements = split_atoms_by_element(
        atom_elements, atom_positions, atom_axis=0
    )
    assert isinstance(positions_by_element, tuple)
    assert len(positions_by_element) == 3
    assert len(elements) == 3

    batched_positions_by_element, _ = split_atoms_by_element(
        atom_elements,
        np.expand_dims(atom_positions, axis=tuple(range(atom_axis))),
        atom_axis=atom_axis,
    )
    assert all(
        pos.shape == bpos.shape[atom_axis:]
        for pos, bpos in zip(positions_by_element, batched_positions_by_element)
    )
