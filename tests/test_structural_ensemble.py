import pytest

import cryojax.simulator as cxs
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from cryojax.io import read_array_from_mrc, read_atoms_from_pdb
from cryojax.simulator import DiscreteStructuralEnsemble


@pytest.fixture
def voxel_potential(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_from_mrc(sample_mrc_path, loads_spacing=True)
    return cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=1.3
    )


@pytest.fixture
def atom_potential(sample_pdb_path):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
        loads_b_factors=True,
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )
    return cxs.PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
        b_factors=b_factors,
    )


@pytest.mark.parametrize(
    "potential",
    [("voxel_potential"), ("atom_potential")],
)
def test_conformation(potential, request):
    potential = request.getfixturevalue(potential)
    pose = cxs.EulerAnglePose()
    conformational_space = tuple([potential for _ in range(3)])
    structure = DiscreteStructuralEnsemble(conformational_space, pose, conformation=0)
    _ = structure.get_potential_in_transformed_frame()
