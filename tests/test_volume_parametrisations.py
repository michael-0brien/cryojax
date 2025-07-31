import pytest

import cryojax.simulator as cxs
from cryojax.constants import convert_b_factor_to_variance
from cryojax.io import read_array_from_mrc, read_atoms_from_pdb
from cryojax.simulator import DiscreteStructuralEnsemble


@pytest.fixture
def voxel_volume(sample_mrc_path):
    real_voxel_grid = read_array_from_mrc(sample_mrc_path)
    return (
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=1.3),
    )


@pytest.fixture
def gmm_volume(sample_pdb_path):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
        loads_b_factors=True,
    )
    scattering_factor_parameters = cxs.PengScatteringFactorParameters(atom_identities)
    return cxs.GaussianMixtureVolume(
        positions=atom_positions,
        amplitudes=scattering_factor_parameters.a,
        variances=convert_b_factor_to_variance(
            scattering_factor_parameters.b + b_factors[:, None]
        ),
    )


@pytest.mark.parametrize(
    "volume",
    [("voxel_volume"), ("gmm_volume")],
)
def test_conformation(volume, request):
    volume = request.getfixturevalue(volume)
    conformational_space = tuple([volume for _ in range(3)])
    volume = DiscreteStructuralEnsemble(conformational_space, conformation=0)
    _ = volume.to_volume_representation()
