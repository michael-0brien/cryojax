import numpy as np
import pytest

import cryojax.simulator as cxs
from cryojax.io import read_array_from_mrc


@pytest.fixture
def voxel_potential(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_from_mrc(sample_mrc_path, loads_spacing=True)
    return cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=1.3
    )


@pytest.fixture
def basic_config(voxel_potential):
    return cxs.BasicConfig(
        shape=voxel_potential.shape[0:2],
        pixel_size=voxel_potential.voxel_size,
        voltage_in_kilovolts=300.0,
    )


@pytest.fixture
def image_model(voxel_potential, basic_config):
    image_model = cxs.make_image_model(
        voxel_potential,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AberratedAstigmaticCTF()),
    )
    return image_model


@pytest.mark.parametrize(
    "cls, model",
    [
        (cxs.IndependentGaussianPixels, "image_model"),
        (cxs.IndependentGaussianFourierModes, "image_model"),
    ],
)
def test_simulate_signal_from_gaussian_distributions(cls, model, request):
    model = request.getfixturevalue(model)
    distribution = cls(model)
    np.testing.assert_allclose(model.simulate(), distribution.compute_signal())
