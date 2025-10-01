import numpy as np
import pytest

import cryojax.simulator as cxs
from cryojax.io import read_array_from_mrc


@pytest.fixture
def volume_and_pixel_size(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_from_mrc(
        sample_mrc_path, loads_grid_spacing=True
    )
    return (
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=1.3),
        voxel_size,
    )


@pytest.fixture
def volume(volume_and_pixel_size):
    return volume_and_pixel_size[0]


@pytest.fixture
def basic_config(volume_and_pixel_size):
    volume, pixel_size = volume_and_pixel_size
    return cxs.BasicImageConfig(
        shape=volume.shape[0:2],
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )


@pytest.fixture
def image_model(volume, basic_config):
    image_model = cxs.make_image_model(
        volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF()),
    )
    return image_model


@pytest.mark.parametrize(
    "cls, model",
    [
        (cxs.UncorrelatedGaussianNoiseModel, "image_model"),
        (cxs.CorrelatedGaussianNoiseModel, "image_model"),
    ],
)
def test_simulate_signal_from_gaussian_distributions(cls, model, request):
    model = request.getfixturevalue(model)
    distribution = cls(model)
    np.testing.assert_allclose(model.simulate(), distribution.compute_signal())
