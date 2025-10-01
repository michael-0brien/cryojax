import jax
import numpy as np
import pytest

import cryojax.simulator as cxs
from cryojax.io import read_array_from_mrc
from cryojax.ndimage import crop_to_shape


jax.config.update("jax_enable_x64", True)


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
    shape = volume.shape[0:2]
    return cxs.BasicImageConfig(
        shape=(int(0.9 * shape[0]), int(0.9 * shape[1])),
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
        pad_options=dict(shape=shape),
    )


@pytest.fixture
def image_model(volume, basic_config):
    return cxs.make_image_model(
        volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF()),
    )


# Test correct image shape
@pytest.mark.parametrize("model", ["image_model"])
def test_real_shape(model, request):
    """Make sure shapes are as expected in real space."""
    model = request.getfixturevalue(model)
    image = model.simulate()
    padded_image = model.simulate(removes_padding=False)
    assert image.shape == model.image_config.shape
    assert padded_image.shape == model.image_config.padded_shape


@pytest.mark.parametrize("model", ["image_model"])
def test_fourier_shape(model, request):
    """Make sure shapes are as expected in fourier space."""
    model = request.getfixturevalue(model)
    image = model.simulate(outputs_real_space=False)
    padded_image = model.simulate(removes_padding=False, outputs_real_space=False)
    assert image.shape == model.image_config.frequency_grid_in_pixels.shape[0:2]
    assert (
        padded_image.shape
        == model.image_config.padded_frequency_grid_in_pixels.shape[0:2]
    )


@pytest.mark.parametrize("extra_dim_y, extra_dim_x", [(1, 1), (1, 0), (0, 1)])
def test_even_vs_odd_image_shape(extra_dim_y, extra_dim_x, volume_and_pixel_size):
    volume, pixel_size = volume_and_pixel_size
    control_shape = volume.shape[0:2]
    test_shape = (control_shape[0] + extra_dim_y, control_shape[1] + extra_dim_x)
    config_control = cxs.BasicImageConfig(
        control_shape, pixel_size=pixel_size, voltage_in_kilovolts=300.0
    )
    config_test = cxs.BasicImageConfig(
        test_shape, pixel_size=pixel_size, voltage_in_kilovolts=300.0
    )
    pose = cxs.EulerAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(cxs.AstigmaticCTF())
    model_control = cxs.make_image_model(
        volume, config_control, pose=pose, transfer_theory=transfer_theory
    )
    model_test = cxs.make_image_model(
        volume, config_test, pose=pose, transfer_theory=transfer_theory
    )
    np.testing.assert_allclose(
        crop_to_shape(model_test.simulate(), control_shape),
        model_control.simulate(),
        atol=1e-4,
    )
