import jax
import numpy as np
import pytest

import cryojax.simulator as cxs
from cryojax.io import read_array_from_mrc
from cryojax.ndimage import crop_to_shape


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def voxel_potential(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_from_mrc(sample_mrc_path, loads_spacing=True)
    return cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=1.3
    )


@pytest.fixture
def basic_config(voxel_potential):
    shape = voxel_potential.shape[0:2]
    return cxs.BasicConfig(
        shape=(int(0.9 * shape[0]), int(0.9 * shape[1])),
        pixel_size=voxel_potential.voxel_size,
        voltage_in_kilovolts=300.0,
        pad_options=dict(shape=shape),
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


# Test correct image shape
@pytest.mark.parametrize("model", ["image_model"])
def test_real_shape(model, request):
    """Make sure shapes are as expected in real space."""
    model = request.getfixturevalue(model)
    image = model.simulate()
    padded_image = model.simulate(removes_padding=False)
    assert image.shape == model.config.shape
    assert padded_image.shape == model.config.padded_shape


@pytest.mark.parametrize("model", ["image_model"])
def test_fourier_shape(model, request):
    """Make sure shapes are as expected in fourier space."""
    model = request.getfixturevalue(model)
    image = model.simulate(outputs_real_space=False)
    padded_image = model.simulate(removes_padding=False, outputs_real_space=False)
    assert image.shape == model.config.frequency_grid_in_pixels.shape[0:2]
    assert padded_image.shape == model.config.padded_frequency_grid_in_pixels.shape[0:2]


@pytest.mark.parametrize("extra_dim_y, extra_dim_x", [(1, 1), (1, 0), (0, 1)])
def test_even_vs_odd_image_shape(extra_dim_y, extra_dim_x, voxel_potential):
    control_shape = voxel_potential.shape[0:2]
    test_shape = (control_shape[0] + extra_dim_y, control_shape[1] + extra_dim_x)
    config_control = cxs.BasicConfig(
        control_shape, pixel_size=voxel_potential.voxel_size, voltage_in_kilovolts=300.0
    )
    config_test = cxs.BasicConfig(
        test_shape, pixel_size=voxel_potential.voxel_size, voltage_in_kilovolts=300.0
    )
    pose = cxs.EulerAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(cxs.AberratedAstigmaticCTF())
    model_control = cxs.make_image_model(
        voxel_potential, config_control, pose=pose, transfer_theory=transfer_theory
    )
    model_test = cxs.make_image_model(
        voxel_potential, config_test, pose=pose, transfer_theory=transfer_theory
    )
    np.testing.assert_allclose(
        crop_to_shape(model_test.simulate(), control_shape),
        model_control.simulate(),
        atol=1e-4,
    )
