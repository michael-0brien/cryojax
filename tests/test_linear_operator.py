import cryojax.experimental as cxe
import cryojax.simulator as cxs
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.io import read_array_from_mrc


@pytest.fixture
def voxel_info(sample_mrc_path):
    return read_array_from_mrc(sample_mrc_path, loads_grid_spacing=True)


@pytest.fixture
def voxel_volume(voxel_info):
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_info[0], pad_scale=1.3)


@pytest.fixture
def voxel_size(voxel_info):
    return voxel_info[1]


@pytest.fixture
def image_config(voxel_volume, voxel_size):
    shape = voxel_volume.shape[0:2]
    return cxs.BasicImageConfig(
        shape=(int(0.9 * shape[0]), int(0.9 * shape[1])),
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
        pad_options=dict(shape=shape),
    )


@pytest.fixture
def image_model(voxel_volume, image_config):
    pose = cxs.EulerAnglePose()
    return cxs.make_image_model(voxel_volume, image_config, pose)


def test_simulate_equality(image_model):
    linear_operator, vector = cxe.make_linear_operator(
        simulate_fn=lambda x: x.simulate(),
        args=image_model,
        where_vector=lambda x: x.volume_parametrization.fourier_voxel_grid,
    )
    image_cxs = image_model.simulate()
    image_lx = linear_operator.mv(vector)
    np.testing.assert_allclose(image_cxs, image_lx)


def test_linear_transpose(image_model):
    where_vector = lambda x: x.volume_parametrization.fourier_voxel_grid
    linear_operator, _ = cxe.make_linear_operator(
        simulate_fn=lambda x: x.simulate(),
        args=image_model,
        where_vector=where_vector,
    )
    voxel_grid = where_vector(image_model)
    backprojection = where_vector(
        linear_operator.T.mv(jnp.zeros(image_model.image_config.shape))
    )
    assert voxel_grid.shape == backprojection.shape


def test_bad_linear_transpose(sample_pdb_path, image_config):
    image_model = cxs.make_image_model(
        cxs.load_tabulated_volume(sample_pdb_path, output_type=cxs.GaussianMixtureVolume),
        image_config,
        pose=cxs.EulerAnglePose(),
    )
    where_vector = lambda x: x.volume_parametrization.positions
    linear_operator, _ = cxe.make_linear_operator(
        simulate_fn=lambda x: x.simulate(),
        args=image_model,
        where_vector=where_vector,
    )
    with pytest.raises(Exception):
        linear_operator.T
