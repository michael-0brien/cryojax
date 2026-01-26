import cryojax.ndimage as im
import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.io import read_array_from_mrc
from cryojax.ndimage import make_coordinate_grid, make_frequency_grid


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
def basic_config(voxel_volume, voxel_size):
    shape = voxel_volume.shape[0:2]
    return cxs.BasicImageConfig(
        shape=shape,
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
    )


def test_mask_2d_running():
    classes = [
        im.SincCorrectionMask,
        im.SquareCosineMask,
        im.CircularCosineMask,
        im.Cylindrical2DCosineMask,
        im.Rectangular2DCosineMask,
    ]
    kwargs = [
        dict(),
        dict(side_length=5, rolloff_width=2),
        dict(radius=5, rolloff_width=2),
        dict(radius=5, rolloff_width=2, length=5, rotation_angle=2.0),
        dict(x_width=5, y_width=5, rolloff_width=2, rotation_angle=2.0),
    ]
    coordinate_grid = make_coordinate_grid((10, 10))
    image = jnp.zeros((10, 10))
    for i, cls in enumerate(classes):
        mask = cls(coordinate_grid, **kwargs[i])
        _ = mask.get()
        _ = mask(image)


def test_mask_3d_running():
    classes = [im.SincCorrectionMask, im.SphericalCosineMask, im.Rectangular3DCosineMask]
    kwargs = [
        dict(),
        dict(radius=5, rolloff_width=2),
        dict(x_width=5, y_width=5, z_width=5, rolloff_width=2),
    ]
    coordinate_grid = make_coordinate_grid((10, 10, 10))
    image = jnp.zeros((10, 10, 10))
    for i, cls in enumerate(classes):
        mask = cls(coordinate_grid, **kwargs[i])
        _ = mask.get()
        _ = mask(image)


def test_filter_running():
    classes = [im.LowpassFilter, im.HighpassFilter]
    kwargs = [dict(), dict()]
    frequency_grid_2d, fourier_image_2d = (
        make_frequency_grid((10, 10)),
        jnp.zeros((10, 10 // 2 + 1)),
    )
    frequency_grid_3d, fourier_image_3d = (
        make_frequency_grid((10, 10, 10)),
        jnp.zeros((10, 10, 10 // 2 + 1)),
    )
    for i, cls in enumerate(classes):
        f_2d = cls(frequency_grid_2d, **kwargs[i])
        _ = f_2d.get()
        _ = f_2d(fourier_image_2d)
        f_3d = cls(frequency_grid_3d, **kwargs[i])
        _ = f_3d.get()
        _ = f_3d(fourier_image_3d)


def test_custom_filter_and_mask_initialization():
    classes = [im.CustomFilter, im.CustomMask]
    array = jnp.zeros((10, 10))
    for cls in classes:
        _ = cls(array)


@pytest.mark.parametrize(
    "image_shape, filter_shape, mode, square",
    (
        ((10, 10), None, "linear", False),
        ((2, 10, 10), None, "linear", False),
        ((2, 10, 10), (9, 9), "linear", False),
        ((2, 10, 10), (11, 11), "linear", False),
        ((2, 10, 10), None, "nearest", False),
        ((2, 10, 10), None, "linear", True),
    ),
)
def test_whitening_filter(image_shape, filter_shape, mode, square):
    rng_key = jax.random.key(1234)
    image = jax.random.normal(rng_key, image_shape)
    f = im.WhiteningFilter(
        image, shape=filter_shape, interpolation_mode=mode, outputs_squared=square
    )
    _ = f.get()


@pytest.mark.parametrize("use_rfft", [False])
def test_rotation_fn(basic_config, voxel_volume, use_rfft):
    rotation_angle = 35.0
    pose_norot = cxs.EulerAnglePose(theta_angle=90.0, psi_angle=0.0)
    pose_ref = cxs.EulerAnglePose(theta_angle=90.0, psi_angle=rotation_angle)
    image_model_norot = cxs.make_image_model(voxel_volume, basic_config, pose=pose_norot)
    image_model_ref = cxs.make_image_model(voxel_volume, basic_config, pose=pose_ref)

    if use_rfft:
        grid = basic_config.get_frequency_grid(physical=False, padding=True)
    else:
        grid = basic_config.get_frequency_grid(physical=False, full=True, padding=True)
    rotation_fn = im.RotateFFT(rotation_angle, grid)

    image_norot = image_model_norot.raw_simulate()
    image_ref = image_model_ref.raw_simulate()
    if use_rfft:
        shape = basic_config.padded_shape
        image_rot = im.irfftn(rotation_fn(im.rfftn(image_norot)), s=shape)
    else:
        image_rot = im.ifftn(rotation_fn(im.fftn(image_norot))).real

    corr = _get_correlation(image_ref, image_rot)
    np.testing.assert_allclose(corr.item(), 1.0, atol=1e-1)


@pytest.mark.parametrize("use_rfft", [True, False])
def test_translation_fn(basic_config, voxel_volume, use_rfft):
    pose_notranslate = cxs.EulerAnglePose()
    pose_translate = cxs.EulerAnglePose(
        offset_x_in_angstroms=50.0, offset_y_in_angstroms=-30.0
    )

    image_model_notrans = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=pose_notranslate,
        translate_mode="none",
    )

    image_model_ref = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=pose_translate,
    )
    if use_rfft:
        grid = basic_config.get_frequency_grid(physical=True)
    else:
        grid = basic_config.get_frequency_grid(physical=True, full=True)

    shift_fn = im.PhaseShiftFFT(
        offset=jnp.array([50.0, -30.0]),
        frequency_grid=grid,
    )

    image_notrans = image_model_notrans.simulate()
    image_ref = image_model_ref.simulate()
    if use_rfft:
        image_trans = im.irfftn(
            shift_fn(im.rfftn(image_notrans)), s=basic_config.padded_shape
        )
    else:
        image_trans = im.ifftn(shift_fn(im.fftn(image_notrans))).real

    np.testing.assert_allclose(image_ref, image_trans)


def _get_correlation(im1, im2):
    return jnp.abs(jnp.sum(im1 * im2)) / (jnp.linalg.norm(im1) * jnp.linalg.norm(im2))
