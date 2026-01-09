import cryojax.ndimage as im
import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
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
        dict(radius=5, rolloff_width=2, length=5, in_plane_rotation_angle=2.0),
        dict(x_width=5, y_width=5, rolloff_width=2, in_plane_rotation_angle=2.0),
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


def test_rotation_op(basic_config, voxel_volume):
    pose_norot = cxs.EulerAnglePose(phi_angle=20.0, theta_angle=80.0, psi_angle=0.0)
    pose_ref = cxs.EulerAnglePose(phi_angle=20.0, theta_angle=80.0, psi_angle=38.0)

    image_model_norot = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=pose_norot,
    )

    image_model_ref = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=pose_ref,
    )

    rotation_op = im.RotateFFT(
        rotation_angle=38.0,
        frequency_grid=basic_config.full_frequency_grid_in_pixels,
        is_rfft=False,
    )

    image_norot = image_model_norot.simulate()
    image_ref = image_model_ref.simulate()
    image_rot = im.ifftn(rotation_op(im.fftn(image_norot))).real

    num = jnp.abs(jnp.sum(image_rot * image_ref))
    den = jnp.linalg.norm(image_rot) * jnp.linalg.norm(image_ref)
    corr = num / den
    assert corr > 0.98

    return


@pytest.mark.parametrize("use_rfft", [True, False])
def test_translation_op(basic_config, voxel_volume, use_rfft):
    pose_norot = cxs.EulerAnglePose()
    pose_ref = cxs.EulerAnglePose(offset_x_in_angstroms=50.0, offset_y_in_angstroms=-30.0)

    image_model_norot = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=pose_norot,
    )

    image_model_ref = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=pose_ref,
    )
    if use_rfft:
        grid = basic_config.frequency_grid_in_angstroms
    else:
        grid = basic_config.full_frequency_grid_in_angstroms

    translation_op = im.PhaseShiftFFT(
        offset=jnp.array([50.0, -30.0]),
        frequency_grid=grid,
        is_rfft=use_rfft,
    )

    image_notrans = image_model_norot.simulate()
    image_ref = image_model_ref.simulate()
    if use_rfft:
        image_trans = im.irfftn(
            translation_op(im.rfftn(image_notrans, s=image_ref.shape)), s=image_ref.shape
        )
    else:
        image_trans = im.ifftn(translation_op(im.fftn(image_notrans))).real

    num = jnp.abs(jnp.sum(image_trans * image_ref))
    den = jnp.linalg.norm(image_trans) * jnp.linalg.norm(image_ref)
    corr = num / den
    assert corr > 0.98

    return
