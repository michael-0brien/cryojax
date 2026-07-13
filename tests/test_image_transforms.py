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
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_info[0], pad_scale=2.0)


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
        im.SquareCosineMask,
        im.CircularCosineMask,
        im.Cylindrical2DCosineMask,
        im.Rectangular2DCosineMask,
    ]
    kwargs = [
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
    classes = [im.SphericalCosineMask, im.Rectangular3DCosineMask]
    kwargs = [
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
    f = im.WhiteningFilter(image, shape=filter_shape, interp=mode, squared=square)
    array = f.get()
    # Output is an rfft-shaped filter of the (resized) spatial shape
    spatial_shape = filter_shape if filter_shape is not None else image_shape[-2:]
    assert array.shape == (spatial_shape[0], spatial_shape[1] // 2 + 1)
    assert jnp.all(jnp.isfinite(array))
    # The mean (zero-frequency) mode is preserved
    assert array[0, 0] == 1.0


def test_whitening_filter_wrong_ndim_raises():
    with pytest.raises(ValueError, match="dimension 2 or 3"):
        im.WhiteningFilter(jnp.zeros((2, 3, 10, 10)))


@pytest.mark.parametrize("squared", [False, True])
def test_whitening_filter_white_noise_is_identity(squared):
    # A white-noise input has a flat power spectrum, so a mean- and
    # variance-preserving whitening filter should reduce to the identity.
    images = jax.random.normal(jax.random.key(0), (128, 64, 64))
    array = im.WhiteningFilter(images, squared=squared).get()
    assert array[0, 0] == 1.0
    assert jnp.abs(jnp.mean(array) - 1.0) < 0.02
    assert jnp.std(array) < 0.1


def test_whitening_filter_preserves_mean():
    # The zero-frequency mode is unity, so filtering leaves the mean unchanged
    noise = jax.random.normal(jax.random.key(1), (64, 64, 64))
    images = jnp.cumsum(noise, axis=1)
    whitening_filter = im.WhiteningFilter(images).get()
    filtered = jnp.fft.irfftn(
        jnp.fft.rfftn(images, axes=(1, 2)) * whitening_filter, s=(64, 64), axes=(1, 2)
    )
    assert jnp.allclose(jnp.mean(filtered), jnp.mean(images), atol=1e-5)


@pytest.mark.parametrize("shape", [(48, 48), (64, 63), (33, 40)])
def test_whitening_filter_variance_normalization_is_exact(shape):
    # The filter is normalized so that an input matching the estimated power
    # spectrum has its total variance preserved exactly. Since the variance
    # sums over the full frequency grid, the weighted mean of 1 / filter**2
    # (weighting each rfft mode by its Hermitian multiplicity) must be exactly
    # one. This fails if the normalization ignores the multiplicity.
    images = jax.random.normal(jax.random.key(1), (64, *shape))
    array = np.asarray(im.WhiteningFilter(images).get())
    # Independently derived Hermitian multiplicity of each rfft mode
    x_size = shape[1] // 2 + 1
    multiplicity = np.full(x_size, 2.0)
    multiplicity[0] = 1.0
    if shape[1] % 2 == 0:
        multiplicity[-1] = 1.0
    multiplicity = np.broadcast_to(multiplicity, array.shape).copy()
    is_ac = ~np.isclose(array, 0.0)
    is_ac[0, 0] = False
    inverse_power = np.where(is_ac, 1.0 / np.where(is_ac, array, 1.0) ** 2, 0.0)
    weighted_mean = np.sum(multiplicity * inverse_power) / np.sum(multiplicity * is_ac)
    assert weighted_mean == pytest.approx(1.0, abs=1e-4)


def test_whitening_filter_downsample_regression():
    # Regression test for the resize path: the real-space autocorrelation
    # kernel must be centered before cropping. Without the fftshift/ifftshift,
    # downsampling discards the kernel peak and the white-noise filter is no
    # longer close to the identity.
    images = jax.random.normal(jax.random.key(2), (128, 64, 64))
    array = im.WhiteningFilter(images, shape=(32, 32)).get()
    assert array.shape == (32, 32 // 2 + 1)
    assert jnp.all(jnp.isfinite(array))
    assert array[0, 0] == 1.0
    assert jnp.abs(jnp.mean(array) - 1.0) < 0.05
    assert jnp.std(array) < 0.15


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
        image_rot = jnp.fft.irfftn(rotation_fn(jnp.fft.rfftn(image_norot)), s=shape)
    else:
        image_rot = jnp.fft.ifftn(rotation_fn(jnp.fft.fftn(image_norot))).real

    corr = _get_correlation(image_ref, image_rot)
    np.testing.assert_allclose(corr.item(), 1.0, atol=1e-1)


@pytest.mark.parametrize("use_rfft", [True, False])
def test_translation_fn(basic_config, voxel_volume, use_rfft):
    pose_notranslate = cxs.EulerAnglePose()
    pose_translate = cxs.EulerAnglePose(
        offset_x_in_angstroms=10.0, offset_y_in_angstroms=-5.0
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
        offset=jnp.array([10.0, -5.0]),
        frequency_grid=grid,
    )

    image_notrans = image_model_notrans.simulate()
    image_ref = image_model_ref.simulate()
    if use_rfft:
        image_trans = jnp.fft.irfftn(
            shift_fn(jnp.fft.rfftn(image_notrans)), s=basic_config.padded_shape
        )
    else:
        image_trans = jnp.fft.ifftn(shift_fn(jnp.fft.fftn(image_notrans))).real

    np.testing.assert_allclose(image_ref, image_trans, atol=(0.0 if use_rfft else 5e-4))


def _get_correlation(im1, im2):
    return jnp.abs(jnp.sum(im1 * im2)) / (jnp.linalg.norm(im1) * jnp.linalg.norm(im2))
