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


@pytest.mark.parametrize("shape", [None, (32, 32), (96, 96), (50, 50)])
@pytest.mark.parametrize("offset", [0.0, 5.0])
def test_whitening_filter_white_noise_is_identity_when_resized(shape, offset):
    # Resizing must not break the identity: a white-noise input still has a flat
    # power spectrum after the filter is resampled to a smaller or larger shape.
    # `offset` gives the images a non-zero mean, which makes the zero-frequency
    # mode far larger than any noise mode. That mode is the image mean rather
    # than part of the power spectrum, so it must not leak into the lowest
    # frequency bin, where it would otherwise wreck the whole filter.
    images = jax.random.normal(jax.random.key(0), (128, 64, 64)) + offset
    array = im.WhiteningFilter(images, shape=shape).get()
    expected_shape = (64, 33) if shape is None else (shape[0], shape[1] // 2 + 1)
    assert array.shape == expected_shape
    assert bool(jnp.all(jnp.isfinite(array)))
    assert array[0, 0] == 1.0
    # ... exclude the zero-frequency mode, which is unity by construction
    ac_modes = array[1:, 1:]
    assert jnp.abs(jnp.mean(ac_modes) - 1.0) < 0.02
    assert jnp.std(ac_modes) < 0.1


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


def test_rotation_fn(basic_config, voxel_volume):
    """Rotating an image in fourier space must agree with rotating the object
    itself (via its pose) before projecting it."""
    rotation_angle = 35.0
    pose_norot = cxs.EulerAnglePose(theta_angle=90.0, psi_angle=0.0)
    pose_ref = cxs.EulerAnglePose(theta_angle=90.0, psi_angle=rotation_angle)
    image_model_norot = cxs.make_image_model(voxel_volume, basic_config, pose=pose_norot)
    image_model_ref = cxs.make_image_model(voxel_volume, basic_config, pose=pose_ref)

    grid = basic_config.get_frequency_grid(physical=False, padding=True)
    rotation_fn = im.RotateFFT(rotation_angle, frequency_grid=grid)

    image_norot = image_model_norot.raw_simulate()
    image_ref = image_model_ref.raw_simulate()
    shape = basic_config.padded_shape
    image_rot = jnp.fft.irfftn(rotation_fn(jnp.fft.rfftn(image_norot)), s=shape)

    corr = _get_correlation(image_ref, image_rot)
    np.testing.assert_allclose(corr.item(), 1.0, atol=1e-1)


def test_rotation_fn_rejects_full_fft_grid(basic_config):
    """`RotateFFT` only interpolates the half-space (rfft) DFT: the frequencies
    the half space does not store are recovered from Hermitian symmetry, and a
    full (fftn) grid has no truncated axis for that to apply to."""
    full_grid = basic_config.get_frequency_grid(physical=False, full=True, padding=True)
    with pytest.raises(ValueError, match="rfft"):
        im.RotateFFT(35.0, frequency_grid=full_grid)


def test_rotation_fn_matches_analytic_rotation():
    """Rotating a sum of gaussians must match rendering those gaussians at
    rotated centers.

    This is the case the old code got badly wrong: ~20 % of a rotated grid's query
    points ask for `q_x < 0`, which the half space does not store, and resolving
    them without the Hermitian fold put the result ~20 % of peak off. (The rfft
    path was disabled in the test above rather than fixed.)

    `RotateFFT` does not deconvolve the linear kernel's `sinc^2` transfer
    function, because doing so needs a real-space visit and an `irfftn`/`rfftn`
    round-trip costs several times more than the interpolation itself. But a user
    who wants that accuracy can pay *nothing* for it: divide the real-space image
    by `sinc^2` before the `rfftn` they were going to do anyway. This test
    documents that recipe, and pins that it helps.
    """
    dim, sigma, angle = 64, 2.0, 37.0
    grid = im.make_coordinate_grid((dim, dim))
    centers = np.array([[6.0, 3.0], [-9.0, 11.0], [12.0, -8.0]])
    amplitudes = np.array([1.0, 0.6, 0.8])

    def render(cs):
        out = jnp.zeros((dim, dim))
        for c, a in zip(cs, amplitudes):
            r_squared = (grid[..., 0] - c[0]) ** 2 + (grid[..., 1] - c[1]) ** 2
            out = out + a * jnp.exp(-r_squared / (2 * sigma**2))
        return out

    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    image, expected = render(centers), render(centers @ rotation.T)

    # The recipe: divide the real-space image by the linear kernel's `sinc^2`
    # transfer function. Interpolating the DFT of `f / sinc^2` at rotated
    # coordinates yields `(f / sinc^2)(Ru) * sinc^2(Ru) = f(Ru)` -- the exactly
    # rotated image. It must be applied to the *input*, in the unrotated frame:
    # correcting the output would need `sinc^2` rotated, and `sinc^2` is not
    # rotationally symmetric.
    x = im.make_1d_coordinate_grid(dim)
    sinc_squared = jnp.sinc(x / dim) ** 2
    deconvolved = image / (sinc_squared[:, None] * sinc_squared[None, :])

    rotation_fn = im.RotateFFT(angle, frequency_grid=im.make_frequency_grid((dim, dim)))
    peak = float(jnp.abs(image).max())

    def rotate(img):
        rotated = jnp.fft.irfftn(rotation_fn(jnp.fft.rfftn(img)), s=(dim, dim))
        return float(jnp.abs(rotated - expected).max()) / peak

    error_raw, error_deconvolved = rotate(image), rotate(deconvolved)

    # Both must be far below the ~20 % that the un-folded Hermitian bug produced.
    assert error_raw < 0.15, f"{100 * error_raw:.2f} % of peak"
    # ...and deconvolving the kernel must measurably improve on that, for free.
    assert error_deconvolved < 0.6 * error_raw, (
        f"raw {100 * error_raw:.2f} %, deconvolved {100 * error_deconvolved:.2f} %"
    )


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
    shift_fn = im.PhaseShiftFFT(
        offset=jnp.array([10.0, -5.0]),
        pixel_size=basic_config.pixel_size,
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


@pytest.mark.parametrize("batch_shape", [(), (7,), (2, 3)])
def test_filters_and_masks_accept_batch_dimensions(batch_shape):
    # Filters and masks broadcast against leading batch dimensions, and doing so
    # must agree with applying them to each image one at a time.
    shape = (16, 16)
    filter = im.LowpassFilter(im.make_frequency_grid(shape))
    mask = im.CircularCosineMask(
        im.make_coordinate_grid(shape), radius=5, rolloff_width=1
    )
    images = jax.random.normal(jax.random.key(0), (*batch_shape, *shape))

    masked = mask(images)
    assert masked.shape == images.shape

    fourier_images = jnp.fft.rfftn(images, axes=(-2, -1))
    filtered = filter(fourier_images)
    assert filtered.shape == fourier_images.shape

    # ... equals applying the transform to each image individually
    flat = fourier_images.reshape(-1, *fourier_images.shape[-2:])
    expected = jnp.stack([filter(image) for image in flat])
    assert jnp.allclose(filtered.reshape(expected.shape), expected)
