import cryojax.ndimage as im
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from cryojax.ndimage import make_frequency_grid, make_radial_frequency_grid
from cryojax.ndimage._spreading import spread_gaussians_2d, spread_gaussians_3d
from jax.test_util import check_grads


#
# Downsampling
#
@pytest.mark.parametrize(
    "shape, downsample_factor",
    (((10, 10), 2), ((11, 11), 2)),
)
def test_downsample_preserves_sum(shape, downsample_factor):
    upsampled_shape = tuple(downsample_factor * s for s in shape)
    rng_key = jr.key(seed=1234)
    upsampled_image = 2.0 + 1.0 * jr.normal(rng_key, upsampled_shape)
    image = im.fourier_crop_to_shape(upsampled_image, shape)
    np.testing.assert_allclose(image.sum(), upsampled_image.sum())


def _flip_about_center(array):
    """Flip an array about the RELION-convention real-space center (index
    `shape // 2` on every axis), rather than about index `0` like `jnp.flip`.
    Equal to `jnp.flip` on odd-length axes; requires an extra one-pixel roll
    on even-length axes.
    """
    out = array
    for axis, size in enumerate(array.shape):
        out = jnp.flip(out, axis=axis)
        if size % 2 == 0:
            out = jnp.roll(out, shift=1, axis=axis)
    return out


@pytest.mark.parametrize(
    "shape, downsample_factor",
    (
        ((20, 20), 2),
        ((21, 21), 2),
        ((20, 20), 3),
        ((30, 30), 2.5),
        ((16, 20), 2),
        ((16, 16, 16), 2),
    ),
)
def test_fourier_crop_downsample_center_unchanged(shape, downsample_factor):
    # A real-space bump centered exactly at the RELION-convention center
    # index (`shape // 2`) should still peak at the (downsampled) center
    # index after downsampling.
    coordinate_grid = im.make_coordinate_grid(shape)
    image_or_volume = jnp.exp(-jnp.sum(coordinate_grid**2, axis=-1) / 8.0)
    downsampled = im.fourier_crop_downsample(image_or_volume, downsample_factor)
    peak_index = jnp.unravel_index(jnp.argmax(downsampled), downsampled.shape)
    center_index = tuple(s // 2 for s in downsampled.shape)
    assert peak_index == center_index


@pytest.mark.parametrize(
    "shape, downsample_factor",
    (
        ((20, 20), 2),
        ((21, 21), 2),
        ((20, 20), 3),
        ((16, 20), 2),
        ((15, 21), 3),
        ((16, 16, 16), 2),
    ),
)
def test_fourier_crop_downsample_preserves_center_symmetry(shape, downsample_factor):
    # A real-space signal that is symmetric about the RELION-convention
    # center should remain symmetric about that same (downsampled) center
    # index after downsampling.
    coordinate_grid = im.make_coordinate_grid(shape)
    image_or_volume = jnp.exp(-jnp.sum(coordinate_grid**2, axis=-1) / 8.0)
    downsampled = im.fourier_crop_downsample(image_or_volume, downsample_factor)
    np.testing.assert_allclose(downsampled, _flip_about_center(downsampled), atol=1e-4)


def test_fourier_crop_downsample_factor_one_is_identity():
    rng_key = jr.key(seed=0)
    image = jr.normal(rng_key, (10, 10))
    downsampled = im.fourier_crop_downsample(image, 1)
    np.testing.assert_allclose(downsampled, image, atol=1e-5)


@pytest.mark.parametrize(
    "shape, pixel_size, downsample_factor, sigma",
    (
        ((60, 60), 0.5, 3, 8.0),
        ((80, 80), 1.0, 2, 10.0),
        ((63, 63), 0.75, 3, 9.0),
        ((40, 40, 40), 0.5, 2, 6.0),
    ),
)
def test_fourier_crop_downsample_matches_directly_rendered_gaussian(
    shape, pixel_size, downsample_factor, sigma
):
    # Downsampling a well-resolved (i.e. not aliased -- `sigma` much greater
    # than either pixel size) Gaussian rendered at a fine pixel size should
    # quantitatively agree with directly rendering the same Gaussian at the
    # coarse pixel size. `preserve_mean=True` is needed so that amplitude
    # (rather than sum) is preserved, matching the amplitude-normalized,
    # directly-rendered Gaussian.
    fine_grid = im.make_coordinate_grid(shape) * pixel_size
    fine_gaussian = jnp.exp(-jnp.sum(fine_grid**2, axis=-1) / (2 * sigma**2))

    downsampled = im.fourier_crop_downsample(
        fine_gaussian, downsample_factor, preserve_mean=True
    )

    coarse_shape = tuple(s // downsample_factor for s in shape)
    coarse_grid = im.make_coordinate_grid(coarse_shape) * (pixel_size * downsample_factor)
    coarse_gaussian = jnp.exp(-jnp.sum(coarse_grid**2, axis=-1) / (2 * sigma**2))

    np.testing.assert_allclose(downsampled, coarse_gaussian, atol=2e-2)


#
# FFT
#
@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10), (11, 11), (11, 11, 11)])
def test_fft_agrees_with_jax_numpy(shape):
    random = jnp.asarray(np.random.randn(*shape))
    # fftn
    np.testing.assert_allclose(random, jnp.fft.ifftn(jnp.fft.fftn(random)).real)
    np.testing.assert_allclose(
        jnp.fft.ifftn(jnp.fft.fftn(random)).real, jnp.fft.ifftn(jnp.fft.fftn(random)).real
    )
    # rfftn
    np.testing.assert_allclose(random, jnp.fft.irfftn(jnp.fft.rfftn(random), s=shape))
    np.testing.assert_allclose(
        jnp.fft.irfftn(jnp.fft.rfftn(random), s=shape),
        jnp.fft.irfftn(jnp.fft.rfftn(random), s=shape),
    )


#
# Cropping and padding
#
@pytest.mark.parametrize(
    "shape, cropped_shape, center, safe_crop",
    (
        ((10, 10), (5, 5), (5, 5), True),
        ((10, 10, 10), (5, 5, 5), (5, 5, 5), True),
        ((20, 20), (5, 5), (5, 5), True),
        ((20, 20), (5, 5), (5, 5), False),
        ((20, 20, 20), (5, 5, 5), (5, 5, 5), True),
        ((20, 20, 20), (5, 5, 5), (5, 5, 5), False),
        ((21, 21), (5, 5), (5, 5), True),
        ((21, 21), (5, 5), (5, 5), False),
        ((20, 20), (6, 6), (6, 6), True),
        ((21, 21), (6, 6), (6, 6), False),
    ),
)
def test_crop_with_center(shape, cropped_shape, center, safe_crop):
    coordinate_grid = im.make_radial_coordinate_grid(shape)
    if not safe_crop:
        center = tuple(jnp.asarray(c) for c in center)
    cropped_grid = im.crop_to_shape(coordinate_grid, cropped_shape, center=center)  # type: ignore
    assert cropped_grid.shape == cropped_shape


@pytest.mark.parametrize(
    "shape, cropped_shape",
    (
        ((10, 10), (5, 5)),
        ((10, 10), (6, 6)),
        ((11, 11), (5, 5)),
        ((11, 11), (6, 6)),
        ((11, 10), (5, 6)),
        ((10, 11), (6, 5)),
        ((11, 10), (6, 5)),
        ((10, 11), (5, 6)),
        ((10, 10, 10), (5, 5, 5)),
        ((10, 10, 10), (6, 6, 6)),
        ((11, 11, 11), (5, 5, 5)),
        ((11, 11, 11), (6, 6, 6)),
    ),
)
def test_crop(shape, cropped_shape):
    larger_frequency_grid = jnp.linalg.norm(
        jnp.asarray(shape, dtype=float)
        * jnp.fft.fftshift(make_frequency_grid(shape, outputs_rfftfreqs=False)),
        axis=-1,
    )
    smaller_frequency_grid = jnp.linalg.norm(
        jnp.asarray(cropped_shape, dtype=float)
        * jnp.fft.fftshift(make_frequency_grid(cropped_shape, outputs_rfftfreqs=False)),
        axis=-1,
    )
    cropped_frequency_grid = im.crop_to_shape(larger_frequency_grid, cropped_shape)
    dc_freq = tuple(jnp.asarray(s // 2, dtype=int) for s in cropped_shape)
    np.testing.assert_allclose(
        smaller_frequency_grid[dc_freq], cropped_frequency_grid[dc_freq]
    )
    np.testing.assert_allclose(smaller_frequency_grid, cropped_frequency_grid)


def test_crop_symmetric_signal():
    signal = np.zeros((20, 20))
    signal[0:7, 0:7] = 1.0
    signal[-7:, 0:7] = 1.0
    signal[0:7, -7:] = 1.0
    signal[-7:, -7:] = 1.0
    signal_crop = im.crop_to_shape(jnp.asarray(signal), (14, 14))
    np.testing.assert_allclose(np.sum(signal_crop[0:7, 0:7]), np.sum(signal_crop[7:, 7:]))


def test_pad_symmetric_signal():
    signal = np.zeros((20, 20))
    signal[0:7, 0:7] = 1.0
    signal[-7:, 0:7] = 1.0
    signal[0:7, -7:] = 1.0
    signal[-7:, -7:] = 1.0
    signal_pad = im.pad_to_shape(jnp.asarray(signal), (32, 32))
    np.testing.assert_allclose(
        np.sum(signal_pad[0:16, 0:16]), np.sum(signal_pad[16:, 16:])
    )


@pytest.mark.parametrize(
    "padded_shape, shape",
    (
        ((10, 10), (5, 5)),
        ((10, 10), (6, 6)),
        ((11, 11), (5, 5)),
        ((11, 11), (6, 6)),
        ((11, 10), (5, 6)),
        ((10, 11), (6, 5)),
        ((11, 10), (6, 5)),
        ((10, 11), (5, 6)),
        ((10, 10, 10), (5, 5, 5)),
        ((10, 10, 10), (6, 6, 6)),
        ((11, 11, 11), (5, 5, 5)),
        ((11, 11, 11), (6, 6, 6)),
    ),
)
def test_pad(padded_shape, shape):
    smaller_frequency_grid = jnp.linalg.norm(
        jnp.asarray(shape, dtype=float)
        * jnp.fft.fftshift(make_frequency_grid(shape, outputs_rfftfreqs=False)),
        axis=-1,
    )
    larger_frequency_grid = jnp.linalg.norm(
        jnp.asarray(padded_shape, dtype=float)
        * jnp.fft.fftshift(make_frequency_grid(padded_shape, outputs_rfftfreqs=False)),
        axis=-1,
    )
    padded_frequency_grid = im.pad_to_shape(smaller_frequency_grid, padded_shape)
    dc_freq = tuple(jnp.asarray(s // 2, dtype=int) for s in padded_shape)
    np.testing.assert_allclose(
        larger_frequency_grid[dc_freq], padded_frequency_grid[dc_freq]
    )
    np.testing.assert_allclose(
        im.crop_to_shape(larger_frequency_grid, shape),
        im.crop_to_shape(padded_frequency_grid, shape),
    )


# Normalization
def test_bg_subtract():
    image = jnp.ones((10, 10))
    image = np.asarray(im.background_subtract_image(image))
    assert np.all(image == 0.0)


# Fourier statistics
@pytest.mark.parametrize(
    "shape",
    [
        (10, 10),
        (10, 10, 10),
    ],
)
def test_powerspectrum_jit(shape):
    pixel_size = 1.2
    fourier_image = jnp.fft.rfftn(jr.normal(jr.key(1234), shape))
    radial_frequency_grid = make_radial_frequency_grid(shape, pixel_size)

    @jax.jit
    def compute_powerspectrum_jit(image, radial_freqs, ps):
        return im.compute_binned_powerspectrum(
            image, radial_freqs, ps, minimum_frequency=0.0, maximum_frequency=0.5
        )

    try:
        _ = compute_powerspectrum_jit(fourier_image, radial_frequency_grid, pixel_size)
    except Exception as err:
        raise Exception(
            "Could not successfully run JIT compiled function "
            "`cryojax.image.compute_binned_powerspectrum`. "
            f"Error traceback was:\n{err}"
        )


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10),
        (10, 10, 10),
    ],
)
def test_frc_fsc_jit(shape):
    if len(shape) == 2:
        correlation_fn = im.compute_fourier_ring_correlation
    else:
        correlation_fn = im.compute_fourier_shell_correlation
    pixel_size = 1.1
    fourier_image_1 = jnp.fft.rfftn(jr.normal(jr.key(1234), shape))
    fourier_image_2 = jnp.fft.rfftn(jr.normal(jr.key(2345), shape))
    radial_frequency_grid = make_radial_frequency_grid(shape, pixel_size)
    threshold = 0.5

    @jax.jit
    def compute_frc_fsc_jit(im1, im2, radial_freqs, ps, thresh):
        return correlation_fn(
            im1,
            im2,
            radial_freqs,
            ps,
            thresh,
            minimum_frequency=0.0,
            maximum_frequency=0.5,
        )

    try:
        _ = compute_frc_fsc_jit(
            fourier_image_1,
            fourier_image_2,
            radial_frequency_grid,
            pixel_size,
            threshold,
        )
    except Exception as err:
        raise Exception(
            "Could not successfully run JIT compiled function "
            f"`cryojax.image.{correlation_fn.__name__}`. "
            f"Error traceback was:\n{err}"
        )


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((8, 8), [1.0, 2.0, 2.0, 2.0, 1.0]),  # even width: Nyquist column weight 1
        ((8, 7), [1.0, 2.0, 2.0, 2.0]),  # odd width: last column weight 2
        ((5, 6, 10), [1.0, 2.0, 2.0, 2.0, 2.0, 1.0]),  # 3D uses the last axis
    ],
)
def test_rfftn_multiplicity(shape, expected):
    np.testing.assert_array_equal(im.make_rfftn_multiplicity(shape), np.array(expected))


@pytest.mark.parametrize("shape", [(8, 8), (8, 7), (7, 7), (6, 10)])
def test_standardize_fft_matches_real_space(shape):
    # Standardizing in Fourier space must give unit real-space standard
    # deviation. This requires the correct Hermitian mode multiplicity; for
    # even widths the old accounting (Nyquist column weighted twice) was wrong.
    image = jr.normal(jr.key(0), shape)
    fourier_image = jnp.fft.rfftn(image)
    standardized = jnp.fft.irfftn(
        im.standardize_fft(fourier_image, real_shape=shape), s=shape
    )
    assert float(jnp.std(standardized)) == pytest.approx(1.0, abs=1e-5)


# #
# # Pixel size rescaling
# #
# @pytest.mark.parametrize("shape", [(20, 20), (21, 21)])
# def test_rescale_pixel_size(shape):
#
#     image_1 = jr.normal(jr.key(0), shape)
#     pixel_size = 2.0
#     rescaled_pixel_size = 1.0
#     image_2 = im.rescale_pixel_size(
#         im.rescale_pixel_size(
#             image_1, pixel_size, rescaled_pixel_size, method="lanczos5"
#         ),
#         rescaled_pixel_size,
#         pixel_size,
#         method="lanczos5",
#     )
#     crop_1 = im.crop_to_shape(image_1, (shape[0] // 2, shape[1] // 2))
#     crop_2 = im.crop_to_shape(image_2, (shape[0] // 2, shape[1] // 2))
#     crop_2 = im.rescale_image(crop_2, crop_1.std(), crop_1.mean())
#     from matplotlib import pyplot as plt

#     # fig, axes = plt.subplots(ncols=2)
#     # vmin, vmax = min(crop_1.min(), crop_2.min()), max(crop_1.max(), crop_2.max())
#     # axes[0].imshow(crop_1, vmin=vmin, vmax=vmax)
#     # axes[1].imshow(crop_2, vmin=vmin, vmax=vmax)
#     # plt.show()
#     np.testing.assert_allclose(crop_1, crop_2)


#
# query_efficient_grid_size
#
def _is_smooth(n: int) -> bool:
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@pytest.mark.parametrize(
    "shape",
    [(10, 10), (11, 11), (13, 17), (100, 100), (10, 10, 10), (11, 13, 17)],
)
def test_query_efficient_grid_size_no_padding(shape):
    result = im.query_efficient_grid_size(shape)
    assert len(result) == len(shape)
    for s, r in zip(shape, result):
        assert r >= s
        assert _is_smooth(r)


@pytest.mark.parametrize(
    "shape, pad_scale",
    [((10, 10), 1.5), ((11, 11), 2.0), ((13, 17), 1.25), ((10, 10, 10), 1.5)],
)
def test_query_efficient_grid_size_with_pad_scale(shape, pad_scale):
    import math

    result = im.query_efficient_grid_size(shape, pad_scale=pad_scale)
    assert len(result) == len(shape)
    for s, r in zip(shape, result):
        assert r >= math.ceil(pad_scale * s)
        assert _is_smooth(r)


@pytest.mark.parametrize(
    "shape", [(10, 10), (11, 11), (10, 11), (10, 10, 10), (11, 13, 15), (242, 242)]
)
def test_query_efficient_grid_size_only_even(shape):
    result = im.query_efficient_grid_size(shape, pad_scale=1.0, only_even=True)
    for r in result:
        assert r % 2 == 0, f"expected even result, got {r}"


def test_query_efficient_grid_size_no_parity_constraint():
    import math

    shape, pad_scale = (11, 13), 1.5
    result = im.query_efficient_grid_size(shape, pad_scale=pad_scale, only_even=False)
    for s, r in zip(shape, result):
        assert _is_smooth(r)
        assert r >= math.ceil(pad_scale * s)


@pytest.mark.parametrize(
    "shape, includes_dc, mode",
    [
        ((8, 8), False, "zero"),
        ((8, 8), True, "zero"),
        ((8, 8), False, "real"),
        ((7, 8), False, "zero"),
        ((8, 7), False, "zero"),
        ((7, 7), False, "zero"),
        ((8, 8, 8), False, "zero"),
        ((8, 8, 8), True, "zero"),
        ((7, 8, 8), False, "zero"),
        ((8, 7, 8), False, "real"),
        ((7, 7, 7), False, "zero"),
    ],
)
def test_enforce_rfftn_self_conjugates(shape, includes_dc, mode):
    """Mask-based implementation must produce bit-identical output to the
    original scatter-based one for all combinations of shape parity."""
    import jax.numpy as jnp
    from cryojax.ndimage import enforce_rfftn_self_conjugates

    rng = np.random.default_rng(0)
    rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    real = rng.standard_normal(rfft_shape)
    imag = rng.standard_normal(rfft_shape)
    arr = jnp.array(real + 1j * imag)

    result = enforce_rfftn_self_conjugates(arr, shape, includes_dc=includes_dc, mode=mode)

    # Determine which positions are self-conjugate for shape/includes_dc
    def _is_sc(idx):
        # idx is a tuple of length ndim (z, y, x) or (y, x)
        if len(shape) == 2:
            y_dim, x_dim = shape
            r, c = idx
            row_sc = r == 0 or (y_dim % 2 == 0 and r == y_dim // 2)
            col_sc = c == 0 or (x_dim % 2 == 0 and c == x_dim // 2)
            at_dc = (r, c) == (0, 0)
        else:
            z_dim, y_dim, x_dim = shape
            z, r, c = idx
            z_sc = z == 0 or (z_dim % 2 == 0 and z == z_dim // 2)
            row_sc = r == 0 or (y_dim % 2 == 0 and r == y_dim // 2)
            col_sc = c == 0 or (x_dim % 2 == 0 and c == x_dim // 2)
            row_sc = z_sc and row_sc
            at_dc = (z, r, c) == (0, 0, 0)
        return row_sc and col_sc and (includes_dc or not at_dc)

    # Every position: SC positions must be modified; non-SC must be unchanged
    for idx in np.ndindex(*rfft_shape):
        sc = _is_sc(idx)
        if sc:
            if mode == "zero":
                assert result[idx] == 0.0, f"SC position {idx} should be zero"
            elif mode == "one":
                assert result[idx] == 1.0, f"SC position {idx} should be one"
            elif mode == "real":
                assert result[idx].imag == 0.0, f"SC position {idx} should be real"
        else:
            np.testing.assert_array_equal(result[idx], arr[idx])


def test_operators_instantiate():
    frequency_grid_1d = im.make_1d_frequency_grid(10)
    frequency_grid_2d = im.make_frequency_grid((10, 10))
    frequency_grid_3d = im.make_frequency_grid((10, 10, 10))
    coordinate_grid_1d = im.make_1d_coordinate_grid(10)
    coordinate_grid_2d = im.make_coordinate_grid((10, 10))
    coordinate_grid_3d = im.make_coordinate_grid((10, 10, 10))
    for cls in _real_operators_1d:
        _ = cls(coordinate_grid_1d)
    for cls in _real_operators_2d:
        _ = cls(coordinate_grid_2d)
    for cls in _real_operators_3d:
        _ = cls(coordinate_grid_3d)
    for cls in _fourier_operators_1d:
        _ = cls(frequency_grid_1d)
    for cls in _fourier_operators_2d:
        _ = cls(frequency_grid_2d)
    for cls in _fourier_operators_3d:
        _ = cls(frequency_grid_3d)


_fourier_operators_common = [
    im.FourierGaussian(),
    im.PeakedFourierGaussian(),
    im.FourierConstant(1.0),
    im.FourierSinc(),
    im.CustomFourierOperator(lambda _, a, b: a + b, 1.0, b=1.0),
    im.FourierConstant(1.0) + im.FourierConstant(1.0),
    im.FourierConstant(1.0) - im.FourierConstant(1.0),
    im.FourierConstant(1.0) * im.FourierConstant(1.0),
]
_real_operators_common = [
    im.RealGaussian(),
    im.RealConstant(1.0),
    im.RealConstant(1.0) + im.RealConstant(1.0),
    im.RealConstant(1.0) - im.RealConstant(1.0),
    im.RealConstant(1.0) * im.RealConstant(1.0),
]

_fourier_operators_1d = _fourier_operators_common
_fourier_operators_2d = [*_fourier_operators_common]
_fourier_operators_3d = _fourier_operators_common
_real_operators_1d = [*_real_operators_common, im.RealGaussian(offset=1.0)]
_real_operators_2d = [*_real_operators_common, im.RealGaussian(offset=(1.0, -1.0))]
_real_operators_3d = [*_real_operators_common, im.RealGaussian(offset=(1.0, -1.0, 0.0))]


# ── _spread.py: custom VJP ────────────────────────────────────────────────────
#
# Minimal, focused tests of `spread_2d`/`spread_3d` in isolation
# (finite-difference checks of the custom VJP rule), independent of the full
# `GaussianMixtureVolume` machinery that calls them.


@pytest.fixture
def points_2d():
    # `spread_2d` takes physical-unit positions (`0` at the real-space
    # center, grid index `n // 2`) and normalizes internally; scale the raw
    # offsets by `pixel_size` so the resulting grid-index footprint matches
    # what it was before that normalization moved inside `spread_2d`.
    key = jax.random.PRNGKey(0)
    m = 5
    ny, nx = 12, 11
    pixel_size = jnp.asarray(1.3)
    x = jax.random.uniform(key, (m,), minval=-3, maxval=3) * pixel_size
    y = (
        jax.random.uniform(jax.random.fold_in(key, 1), (m,), minval=-3, maxval=3)
        * pixel_size
    )
    amplitude = jax.random.normal(jax.random.fold_in(key, 2), (m,)) * 2 + 3
    variance = jnp.abs(jax.random.normal(jax.random.fold_in(key, 3), (m,))) * 0.3 + 0.4
    return x, y, amplitude, variance, pixel_size, (ny, nx)


@pytest.fixture
def points_3d():
    key = jax.random.PRNGKey(1)
    m = 5
    nz, ny, nx = 9, 10, 11
    voxel_size = jnp.asarray(1.1)
    x = jax.random.uniform(key, (m,), minval=-3, maxval=3) * voxel_size
    y = (
        jax.random.uniform(jax.random.fold_in(key, 1), (m,), minval=-3, maxval=3)
        * voxel_size
    )
    z = (
        jax.random.uniform(jax.random.fold_in(key, 2), (m,), minval=-3, maxval=3)
        * voxel_size
    )
    amplitude = jax.random.normal(jax.random.fold_in(key, 3), (m,)) * 2 + 3
    variance = jnp.abs(jax.random.normal(jax.random.fold_in(key, 4), (m,))) * 0.3 + 0.4
    return x, y, z, amplitude, variance, voxel_size, (nz, ny, nx)


@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_spread_2d_custom_vjp(points_2d, use_erf, scalar_variance):
    x, y, amplitude, variance, pixel_size, shape = points_2d
    if scalar_variance:
        variance = variance[0]
    n_spread = 9

    fn = lambda x, y, amplitude, variance, pixel_size: spread_gaussians_2d(
        x,
        y,
        amplitude,
        variance,
        shape,
        pixel_size=pixel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    check_grads(
        fn,
        (x, y, amplitude, variance, pixel_size),
        order=1,
        modes=["rev"],
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_spread_3d_custom_vjp(points_3d, use_erf, scalar_variance):
    x, y, z, amplitude, variance, voxel_size, shape = points_3d
    if scalar_variance:
        variance = variance[0]
    n_spread = 9

    fn = lambda x, y, z, amplitude, variance, voxel_size: spread_gaussians_3d(
        x,
        y,
        z,
        amplitude,
        variance,
        shape,
        voxel_size=voxel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    check_grads(
        fn,
        (x, y, z, amplitude, variance, voxel_size),
        order=1,
        modes=["rev"],
        atol=1e-4,
        rtol=1e-4,
    )


#
# Fourier projection-slice extraction
#
# These exercise the `cryojax.ndimage` API contracts directly. The actual
# slice/Ewald extraction accuracy is covered end-to-end in
# `test_volume_voxel.py`, so we deliberately avoid re-testing that here.
@pytest.mark.parametrize(
    "use_spline, pad_scale, expected_shape",
    (
        (False, 1.0, (8, 8, 5)),
        (True, 1.0, (10, 10, 7)),
        (False, 2.0, (16, 16, 9)),
        (True, 2.0, (18, 18, 11)),
    ),
)
def test_prepare_sampling_fft_shapes(use_spline, pad_scale, expected_shape):
    real_voxel_grid = jr.normal(jr.key(0), (8, 8, 8))
    prepared = im.prepare_sampling_fft(
        real_voxel_grid, pad_scale=pad_scale, use_spline=use_spline
    )
    assert prepared.shape == expected_shape


def test_prepare_sampling_fft_deconvolve_ignored_for_spline():
    # `apply_deconvolve` compensates for trilinear interpolation, so it must
    # be a no-op for spline coefficients but change the raw fourier grid.
    real_voxel_grid = jr.normal(jr.key(0), (8, 8, 8))
    spline_on = im.prepare_sampling_fft(
        real_voxel_grid, use_spline=True, apply_deconvolve=True
    )
    spline_off = im.prepare_sampling_fft(
        real_voxel_grid, use_spline=True, apply_deconvolve=False
    )
    np.testing.assert_array_equal(spline_on, spline_off)

    grid_on = im.prepare_sampling_fft(real_voxel_grid, apply_deconvolve=True)
    grid_off = im.prepare_sampling_fft(real_voxel_grid, apply_deconvolve=False)
    assert not jnp.allclose(grid_on, grid_off)


def test_prepare_sampling_fft_invalid_pad_scale():
    real_voxel_grid = jr.normal(jr.key(0), (8, 8, 8))
    with pytest.raises(ValueError, match="pad_scale"):
        im.prepare_sampling_fft(real_voxel_grid, pad_scale=0.5)


def test_prepare_sampling_fft_rejects_odd_and_noncubic():
    # The rfft half-grid logic assumes cubic, even dimensions.
    with pytest.raises(ValueError, match="even"):
        im.prepare_sampling_fft(jr.normal(jr.key(0), (7, 7, 7)))
    with pytest.raises(ValueError, match="cubic"):
        im.prepare_sampling_fft(jr.normal(jr.key(0), (8, 8, 6)))


def test_sample_fft_slice_rejects_odd_dim():
    grid = im.prepare_sampling_fft(jr.normal(jr.key(0), (8, 8, 8)))
    odd_frequency_slice = im.make_frequency_slice((7, 7), fftshifted=True)
    with pytest.raises(ValueError, match="even"):
        im.sample_fft_slice(grid, odd_frequency_slice)


@pytest.mark.parametrize("use_spline", (False, True))
def test_sample_fft_slice_shape_check(use_spline):
    # Passing a grid with `use_spline=True` (or spline coefficients with
    # `use_spline=False`) must be rejected, since the expected shapes differ.
    real_voxel_grid = jr.normal(jr.key(0), (8, 8, 8))
    # Deliberately mismatched: prepare the *wrong* representation.
    mismatched = im.prepare_sampling_fft(real_voxel_grid, use_spline=not use_spline)
    frequency_slice = im.make_frequency_slice((8, 8), fftshifted=True)
    with pytest.raises(ValueError, match="use_spline"):
        im.sample_fft_slice(mismatched, frequency_slice, use_spline=use_spline)


def test_ewald_sphere_from_slice_curvature():
    # For an unrotated slice, the Ewald surface keeps the in-plane
    # coordinates and displaces out of plane by `(wavelength / voxel_size) *
    # |q|**2 / 2` along the slice normal. The `wavelength=0` call gives the
    # flat, fully-reconstructed slice used as the in-plane reference.
    N, voxel_size, wavelength = 16, 1.3, 0.02
    frequency_slice = im.make_frequency_slice((N, N), fftshifted=True)
    flat = im.ewald_sphere_from_slice(frequency_slice, voxel_size, 0.0)
    surface = im.ewald_sphere_from_slice(frequency_slice, voxel_size, wavelength)

    assert surface.shape == (1, N, N, 3)
    # The `wavelength=0` surface is flat (no out-of-plane displacement).
    np.testing.assert_allclose(flat[..., 2], 0.0, atol=1e-6)
    # In-plane coordinates are unchanged by the curving.
    np.testing.assert_allclose(surface[..., 0:2], flat[..., 0:2], atol=1e-6)
    # Out-of-plane displacement matches the analytic Ewald curvature.
    q_squared = flat[..., 0] ** 2 + flat[..., 1] ** 2
    predicted_z = (wavelength / voxel_size) * q_squared / 2
    np.testing.assert_allclose(surface[..., 2], predicted_z, atol=1e-6)
