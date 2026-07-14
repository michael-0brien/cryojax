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
@pytest.mark.parametrize("interp", ("linear", "cubic"))
@pytest.mark.parametrize(
    "pad_scale, expected_shape", ((1.0, (8, 8, 5)), (2.0, (16, 16, 9)))
)
def test_prepare_sampling_fft_shapes(interp, pad_scale, expected_shape):
    # Every `interp` stores the same plain half-space (rfft) grid: the methods
    # differ only in which sinc power is deconvolved out of it beforehand.
    real_voxel_grid = jr.normal(jr.key(0), (8, 8, 8))
    prepared = im.prepare_sampling_fft(
        real_voxel_grid, pad_scale=pad_scale, interp=interp
    )
    assert prepared.shape == expected_shape


@pytest.mark.parametrize("interp, sinc_power", (("linear", 2), ("cubic", 4)))
def test_prepare_sampling_fft_deconvolves_per_interp(interp, sinc_power):
    # Each interp divides the real-space grid by its own kernel's transfer
    # function, sinc^(p+1), before the transform: sinc^2 for 'linear' and sinc^4
    # for 'cubic'. Check that against an independently written sinc factor.
    dim = 8
    real_voxel_grid = jr.normal(jr.key(0), (dim, dim, dim))
    x = im.make_1d_coordinate_grid(dim)
    sinc = jnp.sinc(x / dim)
    factor = (
        sinc[:, None, None] * sinc[None, :, None] * sinc[None, None, :]
    ) ** sinc_power
    expected = jnp.fft.fftshift(
        im.make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
        * jnp.fft.rfftn(real_voxel_grid / factor),
        axes=(0, 1),
    )
    prepared = im.prepare_sampling_fft(real_voxel_grid, interp=interp)
    np.testing.assert_allclose(
        np.asarray(prepared), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_prepare_sampling_fft_interps_differ():
    # sinc^2 and sinc^4 are different factors, so the two prepared grids must
    # not coincide -- otherwise the deconvolution is silently a no-op.
    real_voxel_grid = jr.normal(jr.key(0), (8, 8, 8))
    linear = im.prepare_sampling_fft(real_voxel_grid, interp="linear")
    cubic = im.prepare_sampling_fft(real_voxel_grid, interp="cubic")
    assert not jnp.allclose(linear, cubic)


def test_prepare_sampling_fft_rejects_bad_interp():
    with pytest.raises(ValueError, match="interp"):
        im.prepare_sampling_fft(jr.normal(jr.key(0), (8, 8, 8)), interp="quintic")


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


def test_sample_fft_slice_shape_check():
    # A grid whose shape doesn't match the frequency slice must be rejected.
    grid = im.prepare_sampling_fft(jr.normal(jr.key(0), (8, 8, 8)))
    frequency_slice = im.make_frequency_slice((16, 16), fftshifted=True)
    with pytest.raises(ValueError, match="rfft"):
        im.sample_fft_slice(grid, frequency_slice)


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


#
# Interpolation kernels: `map_coordinates(..., order=...)`
#
# `map_coordinates` always corresponds to scipy's `prefilter=False` case: the
# array is convolved with the interpolation kernel directly, never prefiltered
# so that the interpolant passes through the samples.
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("ndim", (2, 3))
def test_map_coordinates_matches_scipy_without_prefilter(order, ndim):
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    shape = (12,) * ndim
    array = np.asarray(jr.normal(jr.key(0), shape), dtype=np.float64)
    # Stay well inside the array so that the two libraries' (differing)
    # boundary conventions never enter.
    coords = np.asarray(
        jr.uniform(jr.key(1), (ndim, 50), minval=3.0, maxval=8.0), dtype=np.float64
    )
    expected = scipy_ndimage.map_coordinates(array, coords, order=order, prefilter=False)
    got = im.map_coordinates(
        jnp.asarray(array), tuple(jnp.asarray(c) for c in coords), order=order
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("order", (1, 3))
def test_map_coordinates_kernel_is_partition_of_unity(order):
    # Both kernels' weights sum to 1 at every coordinate, so a constant array
    # must be reproduced exactly, at integer and non-integer coordinates alike.
    array = jnp.full((10, 10, 10), 2.5)
    coords = jr.uniform(jr.key(0), (3, 40), minval=3.0, maxval=6.0)
    got = im.map_coordinates(array, tuple(coords), order=order)
    np.testing.assert_allclose(np.asarray(got), 2.5, rtol=1e-6)


@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("unroll", (True, False))
def test_map_coordinates_unroll_agrees(order, unroll):
    # The unrolled-loop and single-gather strategies must be numerically
    # identical; only their memory/speed profiles differ.
    array = jr.normal(jr.key(0), (10, 10, 10))
    coords = jr.uniform(jr.key(1), (3, 30), minval=-1.0, maxval=11.0)
    a = im.map_coordinates(array, tuple(coords), order=order, unroll=True)
    b = im.map_coordinates(array, tuple(coords), order=order, unroll=False)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_map_coordinates_rejects_bad_order():
    array = jr.normal(jr.key(0), (8, 8))
    coords = (jnp.zeros(3), jnp.zeros(3))
    with pytest.raises(ValueError, match="order"):
        im.map_coordinates(array, coords, order=2)


#
# `sample_fft_slice` boundary handling
#
# The fourier voxel grid is a real volume's DFT stored in the half space, so its
# values obey two exact symmetries: periodicity (`F[k + dim] == F[k]`) and
# Hermitian symmetry (`F[-k] == conj(F[k])`). Together these determine *every*
# interpolation tap, including taps that run off the end of the stored array.
#
# This matters enormously for `interp="cubic"`, whose kernel always reaches one
# node below the query's cell -- including at `q_x ~ 0`, where that node is not
# stored and carries 1/6 of the kernel weight on the volume's brightest
# coefficients (DC among them). Naively zero-filling it costs ~16% error.
def _reference_full_grid_slice(real_voxel_grid, frequency_slice, order, sinc_power):
    """Interpolate the FULL (non-truncated) complex grid, which needs no
    Hermitian symmetry logic at all. Ground truth for the half-grid sampler."""
    dim = real_voxel_grid.shape[0]
    grid = real_voxel_grid
    if sinc_power > 0:
        x = im.make_1d_coordinate_grid(dim)
        s = jnp.sinc(x / dim)
        grid = (
            grid / (s[:, None, None] * s[None, :, None] * s[None, None, :]) ** sinc_power
        )
    phase = im.make_fftshift_phase((dim, dim, dim), outputs_rfft=False)
    full = jnp.fft.fftshift(phase * jnp.fft.fftn(grid))
    k = frequency_slice * dim + dim // 2
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    in_box = (
        (jnp.abs(kx - dim // 2) <= dim // 2)
        & (ky >= 0)
        & (ky <= dim)
        & (kz >= 0)
        & (kz <= dim)
    )
    # Periodic boundary conditions: taps wrap modulo `dim`.
    out = im.map_coordinates(
        full,
        (kz % dim, ky % dim, kx % dim),
        order=order,
        mode="promise_in_bounds",
    )
    return jnp.where(in_box, out, 0.0)


@pytest.mark.parametrize(
    "interp, order, sinc_power",
    (("linear", 1, 2), ("cubic", 3, 4)),
)
def test_sample_fft_slice_matches_full_grid_reference(interp, order, sinc_power):
    """The half-space sampler must reproduce, exactly, an interpolation of the
    full (non-truncated) complex grid. This is what pins down the Hermitian
    fold: without it, `interp="cubic"` is off by ~16 % near the q_x = 0 plane.
    """
    from cryojax.rotations import SO3

    dim = 16
    real_voxel_grid = jr.normal(jr.key(0), (dim, dim, dim))
    frequency_slice = SO3.sample_uniform(jr.key(3)).apply(
        im.make_frequency_slice((dim, dim), fftshifted=True)
    )
    grid = im.prepare_sampling_fft(real_voxel_grid, interp=interp)

    # Undo the output-side fftshift/phase that `sample_fft_slice` applies, to
    # compare the raw interpolated F(q) against the reference.
    got = im.sample_fft_slice(grid, frequency_slice, interp=interp)
    got = im.make_fftshift_phase((dim, dim), outputs_rfft=True) * jnp.fft.fftshift(
        got, axes=(0,)
    )
    expected = _reference_full_grid_slice(
        real_voxel_grid, frequency_slice, order, sinc_power
    )[0]
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-4 * scale)


def test_sample_fft_slice_cubic_reconstructs_exact_transform():
    """`interp="cubic"` deconvolves the cubic kernel's sinc^4 transfer function,
    so interpolating the prepared grid reconstructs the volume's *exact*
    continuous fourier transform, up to aliasing -- not merely an approximation
    of it. Checked against a direct non-uniform DFT, inside the fourier box.

    This is the entire reason cubic beats a prefiltered spline interpolation:
    measured here, cubic is ~2 orders of magnitude more accurate than linear.

    The residual aliasing (and hence the tolerance below) depends on how
    compactly the density sits inside the box -- the deconvolution assumes a
    compactly supported volume, as real, masked cryo-EM maps are. A narrow,
    centered gaussian is used for that reason.
    """
    from cryojax.rotations import SO3

    dim = 16
    real_voxel_grid = np.asarray(
        jnp.exp(-jnp.sum(im.make_coordinate_grid((dim, dim, dim)) ** 2, axis=-1) / 2.0),
        dtype=np.float64,
    )
    frequency_slice = SO3.sample_uniform(jr.key(5)).apply(
        im.make_frequency_slice((dim, dim), fftshifted=True)
    )
    q = np.asarray(frequency_slice, dtype=np.float64).reshape(-1, 3)
    interior = np.linalg.norm(q, axis=-1) <= 0.4

    # Exact non-uniform DFT of the volume at the (continuous) slice coordinates.
    axis = np.arange(dim) - dim // 2
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float64)
    exact = np.exp(-2j * np.pi * (q[interior] @ points.T)) @ real_voxel_grid.ravel()
    scale = np.max(np.abs(exact))

    errors = {}
    for interp in ("linear", "cubic"):
        grid = im.prepare_sampling_fft(real_voxel_grid, interp=interp)
        got = im.sample_fft_slice(grid, frequency_slice, interp=interp)
        got = im.make_fftshift_phase((dim, dim), outputs_rfft=True) * jnp.fft.fftshift(
            got, axes=(0,)
        )
        got = np.asarray(got).reshape(-1)[interior]
        errors[interp] = np.max(np.abs(got - exact)) / scale

    assert errors["cubic"] < 1e-3, errors
    assert errors["cubic"] < errors["linear"] / 50, errors


@pytest.mark.parametrize("order", (1, 3))
def test_map_coordinates_negative_indices_obey_mode(order):
    """A tap index below the low edge must obey `mode`, not silently wrap around
    to the far edge of the array.

    JAX's gather modes only treat `index >= size` as out of bounds; a negative
    index gets numpy's wrap-around meaning under *every* mode. `map_coordinates`
    corrects for that, or a coordinate below zero reads the opposite edge.
    """
    array = jnp.arange(1.0, 6.0)[:, None] * jnp.ones((1, 5))  # rows 1..5
    column = jnp.array([2.0])

    # Half a pixel below the array: the tap at row -1 must be filled with 0, not
    # read as row 4 (= 5.0).
    filled = im.map_coordinates(
        array, (jnp.array([-0.5]), column), order=order, mode="fill", cval=0.0
    )
    assert float(filled[0]) < 1.0, "row -1 wrapped to the far edge instead of filling"

    # "clip" must clamp to the *near* edge (row 0 = 1.0), not the far one (5.0).
    # For order=3 this is not exactly 1.0: the kernel also reaches row 1 (= 2.0),
    # which is legitimately in bounds and carries a small weight.
    clipped = float(
        im.map_coordinates(array, (jnp.array([-0.5]), column), order=order, mode="clip")[
            0
        ]
    )
    assert abs(clipped - 1.0) < 0.1, f"clipped to the far edge instead: {clipped}"


def test_map_coordinates_cubic_interior_not_corrupted_by_low_edge():
    """The cubic kernel always reaches `floor(coordinate) - 1`, so a query point
    comfortably *inside* the array still gathers a tap at index -1. If that tap
    wrapped to the far edge it would corrupt an interior, in-bounds point.
    """
    array = jnp.arange(1.0, 6.0)[:, None] * jnp.ones((1, 5))  # a linear ramp
    coordinates = (jnp.array([0.2]), jnp.array([2.0]))
    linear = float(im.map_coordinates(array, coordinates, order=1, mode="fill")[0])
    cubic = float(im.map_coordinates(array, coordinates, order=3, mode="fill")[0])
    # On a linear ramp both kernels should land near 1.2. A far-edge wrap of the
    # -1 tap (row 4 = 5.0) would drag the cubic result well above it.
    np.testing.assert_allclose(linear, 1.2, atol=1e-6)
    assert abs(cubic - 1.2) < 0.25, f"interior cubic point corrupted: {cubic}"


#
# `map_frequencies`
#
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("ndim", (2, 3))
def test_map_frequencies_matches_full_grid(order, ndim):
    """Interpolating the half-space (rfft) DFT must reproduce, exactly, the same
    interpolation of the *full* (non-truncated) DFT, which needs no Hermitian
    symmetry logic at all. This is what pins the fold down.
    """
    dim = 16
    real_array = jr.normal(jr.key(0), (dim,) * ndim)

    half = jnp.fft.fftshift(jnp.fft.rfftn(real_array), axes=tuple(range(ndim - 1)))
    full = jnp.fft.fftshift(jnp.fft.fftn(real_array))

    # Query points straddling `q_x = 0`, so that roughly half of them ask for
    # frequencies the half space does not store -- that is what exercises the
    # Hermitian fold. They stay clear of the *centered* axes' edges, where the two
    # boundary conventions legitimately differ: the rfft grid wraps periodically,
    # as the DFT requires, while the full-grid reference is only given `fill`.
    keys = jr.split(jr.key(1), ndim)
    coords_centered = [
        jr.uniform(k, (400,), minval=2.0, maxval=float(dim - 3)) for k in keys[:-1]
    ]
    coord_x = jr.uniform(
        keys[-1], (400,), minval=-float(dim // 2 - 2), maxval=float(dim // 2 - 3)
    )
    assert float((coord_x < 0).mean()) > 0.3, "test does not exercise the fold"

    # `map_frequencies` takes frequencies in cycles/pixel, one array per axis
    # in array-axis order -- the truncated axis (`q_x`) last.
    frequencies = (
        *[(c - dim // 2) / dim for c in coords_centered],
        coord_x / dim,
    )
    got = im.map_frequencies(half, frequencies, order=order, mode="fill")
    # On the full grid the truncated axis is centered too, so shift the query.
    expected = im.map_coordinates(
        full,
        (*coords_centered, coord_x + dim // 2),
        order=order,
        mode="fill",
    )
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-4 * scale)


@pytest.mark.parametrize("shape", ((16, 16), (16, 10), (15, 8)))
def test_map_frequencies_rejects_non_rfft_shape(shape):
    # A square (non-truncated) grid, a wrongly-truncated one, and an odd dimension.
    with pytest.raises(ValueError, match="rfft"):
        im.map_frequencies(
            jnp.zeros(shape, dtype=complex), (jnp.zeros((3,)), jnp.zeros((3,)))
        )
