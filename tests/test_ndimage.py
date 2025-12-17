import cryojax.ndimage as im
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from cryojax.ndimage import make_frequency_grid, make_radial_frequency_grid


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
    image = im.fourier_crop_downsample(upsampled_image, downsample_factor)
    np.testing.assert_allclose(image.sum(), upsampled_image.sum())


#
# FFT
#
@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10), (11, 11), (11, 11, 11)])
def test_fft_agrees_with_jax_numpy(shape):
    random = jnp.asarray(np.random.randn(*shape))
    # fftn
    np.testing.assert_allclose(random, im.ifftn(im.fftn(random)).real)
    np.testing.assert_allclose(
        im.ifftn(im.fftn(random)).real, jnp.fft.ifftn(jnp.fft.fftn(random)).real
    )
    # rfftn
    np.testing.assert_allclose(random, im.irfftn(im.rfftn(random), s=shape))
    np.testing.assert_allclose(
        im.irfftn(im.rfftn(random), s=shape),
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
    fourier_image = im.rfftn(jr.normal(jr.key(1234), shape))
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
    fourier_image_1 = im.rfftn(jr.normal(jr.key(1234), shape))
    fourier_image_2 = im.rfftn(jr.normal(jr.key(2345), shape))
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


def test_operators_instantiate():
    frequency_grid_2d = im.make_frequency_grid((10, 10))
    frequency_grid_3d = im.make_frequency_grid((10, 10, 10))
    coordinate_grid_2d = im.make_coordinate_grid((10, 10))
    coordinate_grid_3d = im.make_coordinate_grid((10, 10, 10))
    for cls in _real_operators_2d:
        _ = cls(coordinate_grid_2d)
    for cls in _real_operators_3d:
        _ = cls(coordinate_grid_3d)
    for cls in _fourier_operators_2d:
        _ = cls(frequency_grid_2d)
    for cls in _fourier_operators_3d:
        _ = cls(frequency_grid_3d)


_fourier_operators_2d = [
    im.FourierGaussian(),
    im.PeakedFourierGaussian(),
    im.FourierConstant(1.0),
    im.FourierSinc(),
    im.FourierExp2D(),
    im.CustomFourierOperator(lambda _, a, b: a + b, 1.0, b=1.0),
    im.FourierDC(),
    im.FourierConstant(1.0) + im.FourierConstant(1.0),
    im.FourierConstant(1.0) - im.FourierConstant(1.0),
    im.FourierConstant(1.0) * im.FourierConstant(1.0),
]


_fourier_operators_3d = [
    im.FourierGaussian(),
    im.PeakedFourierGaussian(),
    im.FourierConstant(1.0),
    im.FourierSinc(),
    im.CustomFourierOperator(lambda _, a, b: a + b, 1.0, b=1.0),
    im.FourierDC(),
    im.FourierConstant(1.0) + im.FourierConstant(1.0),
    im.FourierConstant(1.0) - im.FourierConstant(1.0),
    im.FourierConstant(1.0) * im.FourierConstant(1.0),
]

_real_operators_2d = [
    im.RealGaussian(),
    im.RealGaussian(offset=(1.0, -1.0)),
    im.RealConstant(1.0),
    im.RealConstant(1.0) + im.RealConstant(1.0),
    im.RealConstant(1.0) - im.RealConstant(1.0),
    im.RealConstant(1.0) * im.RealConstant(1.0),
]

_real_operators_3d = [
    im.RealGaussian(),
    im.RealGaussian(offset=(1.0, -1.0, 0.0)),
    im.RealConstant(1.0),
    im.RealConstant(1.0) + im.RealConstant(1.0),
    im.RealConstant(1.0) - im.RealConstant(1.0),
    im.RealConstant(1.0) * im.RealConstant(1.0),
]
