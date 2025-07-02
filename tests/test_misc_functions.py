import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import cryojax.ndimage as cxi
from cryojax.coordinates import make_frequency_grid


jax.config.update("jax_enable_x64", True)


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
    image = cxi.downsample_with_fourier_cropping(upsampled_image, downsample_factor)
    np.testing.assert_allclose(image.sum(), upsampled_image.sum())


#
# Normalization
#
def test_fourier_vs_real_normalize(noisy_model):
    key = jax.random.key(1234)
    im1 = cxi.normalize_image(
        noisy_model.render(key, outputs_real_space=True),
        input_is_real_space=True,
    )
    im2 = cxi.irfftn(
        cxi.normalize_image(
            noisy_model.render(outputs_real_space=False),
            input_is_real_space=False,
            input_is_rfft=True,
            shape_in_real_space=im1.shape,  # type: ignore
        ),
        s=noisy_model.instrument_config.shape,
    )  # type: ignore
    for im in [im1, im2]:
        np.testing.assert_allclose(jnp.std(im), jnp.asarray(1.0), rtol=1e-3)
        np.testing.assert_allclose(jnp.mean(im), jnp.asarray(0.0), atol=1e-8)


#
# FFT
#
@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10), (11, 11), (11, 11, 11)])
def test_fft_agrees_with_jax_numpy(shape):
    random = jnp.asarray(np.random.randn(*shape))
    # fftn
    np.testing.assert_allclose(random, cxi.ifftn(cxi.fftn(random)).real)
    np.testing.assert_allclose(
        cxi.ifftn(cxi.fftn(random)).real, jnp.fft.ifftn(jnp.fft.fftn(random)).real
    )
    # rfftn
    np.testing.assert_allclose(random, cxi.irfftn(cxi.rfftn(random), s=shape))
    np.testing.assert_allclose(
        cxi.irfftn(cxi.rfftn(random), s=shape),
        jnp.fft.irfftn(jnp.fft.rfftn(random), s=shape),
    )


#
# Cropping and padding
#
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
    cropped_frequency_grid = cxi.crop_to_shape(larger_frequency_grid, cropped_shape)
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
    signal_crop = cxi.crop_to_shape(jnp.asarray(signal), (14, 14))
    np.testing.assert_allclose(np.sum(signal_crop[0:7, 0:7]), np.sum(signal_crop[7:, 7:]))


def test_pad_symmetric_signal():
    signal = np.zeros((20, 20))
    signal[0:7, 0:7] = 1.0
    signal[-7:, 0:7] = 1.0
    signal[0:7, -7:] = 1.0
    signal[-7:, -7:] = 1.0
    signal_pad = cxi.pad_to_shape(jnp.asarray(signal), (32, 32))
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
    padded_frequency_grid = cxi.pad_to_shape(smaller_frequency_grid, padded_shape)
    dc_freq = tuple(jnp.asarray(s // 2, dtype=int) for s in padded_shape)
    np.testing.assert_allclose(
        larger_frequency_grid[dc_freq], padded_frequency_grid[dc_freq]
    )
    np.testing.assert_allclose(
        cxi.crop_to_shape(larger_frequency_grid, shape),
        cxi.crop_to_shape(padded_frequency_grid, shape),
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
#     image_2 = cxi.rescale_pixel_size(
#         cxi.rescale_pixel_size(
#             image_1, pixel_size, rescaled_pixel_size, method="lanczos5"
#         ),
#         rescaled_pixel_size,
#         pixel_size,
#         method="lanczos5",
#     )
#     crop_1 = cxi.crop_to_shape(image_1, (shape[0] // 2, shape[1] // 2))
#     crop_2 = cxi.crop_to_shape(image_2, (shape[0] // 2, shape[1] // 2))
#     crop_2 = cxi.rescale_image(crop_2, crop_1.std(), crop_1.mean())
#     from matplotlib import pyplot as plt

#     # fig, axes = plt.subplots(ncols=2)
#     # vmin, vmax = min(crop_1.min(), crop_2.min()), max(crop_1.max(), crop_2.max())
#     # axes[0].imshow(crop_1, vmin=vmin, vmax=vmax)
#     # axes[1].imshow(crop_2, vmin=vmin, vmax=vmax)
#     # plt.show()
#     np.testing.assert_allclose(crop_1, crop_2)
