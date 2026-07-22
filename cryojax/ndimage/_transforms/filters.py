"""
Filters to apply to images in Fourier space
"""

import abc
import math
from typing import ClassVar
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ...jax_util import FloatLike, NDArrayLike
from .._coordinates import make_frequency_grid
from .._edges import resize_with_crop_or_pad
from .._fourier_statistics import compute_binned_powerspectrum
from .._fourier_utils import make_rfftn_multiplicity
from .._radial_average import radial_average_to_grid
from .base_transform import AbstractImageTransform


class AbstractFilter(AbstractImageTransform, strict=True):
    """Base class for computing and applying an image filter."""

    is_real_space: ClassVar[bool] = False

    @abc.abstractmethod
    def get(self) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError

    @override
    def __call__(
        self,
        image: (
            Complex[Array, "*batch y_dim x_dim"]
            | Complex[Array, "*batch z_dim y_dim x_dim"]
        ),
    ) -> (
        Complex[Array, "*batch y_dim x_dim"] | Complex[Array, "*batch z_dim y_dim x_dim"]
    ):
        """Apply the filter to an image or volume, which may carry leading
        batch dimensions. The filter is broadcast against them.
        """
        return image * self.get()


class CustomFilter(AbstractFilter, strict=True):
    """Pass a custom filter as an array."""

    array: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        filter: (
            Inexact[NDArrayLike, "y_dim x_dim"]
            | Inexact[NDArrayLike, "z_dim y_dim x_dim"]
        ),
    ):
        self.array = jnp.asarray(filter)

    @override
    def get(self) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.array


class LowpassFilter(AbstractFilter, strict=True):
    """Apply a low-pass filter to an image or volume, with
    a cosine soft-edge.
    """

    array: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        frequency_cutoff_fraction: FloatLike = 0.95,
        rolloff_width_fraction: FloatLike = 0.05,
    ):
        """**Arguments:**

        - `frequency_grid`:
            The frequency grid of the image or volume, in pixel-units.
        - `frequency_cutoff_fraction`:
            The cutoff frequency as a fraction of the Nyquist frequency.
            By default, `0.95`.
        - `rolloff_width_fraction`:
            The rolloff width as a fraction of the Nyquist frequency.
            By default, ``0.05``.
        """
        self.array = _compute_lowpass_filter(
            frequency_grid,
            jnp.asarray(frequency_cutoff_fraction),
            jnp.asarray(rolloff_width_fraction),
        )

    @override
    def get(self) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.array


class HighpassFilter(AbstractFilter, strict=True):
    """Apply a high-pass filter to an image or volume, with
    a cosine soft-edge.
    """

    array: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
        frequency_cutoff_fraction: FloatLike = 0.95,
        rolloff_width_fraction: FloatLike = 0.05,
    ):
        """**Arguments:**

        - `frequency_grid`:
            The frequency grid of the image or volume, in
            pixel-units.
        - `frequency_cutoff_fraction`:
            The cutoff frequency as a fraction of the Nyquist frequency.
            By default, `0.95`.
        - `rolloff_width_fraction`:
            The rolloff width as a fraction of the Nyquist frequency.
            By default, ``0.05``.
        """
        self.array = 1.0 - _compute_lowpass_filter(
            frequency_grid,
            jnp.asarray(frequency_cutoff_fraction),
            jnp.asarray(rolloff_width_fraction),
        )

    @override
    def get(self) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.array


class WhiteningFilter(AbstractFilter, strict=True):
    """Compute a whitening filter from an image. This is taken
    to be the inverse square root of the 2D radially averaged
    power spectrum.

    The filter is normalized to preserve the mean and variance of
    the image it is applied to: the zero-frequency (mean) mode is
    left unchanged and the remaining modes are rescaled so that a
    white-noise input maps to the identity filter.
    """

    array: Inexact[Array, "y_dim x_dim"]

    def __init__(
        self,
        images: Float[NDArrayLike, "_ _"] | Float[NDArrayLike, "_ _ _"],
        shape: tuple[int, int] | None = None,
        *,
        interp: str = "linear",
        squared: bool = False,
    ):
        """**Arguments:**

        - `images`:
            The image (or stack of images) from which to compute the power spectrum.
        - `shape`:
            The shape of the resulting filter. This downsamples or
            upsamples the filter by cropping or padding in real space.
        - `interp`:
            The method of interpolating the binned, radially averaged
            power spectrum onto a 2D grid. Either `nearest` or `linear`.
        - `squared`:
            If `False`, the whitening filter is the inverse square root of the image
            power. If `True`, the filter is the inverse of the image power.
        """
        images = jnp.asarray(images)
        if images.ndim not in (2, 3):
            raise ValueError(
                "`WhiteningFilter` expects a single image or a stack of images, i.e. "
                f"an array of dimension 2 or 3, but got an array of shape {images.shape}."
            )
        images = jnp.expand_dims(images, 0) if images.ndim == 2 else images
        self.array = _compute_whitening_filter(
            images, shape, interp=interp, squared=squared
        )

    @override
    def get(self) -> Inexact[Array, "y_dim x_dim"]:
        return self.array


def _compute_lowpass_filter(
    frequency_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    cutoff_fraction: Float[Array, ""],
    rolloff_width_fraction: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    k_max = 0.5
    cutoff_radius = cutoff_fraction * k_max
    rolloff_width = rolloff_width_fraction * k_max

    radial_frequency_grid = jnp.linalg.norm(frequency_grid, axis=-1)

    def compute_filter_at_frequency(radial_frequency):
        return jnp.where(
            radial_frequency <= cutoff_radius,
            1.0,
            jnp.where(
                radial_frequency > cutoff_radius + rolloff_width,
                0.0,
                0.5
                * (
                    1
                    + jnp.cos(jnp.pi * (radial_frequency - cutoff_radius) / rolloff_width)
                ),
            ),
        )

    compute_filter = (
        jax.vmap(jax.vmap(compute_filter_at_frequency))
        if radial_frequency_grid.ndim == 2
        else jax.vmap(jax.vmap(jax.vmap(compute_filter_at_frequency)))
    )

    return compute_filter(radial_frequency_grid)


def _compute_whitening_filter(
    images: Float[Array, "n_images y_dim x_dim"],
    shape: tuple[int, int] | None,
    *,
    interp: str,
    squared: bool,
) -> Float[Array, "{shape[0]} {shape[1]}"]:
    # Radially average the power spectrum over the image stack
    radial_freqs = jnp.linalg.norm(make_frequency_grid(images.shape[1:]), axis=-1)
    n_pixels = math.prod(images.shape[1:])
    fourier_images = jnp.fft.rfftn(images, axes=(1, 2)) / jnp.sqrt(n_pixels)
    compute_power_stack = jax.vmap(
        lambda im, freq: compute_binned_powerspectrum(
            im,
            freq,
            maximum_frequency=math.sqrt(2) / 2,
            real_shape=(images.shape[1], images.shape[2]),
        ),
        in_axes=[0, None],
        out_axes=(0, None),
    )
    power_stack, freq_bins = compute_power_stack(fourier_images, radial_freqs)
    radial_power = jnp.mean(power_stack, axis=0)
    # Interpolate the radial profile onto a 2D grid, optionally resampling
    # it to a different shape
    power = radial_average_to_grid(
        radial_power, freq_bins, radial_freqs, interpolation_mode=interp
    )
    out_shape = shape if shape is not None else (images.shape[1], images.shape[2])
    if shape is not None:
        power = _resize_power_spectrum(power, (images.shape[1], images.shape[2]), shape)
    # Invert the power to get the (unnormalized) whitening filter, guarding
    # against division by zero
    inverse_fn = jax.lax.reciprocal if squared else jax.lax.rsqrt
    is_zero = jnp.isclose(power, 0.0)
    whitening_filter = jnp.where(is_zero, 0.0, inverse_fn(power))
    # Normalize to preserve the mean and variance of the filtered image: keep
    # the zero-frequency (mean) mode at unity and rescale the remaining modes
    # so that a white-noise input (flat power) maps to the identity filter.
    # The mean power is weighted by the Hermitian multiplicity of each rfft
    # mode (see `_rfft_mode_multiplicity`) so that the total image variance,
    # which sums over the full frequency grid, is preserved exactly.
    is_ac = is_zero.at[0, 0].set(True)  # exclude DC and empty modes
    weights = jnp.where(is_ac, 0.0, make_rfftn_multiplicity(out_shape))
    mean_power = jnp.sum(weights * power) / jnp.sum(weights)
    scale = mean_power if squared else jnp.sqrt(mean_power)
    whitening_filter = (scale * whitening_filter).at[0, 0].set(1.0)

    return whitening_filter


def _resize_power_spectrum(
    power: Float[Array, "y_dim x_dim//2+1"],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> Float[Array, "{target_shape[0]} {target_shape[1]}//2+1"]:
    # Resample the (radially symmetric) power spectrum to a new shape by
    # cropping/padding its real-space autocorrelation kernel. The kernel must
    # be centered (fftshift) before resizing so its zero-lag peak sits at the
    # array center, otherwise the crop/pad (which operates about the center)
    # discards the peak. This shift used to be baked into the
    # `cryojax.ndimage.irfftn`/`rfftn` wrappers and is now applied explicitly.
    kernel = jnp.fft.fftshift(jnp.fft.irfftn(power, s=source_shape))
    resized = jnp.fft.ifftshift(
        resize_with_crop_or_pad(kernel, target_shape, mode="edge")
    )
    power = jnp.fft.rfftn(resized).real
    # ... resampling can introduce small negative values
    return jnp.where(power < 0, 0.0, power)
