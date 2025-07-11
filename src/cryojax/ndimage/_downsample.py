"""Routines for downsampling arrays"""

from typing import overload

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ._edges import crop_to_shape
from ._fft import fftn, ifftn, rfftn


@overload
def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _ _"],
    downsampling_factor: float | int,
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _ _"]: ...


@overload
def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"],
    downsampling_factor: float | int,
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _"]: ...


def downsample_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"],
    downsampling_factor: float | int,
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array using fourier cropping.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsample_factor`:
        A scale factor at which to downsample `image_or_volume`
        by. Must be a value greater than `1`.
    - `outputs_real_space`:
        If `False`, the `image_or_volume` is returned in fourier space.

    **Returns:**

    The downsampled `image_or_volume`, at the new real-space shape
    `downsampled_shape`. If `outputs_real_space = False`, return
    the downsampled array in fourier space, with the zero frequency
    component in the corner. For real signals, hermitian symmetry is
    assumed.
    """
    downsampling_factor = float(downsampling_factor)
    if downsampling_factor < 1.0:
        raise ValueError("`downsampling_factor` must be greater than 1.0")
    if image_or_volume.ndim == 2:
        image = image_or_volume
        new_shape = (
            int(image.shape[0] / downsampling_factor),
            int(image.shape[1] / downsampling_factor),
        )
        downsampled_array = downsample_to_shape_with_fourier_cropping(
            image, new_shape, outputs_real_space=outputs_real_space
        )
    elif image_or_volume.ndim == 3:
        volume = image_or_volume
        new_shape = (
            int(volume.shape[0] / downsampling_factor),
            int(volume.shape[1] / downsampling_factor),
            int(volume.shape[2] / downsampling_factor),
        )
        downsampled_array = downsample_to_shape_with_fourier_cropping(
            volume, new_shape, outputs_real_space=outputs_real_space
        )
    else:
        raise ValueError(
            "`downsample_with_fourier_cropping` can only crop images and volumes. "
            f"Got an array with number of dimensions {image_or_volume.ndim}."
        )

    return downsampled_array


@overload
def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"],
    downsampled_shape: tuple[int, int],
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _"]: ...


@overload
def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _ _"],
    downsampled_shape: tuple[int, int, int],
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _ _"]: ...


def downsample_to_shape_with_fourier_cropping(
    image_or_volume: Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array to a specified shape using fourier cropping.

    For real signals, the Hartley Transform is used to downsample the signal.
    For complex signals, the Fourier Transform is used to downsample the signal.

    The real case is based on the `downsample_transform` function in cryoDRGN
    https://github.com/ml-struct-bio/cryodrgn/blob/4ba75502d4dd1d0e5be3ecabf4a005c652edf4b5/cryodrgn/commands/downsample.py#L154

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsampled_shape`:
        The new shape after fourier cropping.
    - `outputs_real_space`:
        If `False`, the `image_or_volume` is returned in fourier space.

    **Returns:**

    The downsampled `image_or_volume`, at the new real-space shape
    `downsampled_shape`. If `outputs_real_space = False`, return
    the downsampled array in fourier space, with the zero frequency
    component in the corner. For real signals, hermitian symmetry is
    assumed.
    """
    if jnp.iscomplexobj(image_or_volume):
        return _downsample_complex_signal_to_shape(
            image_or_volume, downsampled_shape, outputs_real_space=outputs_real_space
        )
    else:
        return _downsample_real_signal_to_shape(
            image_or_volume, downsampled_shape, outputs_real_space=outputs_real_space
        )


def _downsample_real_signal_to_shape(
    image_or_volume: Float[Array, "_ _"] | Float[Array, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    # Forward Hartley Transform
    hartley_array = jnp.fft.fftshift(fftn(image_or_volume))
    hartley_array = hartley_array.real - hartley_array.imag

    # Crop to the desired shape
    ds_image_or_volume = crop_to_shape(hartley_array, downsampled_shape)

    # Inverse Hartley Transform
    ds_image_or_volume = jnp.fft.fftshift(fftn(ds_image_or_volume))
    ds_image_or_volume /= ds_image_or_volume.size
    ds_image_or_volume = ds_image_or_volume.real - ds_image_or_volume.imag

    if outputs_real_space:
        return ds_image_or_volume
    else:
        return rfftn(ds_image_or_volume)


def _downsample_complex_signal_to_shape(
    image_or_volume: Complex[Array, "_ _"] | Complex[Array, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
) -> Complex[Array, "_ _"] | Complex[Array, "_ _ _"]:
    fourier_array = jnp.fft.fftshift(fftn(image_or_volume))

    # Crop to the desired shape
    cropped_fourier_array = crop_to_shape(fourier_array, downsampled_shape)

    if outputs_real_space:
        return ifftn(jnp.fft.ifftshift(cropped_fourier_array))
    else:
        return jnp.fft.ifftshift(cropped_fourier_array)
