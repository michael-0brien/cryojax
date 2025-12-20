"""Routines for downsampling arrays using fourier cropping."""

import math
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Complex, Float, Inexact

from ..jax_util import NDArrayLike
from ._coordinates import make_frequency_grid
from ._edges import crop_to_shape
from ._fft import fftn, ifftn, rfftn


def block_reduce_downsample(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    downsample_factor: int,
    operation: Callable[[Array, Array], Array] = lax.add,
    center_correct: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array by pooling together blocks.
    Wraps `equinox.nn.Pool`.

    **Arguments:**

    - `image_or_volume`:
        image or volume array to downsample. The shape must be
        a multiple of `downsample_factor`
    - `downsample_factor`:
        A scale factor at which to downsample `image_or_volume`
        by. Must be a value greater than `1`.
    - `operation`:
        A function such as `operation = lambda x, y: f(x, y)`,
        where `x` and `y` are JAX arrays. See [`equinox.nn.Pool`]
        (https://docs.kidger.site/equinox/api/nn/pool/#equinox.nn.Pool)
        for documentation.
    - `center_correct`:
        If `True`, apply a phase shift in the fourier domain to correct
        the array center after downsampling. Applies only to even
        `downsample_factor`.

    **Returns:**

    The downsampled `image_or_volume` at shape reduced by
    `downsample_factor`.
    """
    array, k = image_or_volume, downsample_factor
    if k < 1:
        raise ValueError(
            "Called `block_reduce_downsample` with `downsample_factor` less than 1."
        )
    if array.ndim not in [2, 3]:
        raise ValueError(
            "`block_reduce_downsample` was passed an array with "
            f"`ndim = {array.ndim}`, but this function "
            "only supports images and volumes as input."
        )
    if any(s % k != 0 for s in array.shape):
        raise ValueError(
            "`block_reduce_downsample` only supports "
            "downsampling arrays with dimensions that "
            "are a multiple of `downsample_factor`."
            f"Got `downsample_factor = {downsample_factor}` "
            f"but `shape = {array.shape}`."
        )
    # Pooling function downsamples array
    shape = array.shape
    target_shape = tuple(s // k for s in shape)
    kernel_size = array.ndim * (k,)
    if k % 2 == 1:
        padding = tuple(
            ((k - 1) // 2, (k - 1) // 2) if s % 2 == 0 else (0, 0)
            for k, s in zip(kernel_size, shape)
        )
    else:
        padding = tuple((0, 0) for _ in shape)
        if center_correct:
            is_complex = jnp.iscomplexobj(array)
            q = make_frequency_grid(array.shape, outputs_rfftfreqs=False)
            if len(set(target_shape)) > 1:
                raise NotImplementedError(
                    "Tried to call `block_reduce_downsample` "
                    "with `center_correct = True`, even `downsample_factor`, "
                    "and a non-square image/volume. This is not implemented."
                )
            dim = target_shape[0]
            if dim % 2 == 0:
                shift = jnp.full((array.ndim,), (k - 1) / 2)
            else:
                shift = jnp.full((array.ndim,), -0.5)
            phase_shift = jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(q, shift)))
            array = ifftn(phase_shift * fftn(array))
            if not is_complex:
                array = array.real
    block_reduce_fn = lambda x: eqx.nn.Pool(
        init=jnp.asarray(0.0, array.dtype),
        operation=operation,
        num_spatial_dims=array.ndim,
        kernel_size=kernel_size,
        stride=kernel_size,
        padding=padding,
        use_ceil=False,
    )(x[None, ...])[0]

    array_ds = block_reduce_fn(array)

    return array_ds


def fourier_crop_downsample(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    downsample_factor: float | int,
    outputs_real_space: bool = True,
    preserve_mean: bool = False,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array using fourier cropping.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `downsample_factor`:
        A scale factor at which to downsample `image_or_volume`
        by. Must be a value greater than `1`.
    - `outputs_real_space`:
        If `False`, the `image_or_volume` is returned in fourier space
        with the zero-frequency component in the corner. For real signals,
        hermitian symmetry is assumed.
    - `preserve_mean`:
        Preserve the mean of the volume after downsampling, rather
        than the sum.

    **Returns:**

    The downsampled `image_or_volume` at shape reduced by
    `downsample_factor`.
    """
    downsample_factor = float(downsample_factor)
    if downsample_factor < 1.0:
        raise ValueError(
            "Called `fourier_crop_downsample` with `downsample_factor` less than 1."
        )
    if image_or_volume.ndim == 2:
        image = image_or_volume
        new_shape = (
            int(image.shape[0] / downsample_factor),
            int(image.shape[1] / downsample_factor),
        )
        downsampled_array = fourier_crop_to_shape(
            image,
            new_shape,
            preserve_mean=preserve_mean,
            outputs_real_space=outputs_real_space,
        )
    elif image_or_volume.ndim == 3:
        volume = image_or_volume
        new_shape = (
            int(volume.shape[0] / downsample_factor),
            int(volume.shape[1] / downsample_factor),
            int(volume.shape[2] / downsample_factor),
        )
        downsampled_array = fourier_crop_to_shape(
            volume,
            new_shape,
            preserve_mean=preserve_mean,
            outputs_real_space=outputs_real_space,
        )
    else:
        raise ValueError(
            "`fourier_crop_downsample` was passed an array with "
            f"`ndim = {image_or_volume.ndim}`, but this function "
            "only supports images and volumes as input."
        )

    return downsampled_array


def fourier_crop_to_shape(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
    preserve_mean: bool = False,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array to a specified shape using fourier cropping.

    For real signals, the Hartley Transform is used to downsample the signal.
    For complex signals, the Fourier Transform is used to downsample the signal.

    The real case is based on the `downsample_transform` function in cryoDRGN
    https://github.com/ml-struct-bio/cryodrgn/blob/4ba75502d4dd1d0e5be3ecabf4a005c652edf4b5/cryodrgn/commands/downsample.py#L154

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `shape`:
        The new shape after fourier cropping.
    - `outputs_real_space`:
        If `False`, the `image_or_volume` is returned in fourier space
        with the zero-frequency component in the corner. For real signals,
        hermitian symmetry is assumed.
    - `preserve_mean`:
        Preserve the mean of the volume after downsampling, rather
        than the sum.

    **Returns:**

    The downsampled `image_or_volume`, at the new real-space shape
    `shape`.
    """
    if jnp.iscomplexobj(image_or_volume):
        signal = _fft_ds_complex_signal_to_shape(
            image_or_volume, shape, outputs_real_space=outputs_real_space
        )
    else:
        signal = _fft_ds_real_signal_to_shape(
            image_or_volume, shape, outputs_real_space=outputs_real_space
        )
    n_pixels, n_pixels_ds = math.prod(image_or_volume.shape), math.prod(shape)
    return (n_pixels_ds / n_pixels) * signal if preserve_mean else signal


def _fft_ds_real_signal_to_shape(
    image_or_volume: Float[NDArrayLike, "_ _"] | Float[NDArrayLike, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    # Forward Hartley Transform
    hartley_array = jnp.fft.fftshift(fftn(image_or_volume))
    hartley_array = hartley_array.real - hartley_array.imag

    # Crop to the desired shape
    ds_array = crop_to_shape(hartley_array, downsampled_shape)

    # Inverse Hartley Transform
    ds_array = jnp.fft.fftshift(fftn(ds_array))
    ds_array /= ds_array.size
    ds_array = ds_array.real - ds_array.imag

    if outputs_real_space:
        return ds_array
    else:
        return rfftn(ds_array)


def _fft_ds_complex_signal_to_shape(
    image_or_volume: Complex[NDArrayLike, "_ _"] | Complex[NDArrayLike, "_ _ _"],
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
