"""Routines for downsampling arrays using fourier cropping."""

import math
from collections.abc import Callable
from typing import Literal, overload

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Inexact

from ..jax_util import NDArrayLike
from ._coordinates import make_frequency_grid
from ._fourier_utils import convert_rfftn_to_fftn


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
            array = jnp.fft.ifftn(phase_shift * jnp.fft.fftn(array))
            if not is_complex:
                array = array.real
    block_reduce_fn = lambda x: eqx.nn.Pool(
        init=0.0,
        operation=operation,
        num_spatial_dims=array.ndim,
        kernel_size=kernel_size,
        stride=kernel_size,
        padding=padding,
        use_ceil=False,
    )(x[None, ...])[0]

    array_ds = block_reduce_fn(array)

    return array_ds


def _resolve_downsample_shape_and_padding(
    shape: tuple[int, ...], downsample_factor: float
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    """For each axis, find the new (downsampled) length and the amount of
    real-space padding needed so that `(length + padding) / new_length` is as
    close as possible to `downsample_factor` -- exactly equal, if
    `downsample_factor` is an integer.
    """
    new_shape = []
    pad_widths = []
    achieved_factor = []
    for length in shape:
        new_length = max(1, round(length / downsample_factor))
        padded_length = round(new_length * downsample_factor)
        if padded_length < length:
            # only pad up, never crop away input data
            new_length += 1
            padded_length = round(new_length * downsample_factor)
        new_shape.append(new_length)
        pad_widths.append(padded_length - length)
        achieved_factor.append(padded_length / new_length)
    return tuple(new_shape), tuple(pad_widths), tuple(achieved_factor)


@overload
def fourier_crop_downsample(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    downsample_factor: float | int,
    outputs_real_space: bool = True,
    preserve_mean: bool = False,
    *,
    outputs_factor: Literal[False] = False,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]: ...


@overload
def fourier_crop_downsample(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    downsample_factor: float | int,
    outputs_real_space: bool = True,
    preserve_mean: bool = False,
    *,
    outputs_factor: Literal[True],
) -> tuple[Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"], tuple[float, ...]]: ...


def fourier_crop_downsample(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    downsample_factor: float | int,
    outputs_real_space: bool = True,
    preserve_mean: bool = False,
    *,
    outputs_factor: bool = False,
) -> (
    Inexact[Array, "_ _"]
    | Inexact[Array, "_ _ _"]
    | tuple[Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"], tuple[float, ...]]
):
    """Downsample an array using fourier cropping.

    To make `downsample_factor` exact (i.e. so that a caller can rescale a
    pixel/voxel size by exactly `downsample_factor`, rather than by the ratio
    implied by naively truncating `shape / downsample_factor`), the array is
    first padded (by edge replication) so that its shape is an exact multiple
    of the new, downsampled shape, then a sub-pixel phase shift is applied
    before cropping in fourier space so that the downsampled array's center
    stays anchored to the original (unpadded) array's own center.

    This is exact whenever `downsample_factor` is an integer. For a
    non-integer `downsample_factor`, an exact ratio is generally not
    achievable with integer-pixel padding; the new shape and padding are
    instead chosen to make the achieved ratio as close to `downsample_factor`
    as possible (error shrinking as the output size grows), and this resolved
    ratio can be recovered with `outputs_factor = True`.

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
    - `outputs_factor`:
        If `True`, also return the downsample factor actually resolved on
        each axis, as a tuple the same length as `image_or_volume.ndim`.
        Equal to `downsample_factor` on every axis when `downsample_factor`
        is an integer.

    **Returns:**

    The downsampled `image_or_volume` at shape reduced by
    `downsample_factor`. If `outputs_factor = True`, a
    `(downsampled_array, resolved_downsample_factor)` tuple instead.
    """
    downsample_factor = float(downsample_factor)
    if downsample_factor < 1.0:
        raise ValueError(
            "Called `fourier_crop_downsample` with `downsample_factor` less than 1."
        )
    if image_or_volume.ndim not in (2, 3):
        raise ValueError(
            "`fourier_crop_downsample` was passed an array with "
            f"`ndim = {image_or_volume.ndim}`, but this function "
            "only supports images and volumes as input."
        )
    new_shape, pad_widths, resolved_factor = _resolve_downsample_shape_and_padding(
        image_or_volume.shape, downsample_factor
    )
    if any(pad_widths):
        padded_array = jnp.pad(
            image_or_volume, tuple((0, p) for p in pad_widths), mode="edge"
        )
        # For a given axis, padding by `p` pixels shifts that axis's center by
        # `p / 2` in the common case -- except that `crop_to_shape`'s crop
        # window is asymmetric by one pixel whenever the *padded* length on
        # that axis is odd, which requires rounding `p / 2` down instead of up
        # to compensate.
        shift = tuple(
            p // 2 if (s + p) % 2 == 1 else (p + 1) // 2
            for s, p in zip(image_or_volume.shape, pad_widths)
        )
    else:
        padded_array = image_or_volume
        shift = None
    downsampled_array = _fourier_crop_to_shape(
        padded_array,
        new_shape,  # type: ignore
        outputs_real_space=outputs_real_space,
        preserve_mean=preserve_mean,
        shift=shift,
    )

    if outputs_factor:
        return downsampled_array, resolved_factor
    else:
        return downsampled_array


def fourier_crop_to_shape(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
    outputs_rfft: bool = True,
    preserve_mean: bool = False,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Downsample an array to a specified shape using fourier cropping.

    The modes kept are those of the new shape on cryoJAX's frequency grid: for an even
    size, `+N/2` on the last axis of a real signal's `rfftn` and `-N/2` on every
    other axis. The center pixel `N // 2` of each axis stays at the center `n // 2`.

    **Arguments:**

    - `image_or_volume`: The image or volume array to downsample.
    - `shape`:
        The new shape after fourier cropping.
    - `outputs_real_space`:
        If `False`, the `image_or_volume` is returned in fourier space
        with the zero-frequency component in the corner. For a real signal this is
        the cropped spectrum itself, whose inverse `rfftn` is the real-space output.
    - `outputs_rfft`:
        Returns the result `fftn` instead of `rfftn` if equal to `False`.
        Only applies to real signals, and ignored if `outputs_real_space` is `True`.
    - `preserve_mean`:
        Preserve the mean of the volume after downsampling, rather
        than the sum.

    **Returns:**

    The downsampled `image_or_volume`, at the new real-space shape
    `shape`.
    """
    return _fourier_crop_to_shape(
        image_or_volume,
        shape,
        outputs_real_space=outputs_real_space,
        outputs_rfft=outputs_rfft,
        preserve_mean=preserve_mean,
    )


def _fourier_crop_to_shape(
    image_or_volume: Inexact[NDArrayLike, "_ _"] | Inexact[NDArrayLike, "_ _ _"],
    shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
    outputs_rfft: bool = True,
    preserve_mean: bool = False,
    shift: tuple[float, ...] | None = None,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    """Shared backend for `fourier_crop_to_shape` and `fourier_crop_downsample`.

    `shift` is an optional per-axis sub-pixel shift (in pixels/voxels of
    `image_or_volume`), applied before cropping in fourier space. It is
    private to this module: `fourier_crop_downsample` uses it to correct for
    the real-space padding it applies before calling here, so the downsampled
    array's center stays anchored to the original (unpadded) array's own
    center; `fourier_crop_to_shape` always calls with `shift = None`.
    """
    array = jnp.asarray(image_or_volume)
    in_shape = array.shape
    is_real = not jnp.iscomplexobj(array)
    spectrum = jnp.fft.rfftn(array) if is_real else jnp.fft.fftn(array)
    # Keep the new shape's modes by static slices: every full axis keeps
    # 0 ... ceil(n/2) - 1 and -floor(n/2) ... -1, an rfft axis 0 ... n // 2
    n_full_axes = len(in_shape) - 1 if is_real else len(in_shape)
    for axis in range(n_full_axes):
        n_in, n_out = in_shape[axis], shape[axis]
        spectrum = jnp.concatenate(
            [
                lax.slice_in_dim(spectrum, 0, (n_out + 1) // 2, axis=axis),
                lax.slice_in_dim(spectrum, n_in - n_out // 2, n_in, axis=axis),
            ],
            axis=axis,
        )
    if is_real:
        spectrum = spectrum[..., : shape[-1] // 2 + 1]
    spectrum = spectrum * _centering_phase(in_shape, shape, shift, is_real)
    if preserve_mean:
        spectrum = spectrum * (math.prod(shape) / math.prod(in_shape))
    if outputs_real_space:
        return jnp.fft.irfftn(spectrum, s=shape) if is_real else jnp.fft.ifftn(spectrum)
    if is_real and not outputs_rfft:
        return convert_rfftn_to_fftn(spectrum, shape)
    return spectrum


def _centering_phase(
    in_shape: tuple[int, ...],
    shape: tuple[int, ...],
    shift: tuple[float, ...] | None,
    last_axis_is_rfft: bool,
) -> Array:
    """`exp(2 pi i k [(N//2 - shift)/N - (n//2)/n])` on the kept modes `k` of each axis,
    as an outer product: moves the center pixel `N // 2`, less `shift`, to `n // 2`.
    Equal to 1 on an axis where both sizes are even and there is no shift."""
    ndim = len(shape)
    shift = (0.0,) * ndim if shift is None else shift
    phase = jnp.asarray(1.0)
    for axis, (n_in, n_out, s) in enumerate(zip(in_shape, shape, shift)):
        offset = (n_in // 2 - s) / n_in - (n_out // 2) / n_out
        if offset == 0:
            continue
        if last_axis_is_rfft and axis == ndim - 1:
            modes = jnp.arange(n_out // 2 + 1)
        else:
            modes = jnp.fft.ifftshift(jnp.arange(-(n_out // 2), (n_out + 1) // 2))
        reshape = [1] * ndim
        reshape[axis] = modes.shape[0]
        phase = phase * jnp.exp(2j * jnp.pi * offset * modes).reshape(reshape)
    return phase
