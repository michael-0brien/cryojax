"""Routines for downsampling arrays using fourier cropping."""

import math
from collections.abc import Callable
from typing import Literal, overload

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Complex, Float, Inexact

from ..jax_util import NDArrayLike
from ._coordinates import make_1d_frequency_grid, make_frequency_grid
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
    - `outputs_rfft`:
        Returns the result `fftn` instead of `rfftn` if equal to `False`.
        Only applies to real signals, and ignored if `outputs_real_space` is `False`.
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
    if jnp.iscomplexobj(image_or_volume):
        signal = _fft_ds_complex_signal_to_shape(
            image_or_volume, shape, outputs_real_space=outputs_real_space, shift=shift
        )
    else:
        signal = _fft_ds_real_signal_to_shape(
            image_or_volume,
            shape,
            outputs_real_space=outputs_real_space,
            outputs_rfft=outputs_rfft,
            shift=shift,
        )
    n_pixels, n_pixels_ds = math.prod(image_or_volume.shape), math.prod(shape)
    return (n_pixels_ds / n_pixels) * signal if preserve_mean else signal


def _phase_shift_factor(
    shape: tuple[int, ...], shift: tuple[float, ...], *, last_axis_is_rfft: bool
) -> Array:
    """Build the N-D phase ramp `exp(-2pi*i*sum(freq * shift))` as an outer
    product of per-axis 1D phase factors, applied via broadcasting rather than
    materializing a full N-D frequency grid."""
    ndim = len(shape)
    phase = jnp.asarray(1.0)
    for axis in range(ndim):
        freqs_1d = make_1d_frequency_grid(
            shape[axis],
            outputs_rfftfreqs=(last_axis_is_rfft and axis == ndim - 1),
            fftshifted=not last_axis_is_rfft,
        )
        reshape = [1] * ndim
        reshape[axis] = freqs_1d.shape[0]
        phase = phase * jnp.exp(
            -1.0j * 2 * jnp.pi * shift[axis] * freqs_1d.reshape(reshape)
        )
    return phase


def _fft_ds_real_signal_to_shape(
    image_or_volume: Float[NDArrayLike, "_ _"] | Float[NDArrayLike, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
    outputs_rfft: bool = True,
    shift: tuple[float, ...] | None = None,
) -> Inexact[Array, "_ _"] | Inexact[Array, "_ _ _"]:
    shape = image_or_volume.shape
    ndim = len(shape)

    # Forward Hartley Transform, computed via `rfftn` (roughly half the
    # compute/memory of a full `fftn`, since the input is real) instead of a
    # full complex FFT. `rfftn` stores only the non-negative-frequency half of
    # the last axis, in natural (DC-at-corner) order.
    rfft_array = rfftn(jnp.fft.ifftshift(image_or_volume))
    if shift is not None:
        rfft_array = (
            _phase_shift_factor(shape, shift, last_axis_is_rfft=True) * rfft_array
        )

    # Per axis, the natural-order indices of the low-frequency modes kept by
    # `crop_to_shape`'s centered crop window (positive frequencies, then the
    # wrapped negative frequencies), gathered directly by index rather than via
    # `fftshift` + `crop_to_shape` + `ifftshift` on the full-size spectrum.
    def _extraction_indices(dim: int, dim_ds: int) -> tuple[Array, Array]:
        q_min = -(dim_ds // 2)
        q_max = (dim_ds - 1) // 2
        return jnp.arange(q_max + 1), jnp.arange(dim + q_min, dim)

    idx_pos, idx_neg = zip(
        *(
            _extraction_indices(dim, dim_ds)
            for dim, dim_ds in zip(shape, downsampled_shape)
        )
    )
    # All axes but the last are stored in full by `rfftn` -- gather both the
    # positive- and negative-frequency indices for those directly.
    leading_idx = [jnp.concatenate([idx_pos[i], idx_neg[i]]) for i in range(ndim - 1)]
    last_pos, last_neg = idx_pos[-1], idx_neg[-1]

    block_pos = rfft_array[jnp.ix_(*leading_idx, last_pos)]
    if last_neg.size > 0:
        # The last axis's negative-frequency modes aren't stored by `rfftn`, so
        # reconstruct them from Hermitian symmetry:
        # `X[..., y_i, ..., N - m] = conj(X[..., (N_i - y_i) % N_i, ..., m])`,
        # gathered from the already-computed `rfft_array` rather than via
        # another transform.
        m_neg = shape[-1] - last_neg
        mirror_idx = [(shape[i] - leading_idx[i]) % shape[i] for i in range(ndim - 1)]
        block_neg = jnp.conj(rfft_array[jnp.ix_(*mirror_idx, m_neg)])
        hartley_array = jnp.concatenate([block_pos, block_neg], axis=-1)
    else:
        hartley_array = block_pos
    hartley_array = hartley_array.real - hartley_array.imag

    # Inverse Hartley Transform. No `fftshift`/`ifftshift` is needed here (or
    # before gathering `hartley_array` above): they are exact inverses of each
    # other and the Hartley combine in between is pointwise, so the shifts that
    # would otherwise bracket the crop cancel algebraically.
    ds_array = jnp.fft.fftshift(fftn(hartley_array))
    ds_array /= hartley_array.size
    ds_array = ds_array.real - ds_array.imag

    if outputs_real_space:
        return ds_array
    else:
        return rfftn(ds_array) if outputs_rfft else fftn(ds_array)


def _fft_ds_complex_signal_to_shape(
    image_or_volume: Complex[NDArrayLike, "_ _"] | Complex[NDArrayLike, "_ _ _"],
    downsampled_shape: tuple[int, int] | tuple[int, int, int],
    outputs_real_space: bool = True,
    shift: tuple[float, ...] | None = None,
) -> Complex[Array, "_ _"] | Complex[Array, "_ _ _"]:
    fourier_array = jnp.fft.fftshift(fftn(image_or_volume))
    if shift is not None:
        fourier_array = (
            _phase_shift_factor(image_or_volume.shape, shift, last_axis_is_rfft=False)
            * fourier_array
        )

    # Crop to the desired shape
    cropped_fourier_array = crop_to_shape(fourier_array, downsampled_shape)

    if outputs_real_space:
        return ifftn(jnp.fft.ifftshift(cropped_fourier_array))
    else:
        return jnp.fft.ifftshift(cropped_fourier_array)
