"""
Interpolation utilities.

Two front-ends:

- `map_coordinates`, to interpolate an ordinary array at index coordinates.
- `map_frequencies`, to interpolate the fourier transform of a real signal at
  arbitrary frequencies.

Both are backed by one separable gather, `_interp_taps`. It takes the boundary
handling as per-axis *resolvers*, which is what lets a single implementation
serve both --- including the cross-axis coupling that Hermitian symmetry
introduces, and that no boundary `mode` could express.

Adapted from Louis Desdoigts's [version of `jax.scipy.map_coordinates`](https://github.com/LouisDesdoigts/jax/blob/cubic-spline-updated/jax/_src/scipy/ndimage.py),
which was developed for the project [`dLux`](https://louisdesdoigts.github.io/dLux/).
"""  # noqa: E501

from collections.abc import Callable, Sequence
from typing import Literal, cast

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Bool

from ._coordinates import make_1d_coordinate_grid


# Each interpolation method is a pair of an interpolation kernel order and the
# power of `sinc` that deconvolves that kernel's transfer function. The order-`p`
# B-spline is the unit box convolved with itself `p + 1` times, so its Fourier
# transform -- the blur it applies to the real-space signal -- is exactly
# `sinc^(p + 1)`. See `deconvolve_interpolation_kernel`.
_INTERP_METHODS: dict[str, tuple[int, int]] = {
    # (interpolation order, sinc power to deconvolve)
    "linear": (1, 2),
    "cubic": (3, 4),
}

# A tap is an array index paired with the weight the kernel gives it.
Taps = list[tuple[Array, ArrayLike]]
# Resolves a tap index on one axis into a valid array index. `needs_conjugate` is
# the flag from the truncated axis's fold (see `map_frequencies`); resolvers
# on independent axes ignore it.
Resolver = Callable[[Array, Bool[Array, "..."] | None], Array]


def parse_interp(interp: Literal["linear", "cubic"]) -> tuple[int, int]:
    """The `(order, sinc_power)` of an interpolation method."""
    if interp not in _INTERP_METHODS:
        raise ValueError(
            f"Invalid value `interp={interp!r}`. Supported interpolation "
            f"methods are {sorted(_INTERP_METHODS)}."
        )
    return _INTERP_METHODS[interp]


def map_coordinates(
    input: Array,
    coordinates: Sequence[Array],
    order: int = 1,
    mode: str = "fill",
    cval: float | complex = 0.0,
    unroll: bool = True,
) -> Array:
    """Interpolate a 2D or 3D array at arbitrary coordinates.

    Similar to `scipy.ndimage.map_coordinates`, but always corresponds to its
    `prefilter=False` case: `input` is convolved with the interpolation kernel
    directly. To interpolate the fourier transform of a real signal, use
    [`cryojax.ndimage.map_frequencies`][] instead.

    **Arguments:**

    - `input`:
        The 2D or 3D array to interpolate.
    - `coordinates`:
        A sequence of length `input.ndim`, one coordinate array per axis, in
        array-axis order and in index units (so `input[0, 0]` sits at coordinate
        `(0, 0)`). Each must be broadcastable to the same shape.
    - `order`:
        The order of the interpolation kernel: `1` for linear, `3` for cubic
        B-spline. Cubic is more accurate, at the cost of reading a `4^ndim`
        rather than a `2^ndim` neighborhood per query point.
    - `mode` :
        How to extrapolate beyond the edges of `input`, using JAX's
        [out-of-bounds indexing](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html)
        modes, e.g. `"fill"` or `"clip"`.
    - `cval`:
        The value returned for out-of-bounds coordinates when `mode` is
        `"fill"`. Ignored for other modes.
    - `unroll`:
        If `True` (the default), gather the interpolation taps one at a time,
        which keeps memory use small and predictable at large batch sizes. If
        `False`, gather them all at once, which is often substantially faster for
        `order=3` but whose peak memory grows with the number of query points.

    **Returns:**

    The interpolated values, with the shape that the coordinate arrays broadcast
    to.
    """  # noqa: E501
    input_arr = jnp.asarray(input)
    coordinate_arrs = _check_coordinates(input_arr, coordinates)

    taps = [indices_and_weights_fn(order)(c) for c in coordinate_arrs]
    # Every axis is independent: no coupling, and no conjugation.
    resolvers = [_bounded_resolver(size, mode) for size in input_arr.shape]

    result = _interp_taps(
        input_arr,
        taps,
        resolvers=resolvers,
        resolve_last=lambda index: (resolvers[-1](index, None), None),
        mode=mode,
        cval=cval,
        unroll=unroll,
    )
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def map_frequencies(
    input: Array,
    frequencies: Array,
    order: int = 1,
    mode: str = "fill",
    unroll: bool = True,
) -> Array:
    """Interpolate the fourier transform of a *real* 2D image or 3D volume, at
    arbitrary frequencies.

    A real signal's fourier transform is only stored in the half space, since the
    other half is redundant. Negative `q_x` is therefore not stored --- but its
    value is still exactly known, from the transform's symmetries, and this
    function recovers it. That matters: roughly a fifth of the frequencies of a
    rotated grid are negative in `q_x`.

    **Arguments:**

    - `input`:
        The fourier transform of a real 2D image or 3D volume, as prepared by
        [`cryojax.ndimage.prepare_sampling_fft`][]. Shape `(dim, dim // 2 + 1)`
        or `(dim, dim, dim // 2 + 1)`.
    - `frequencies`:
        The frequencies to interpolate at, in cycles/pixel, of shape
        `(..., ndim)` and ordered `(q_x, q_y)` or `(q_x, q_y, q_z)` --- as
        returned by [`cryojax.ndimage.make_frequency_grid`][]. `q_x` may be
        negative.
    - `order`:
        The order of the interpolation kernel: `1` for linear, `3` for cubic
        B-spline.
    - `mode`:
        What to return for frequencies that fall outside the fourier box, e.g. at
        the corners of a rotated grid. Either `"fill"` (the default), which
        returns zero, or `"clip"`, which clamps them onto the edge of the box.
    - `unroll`:
        See [`cryojax.ndimage.map_coordinates`][].

    **Returns:**

    The interpolated values, of shape `frequencies.shape[:-1]`.
    """
    input_arr = jnp.asarray(input)
    frequencies = jnp.asarray(frequencies)
    ndim = input_arr.ndim
    _check_ndim(ndim)
    dim = input_arr.shape[0]
    nyquist = dim // 2
    expected_shape = (dim,) * (ndim - 1) + (nyquist + 1,)
    if dim % 2 != 0 or input_arr.shape != expected_shape:
        raise ValueError(
            "`map_frequencies` expects a square, even-dimension half-space "
            "(rfft) array, of shape `(dim,) * (ndim - 1) + (dim // 2 + 1,)`, but "
            f"got an array of shape `{input_arr.shape}` (expected "
            f"`{expected_shape}`)."
        )
    if frequencies.shape[-1] != ndim:
        raise ValueError(
            f"`map_frequencies` expects `frequencies` of shape `(..., {ndim})`, "
            f"but got shape `{frequencies.shape}`."
        )

    # Frequencies (cycles/pixel) to index coordinates. The truncated axis is in
    # the rfft/corner convention (no offset); the others are `fftshift`ed to the
    # center convention. Array axes run (z, y, x), so the centered coordinates are
    # the frequency components in reverse order.
    coordinate_x = frequencies[..., 0] * dim
    coordinates_centered = [
        frequencies[..., i] * dim + nyquist for i in reversed(range(1, ndim))
    ]

    # Only `q_x >= 0` is stored. Reflect the query through the origin whenever it
    # is negative, and conjugate the result to correct for it. This is exact: it
    # happens once per query point, on the continuous coordinate, before any taps
    # are generated. Negating a frequency maps a centered index `j` to `dim - j`.
    is_reflected = coordinate_x < 0
    coordinate_x = jnp.abs(coordinate_x)
    coordinates_centered = [
        jnp.where(is_reflected, dim - k, k) for k in coordinates_centered
    ]

    if mode == "clip":
        coordinate_x = jnp.clip(coordinate_x, 0.0, float(nyquist))
        coordinates_centered = [
            jnp.clip(k, 0.0, float(dim)) for k in coordinates_centered
        ]
        is_in_box = None
    elif mode == "fill":
        # In the box iff every frequency is at most Nyquist. The truncated axis's
        # coordinate is now non-negative; the centered axes run from index 0
        # (frequency `-dim // 2`) to index `dim` (`+dim // 2`, the same aliased
        # frequency).
        is_in_box = coordinate_x <= nyquist
        for k in coordinates_centered:
            is_in_box = is_in_box & (k >= 0) & (k <= dim)
    else:
        raise ValueError(
            f"Invalid value `mode={mode!r}` in `map_frequencies`. "
            "Supported values are 'fill' and 'clip'."
        )

    taps_fn = indices_and_weights_fn(order)
    taps = [taps_fn(k) for k in [*coordinates_centered, coordinate_x]]

    result = _interp_taps(
        input_arr,
        taps,
        resolvers=[_centered_resolver(dim)] * (input_arr.ndim - 1) + [_unused_resolver],
        resolve_last=lambda index: _fold_truncated_axis(index, dim, nyquist),
        # Every tap index is resolved by symmetry, so none can be out of bounds.
        mode="promise_in_bounds",
        cval=0.0,
        unroll=unroll,
        couples_axes=True,
    )
    result = jnp.where(is_reflected, jnp.conj(result), result)
    if is_in_box is not None:
        result = jnp.where(is_in_box, result, jnp.zeros((), dtype=result.dtype))
    return result


def deconvolve_interpolation_kernel(real_array: Array, sinc_power: int) -> Array:
    """Divide a real-space signal by the transfer function of an interpolation
    kernel, which for an order-`p` B-spline is `sinc^(p + 1)`.

    Interpolating a signal's DFT with a kernel returns, not the true fourier
    transform, but the transform of the signal *multiplied in real space* by that
    kernel's transfer function. Dividing it out beforehand cancels that exactly.
    See `prepare_sampling_fft`.
    """
    factor = jnp.ones((), dtype=real_array.dtype)
    for axis, size in enumerate(real_array.shape):
        sinc = jnp.sinc(make_1d_coordinate_grid(size) / size) ** sinc_power
        other_axes = [a for a in range(real_array.ndim) if a != axis]
        factor = factor * jnp.expand_dims(sinc, other_axes)
    return real_array / factor


# ============================================================================
# The shared separable gather
#
# `_interp_taps` reduces the taps against the array one axis at a time. This is
# algebraically equivalent to (but cheaper than) forming the outer product of
# per-axis weights for every tap combination: for `n` taps per axis, that spends
# an extra multiply combining weights for each of the `n^ndim` combinations,
# whereas reducing one axis at a time spends only one multiply per gathered value.
#
# The *last* array axis is reduced outermost, because it is the axis whose
# resolver may fold (`map_frequencies`'s truncated axis). When it does, the
# remaining axes' indices depend on that fold, and the gathered block must be
# conjugated -- and since the weights are real, the conjugation commutes with the
# weighted sum, so it can be applied once per outer tap rather than per element.
#
# `map_coordinates` passes a `resolve_last` that never folds, so the same code
# collapses to an ordinary separable interpolation.
# ============================================================================


def _interp_taps(
    array: Array,
    taps: Sequence[Taps],
    *,
    resolvers: Sequence[Resolver],
    resolve_last: Callable[[Array], tuple[Array, Bool[Array, "..."] | None]],
    mode: str,
    cval: float | complex,
    unroll: bool,
    couples_axes: bool = False,
) -> Array:
    """Gather and reduce interpolation taps against `array`.

    **Arguments:**

    - `taps`: per array axis, that axis's `(index, weight)` kernel taps.
    - `resolvers`: per array axis, how to turn a tap index into a valid array
      index. The entry for the last axis is unused; `resolve_last` handles it.
    - `resolve_last`: resolves the last axis's tap index, and reports whether the
      gathered value must be conjugated (`None` if it never must).
    - `couples_axes`: if `True`, the leading axes' resolvers depend on the last
      axis's fold, so their indices gain a leading tap axis.
    """
    impl = _interp_loop if unroll else _interp_single_gather
    return impl(array, taps, resolvers, resolve_last, mode, cval, couples_axes)


def _interp_loop(
    array: Array,
    taps: Sequence[Taps],
    resolvers: Sequence[Resolver],
    resolve_last: Callable[[Array], tuple[Array, Bool[Array, "..."] | None]],
    mode: str,
    cval: float | complex,
    couples_axes: bool,
) -> Array:
    """Gather one tap at a time. Never holds more than one tap's worth of gathered
    values at once, at the cost of `n^ndim` separate gather calls."""
    del couples_axes  # each tap is resolved on its own; nothing to broadcast
    *taps_leading, taps_last = taps
    total = cast(Array, None)
    for index_last, weight_last in taps_last:
        index_last, needs_conjugate = resolve_last(index_last)
        block = _reduce_leading(
            array, taps_leading, resolvers, index_last, needs_conjugate, mode, cval
        )
        if needs_conjugate is not None:
            block = jnp.where(needs_conjugate, jnp.conj(block), block)
        total = block * weight_last if total is None else total + block * weight_last
    return total


def _reduce_leading(
    array: Array,
    taps_leading: Sequence[Taps],
    resolvers: Sequence[Resolver],
    index_last: Array,
    needs_conjugate: Bool[Array, "..."] | None,
    mode: str,
    cval: float | complex,
) -> Array:
    """Weighted sum over the leading axes' taps, at a fixed last-axis index."""
    if len(taps_leading) == 1:
        total = cast(Array, None)
        for index_0, weight_0 in taps_leading[0]:
            resolved_0 = resolvers[0](index_0, needs_conjugate)
            value = array.at[resolved_0, index_last].get(mode=mode, fill_value=cval)
            total = value * weight_0 if total is None else total + value * weight_0
        return total

    taps_0, taps_1 = taps_leading
    total = cast(Array, None)
    for index_0, weight_0 in taps_0:
        resolved_0 = resolvers[0](index_0, needs_conjugate)
        row = cast(Array, None)
        for index_1, weight_1 in taps_1:
            resolved_1 = resolvers[1](index_1, needs_conjugate)
            value = array.at[resolved_0, resolved_1, index_last].get(
                mode=mode, fill_value=cval
            )
            row = value * weight_1 if row is None else row + value * weight_1
        total = row * weight_0 if total is None else total + row * weight_0
    return total


def _interp_single_gather(
    array: Array,
    taps: Sequence[Taps],
    resolvers: Sequence[Resolver],
    resolve_last: Callable[[Array], tuple[Array, Bool[Array, "..."] | None]],
    mode: str,
    cval: float | complex,
    couples_axes: bool,
) -> Array:
    """Consolidate every tap into one gather. Materializes the full
    tap-combination tensor at once, so peak memory scales with the number of query
    points -- but it is usually faster (see `map_coordinates`'s `unroll`)."""
    *taps_leading, taps_last = taps
    index_last, weight_last = _stack(taps_last)  # (*S, n_last)
    index_last, needs_conjugate = resolve_last(index_last)

    def resolve_leading(axis: int) -> tuple[Array, Array]:
        """Resolve one leading axis's taps. When the resolvers are coupled to the
        last axis's fold, the indices gain its tap dimension; otherwise they are
        independent of it and simply broadcast against it."""
        index, weight = _stack(taps_leading[axis])  # (*S, n_axis)
        if couples_axes:
            assert needs_conjugate is not None
            index = resolvers[axis](
                index[..., None, :], needs_conjugate[..., :, None]
            )  # (*S, n_last, n_axis)
        else:
            index = resolvers[axis](index, None)[..., None, :]  # (*S, 1, n_axis)
        return index, weight

    if len(taps_leading) == 1:
        index_0, weight_0 = resolve_leading(0)
        gathered = array.at[index_0, index_last[..., :, None]].get(
            mode=mode, fill_value=cval
        )  # (*S, n_last, n_0)
        over_leading = jnp.sum(gathered * weight_0[..., None, :], axis=-1)
    else:
        index_0, weight_0 = resolve_leading(0)
        index_1, weight_1 = resolve_leading(1)
        gathered = array.at[
            index_0[..., :, :, None],
            index_1[..., :, None, :],
            index_last[..., :, None, None],
        ].get(mode=mode, fill_value=cval)  # (*S, n_last, n_0, n_1)
        over_1 = jnp.sum(gathered * weight_1[..., None, None, :], axis=-1)
        over_leading = jnp.sum(over_1 * weight_0[..., None, :], axis=-1)

    if needs_conjugate is not None:
        over_leading = jnp.where(needs_conjugate, jnp.conj(over_leading), over_leading)
    return jnp.sum(over_leading * weight_last, axis=-1)


def _stack(taps: Taps) -> tuple[Array, Array]:
    """Stack one axis's taps into `(indices, weights)` of shape
    `(*query_shape, n_taps)`."""
    return (
        jnp.stack([index for index, _ in taps], axis=-1),
        jnp.stack([jnp.asarray(weight) for _, weight in taps], axis=-1),
    )


# ============================================================================
# Boundary resolvers
# ============================================================================


def _bounded_resolver(size: int, mode: str) -> Resolver:
    """Resolve a tap index on an ordinary, independent axis of length `size`.

    JAX's gather modes only treat `index >= size` as out of bounds. A *negative*
    index is instead given numpy's wrap-around meaning (`array[-1]` is the last
    element) before the mode is ever consulted -- under every mode. So without
    this, a tap below the low edge silently reads from the *opposite* edge.

    Remapping negatives onto `size` puts them unambiguously out of range at the
    high edge, where the mode does apply. `"clip"` is the exception: it must clamp
    to the *near* edge, so clamp it here directly.
    """

    def resolve(index: Array, needs_conjugate: Bool[Array, "..."] | None) -> Array:
        del needs_conjugate  # this axis is independent of every other
        if mode == "promise_in_bounds":
            return index
        if mode == "clip":
            return jnp.clip(index, 0, size - 1)
        return jnp.where(index < 0, size, index)

    return resolve


def _centered_resolver(dim: int) -> Resolver:
    """Resolve a tap index on a centered (`fftshift`ed) axis of a half-space DFT,
    on which array index `j` holds frequency `j - dim // 2`.

    The DFT is periodic, so indices wrap modulo `dim`. And when the truncated axis
    was conjugate-folded, this axis's frequency is negated too, mapping index `j`
    to `dim - j`.
    """

    def resolve(index: Array, needs_conjugate: Bool[Array, "..."] | None) -> Array:
        if needs_conjugate is None:
            return index % dim
        return jnp.where(needs_conjugate, -index, index) % dim

    return resolve


def _fold_truncated_axis(
    index: Array, dim: int, nyquist: int
) -> tuple[Array, Bool[Array, "..."]]:
    """Resolve a tap index on the truncated (rfft) axis, which stores only
    frequencies `[0, nyquist]`. Frequencies outside that range are not stored, but
    are exactly determined by Hermitian symmetry -- at the price of also negating
    the centered axes and conjugating the gathered value, so report whether that
    happened."""
    needs_conjugate = (index < 0) | (index > nyquist)
    folded = jnp.where(
        index < 0,
        -index,  # F[-k] == conj(F[k])
        jnp.where(index > nyquist, dim - index, index),  # k > nyquist wraps to k - dim
    )
    # Query points outside the box can push taps beyond even the folded range.
    # Their values are discarded by `boundary`, but the gather still runs, so keep
    # the index valid rather than relying on gather clamping.
    return jnp.clip(folded, 0, nyquist), needs_conjugate


def _unused_resolver(index: Array, needs_conjugate: Bool[Array, "..."] | None) -> Array:
    raise AssertionError("the last axis is resolved by `resolve_last`")


def _check_ndim(ndim: int) -> None:
    if ndim not in (2, 3):
        raise NotImplementedError(
            f"Only 2D or 3D inputs are supported, but got an input of dimension {ndim}."
        )


def _check_coordinates(array: Array, coordinates: Sequence[Array]) -> list[Array]:
    _check_ndim(array.ndim)
    if len(coordinates) != array.ndim:
        raise ValueError(
            f"Coordinates must be a sequence of length {array.ndim}, but found "
            f"that it was equal to {len(coordinates)}."
        )
    return [jnp.asarray(c) for c in coordinates]


# ============================================================================
# Interpolation kernels, as (index, weight) taps
#
# Each returns the taps of the separable convolution kernel at `coordinate`: the
# array indices it reads, and the weight applied to each. Weights are a partition
# of unity (they sum to 1 at every coordinate), so both kernels reproduce a
# constant exactly.
# ============================================================================


def indices_and_weights_fn(order: int) -> Callable[[Array], Taps]:
    """The tap-generating function for an interpolation kernel of the given
    `order`."""
    if order == 1:
        return _linear_indices_and_weights
    elif order == 3:
        return _cubic_indices_and_weights
    else:
        raise ValueError(
            f"Interpolation `order` must be either `1` (linear) or `3` (cubic "
            f"B-spline), but got `order={order}`."
        )


def _linear_indices_and_weights(coordinate: Array) -> Taps:
    """The two taps of the linear interpolation kernel at `coordinate`, i.e. the
    nodes `floor(coordinate) + [0, 1]`. The kernel is the first-order B-spline
    (the unit box convolved with itself), whose Fourier transform is `sinc^2`."""
    lower = jnp.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _cubic_indices_and_weights(coordinate: Array) -> Taps:
    """The four taps of the cubic B-spline kernel at `coordinate`, i.e. the nodes
    `floor(coordinate) + [-1, 0, 1, 2]`. The kernel is the third-order B-spline
    (the unit box convolved with itself four times), whose Fourier transform is
    `sinc^4`.

    Note the tap at `floor(coordinate) - 1`: unlike linear interpolation, the
    cubic kernel always reaches one node *below* the coordinate's cell. Callers
    resolving these taps against an array must account for it -- that node lies
    below the low edge even for query points comfortably inside the array.
    """
    floor = jnp.floor(coordinate)
    base_index = floor.astype(jnp.int32)
    frac = coordinate - floor
    return [
        (base_index + offset - 1, _cubic_basis(frac - offset + 1)) for offset in range(4)
    ]


def _cubic_basis(t: Array) -> Array:
    """The cubic B-spline basis function, normalized so that the four taps at any
    coordinate sum to 1."""
    at = jnp.abs(t)
    fn1 = lambda t: (2 - t) ** 3
    fn2 = lambda t: 4 - 6 * t**2 + 3 * t**3
    return (
        jnp.where(
            at >= 1,
            jnp.where(at <= 2, fn1(at), 0),  # type: ignore
            jnp.where(at <= 1, fn2(at), 0),  # type: ignore
        )
        / 6
    )


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)
