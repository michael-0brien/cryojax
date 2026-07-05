"""
These versions of the scipy function map_coordinates is modified from Louis Desdoigts's
version: https://github.com/LouisDesdoigts/jax/blob/cubic-spline-updated/jax/_src/scipy/ndimage.py.

This code was developed for the project [`dLux`](https://louisdesdoigts.github.io/dLux/).
"""

import functools
import operator
from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import lineax as lx
from jax import lax, vmap
from jaxtyping import Array, ArrayLike


def map_coordinates(
    input: Array,
    coordinates: Sequence[Array],
    out_of_bounds_mode: str = "fill",
    fill_value: float | complex = 0.0,
    gather_mode: Literal["loop", "single_gather"] = "loop",
) -> Array:
    """
    Similar to `scipy.map_coordinates`, but diverges from the API. Always
    uses linear interpolation; for cubic spline interpolation, precompute
    coefficients with [`cryojax.ndimage.compute_spline_coefficients`][] and
    call [`cryojax.ndimage.map_coordinates_spline`][] instead.

    Adapted from Louis Desdoigts's [version of `jax.scipy.map_coordinates`](https://github.com/LouisDesdoigts/jax/blob/cubic-spline-updated/jax/_src/scipy/ndimage.py),
    which was developed for the project [`dLux`](https://louisdesdoigts.github.io/dLux/).

    **Arguments:**

    - `input`:
        The 2D or 3D array to interpolate.
    - `coordinates`:
        A sequence of length `input.ndim`, one coordinate array per axis
        (in index/pixel units, e.g. `input[0, 0]`/`input[0, 0, 0]` sits at
        coordinate `(0, 0)`/`(0, 0, 0)`), each broadcastable to the same
        shape. The output has that (broadcast) shape.
    - `out_of_bounds_mode` :
        Uses built-in [JAX out-of-bounds indexing](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html)
        to determine how to extrapolate beyond boundaries.
    - `fill_value`:
        The value used for out-of-bounds coordinates when
        `out_of_bounds_mode` is `"fill"`. Ignored for other modes.
    - `gather_mode`:
        See [`cryojax.ndimage.map_coordinates_spline`][] for a description of
        `"loop"` vs `"single_gather"`. For linear interpolation, `"loop"`
        (the default) was found to be as fast or faster than
        `"single_gather"` across every batch size tested on GPU, so there is
        little reason to change this unless benchmarking your own workload
        says otherwise.

    **Returns:**

    The array of values interpolated at `coordinates`, with the shape that
    the coordinate arrays broadcast to.
    """  # noqa: E501
    input_arr = jnp.asarray(input)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]
    _check_ndim(input_arr.ndim, coordinate_arrs)
    taps = [_linear_indices_and_weights(c) for c in coordinate_arrs]
    result = _interp_dispatch(
        input_arr, taps, out_of_bounds_mode, fill_value, gather_mode
    )
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def map_coordinates_spline(
    coefficients: Array,
    coordinates: Sequence[Array],
    out_of_bounds_mode: str = "fill",
    fill_value: float | complex = 0.0,
    gather_mode: Literal["loop", "single_gather"] = "single_gather",
) -> Array:
    """
    Similar to `scipy.map_coordinates`, but takes coefficients computed from
    [`cryojax.ndimage.compute_spline_coefficients`][] as input.

    **Arguments:**

    - `coefficients`:
        The precomputed cubic spline coefficients of a 2D or 3D array, from
        [`cryojax.ndimage.compute_spline_coefficients`][]. Note that these
        are *not* the same as the original array's values -- always compute
        them with `compute_spline_coefficients` first, never pass a raw
        data array here directly.
    - `coordinates`:
        A sequence of length `coefficients.ndim`, one coordinate array per
        axis (in index/pixel units, e.g. coordinate `(0, 0)`/`(0, 0, 0)`
        corresponds to the origin of the array that
        `compute_spline_coefficients` was called on), each broadcastable to
        the same shape. The output has that (broadcast) shape.
    - `out_of_bounds_mode` :
        Uses built-in [JAX out-of-bounds indexing](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html)
        to determine how to extrapolate beyond boundaries.
    - `fill_value`:
        The value used for out-of-bounds coordinates when
        `out_of_bounds_mode` is `"fill"`. Ignored for other modes.
    - `gather_mode`:
        Cubic spline interpolation gathers a `4^ndim` neighborhood around
        each query point (vs. `2^ndim` for linear). Two ways to compute
        that:

        - `"single_gather"` (the default): consolidate all `4^ndim` taps
          into one gather call. Substantially faster in most cases tested
          (often several-fold, occasionally much more, on GPU), but its
          peak memory scales linearly with the number of query points
          (batch size), since it materializes the full tap-combination
          tensor at once -- at large batch sizes this speed advantage
          shrinks (towards, but not below, parity in every case tested) and
          peak memory can be several-fold higher than `"loop"` for the same
          input.
        - `"loop"`: gather one tap combination at a time and accumulate.
          Never holds more than one tap's worth of gathered values at once,
          so its memory footprint is much smaller and more predictable at
          large batch sizes, at the cost of being substantially slower in
          most cases tested.

        If you hit GPU out-of-memory errors with the default, or are
        working with very large batches of query points, try `"loop"`.

    **Returns:**

    The array of values interpolated at `coordinates`, with the shape that
    the coordinate arrays broadcast to.
    """  # noqa: E501
    coefficients_arr = jnp.asarray(coefficients)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]
    _check_ndim(coefficients_arr.ndim, coordinate_arrs)
    taps = [_cubic_indices_and_weights(c) for c in coordinate_arrs]
    return _interp_dispatch(
        coefficients_arr, taps, out_of_bounds_mode, fill_value, gather_mode
    )


def compute_spline_coefficients(data: Array) -> Array:
    """Solve for the cubic spline coefficients of an input array, for later
    use with [`cryojax.ndimage.map_coordinates_spline`][].

    Solves a tridiagonal system independently along each axis (via
    `lineax`), so the cost is linear in the number of elements of `data`.
    Since it doesn't depend on the query coordinates, call this once and
    reuse the result across many `map_coordinates_spline` calls (e.g. many
    different sets of query points against the same underlying array)
    rather than recomputing it every time.

    **Arguments:**

    - `data`:
        The 2D or 3D array to compute cubic spline coefficients for.

    **Returns:**

    An array of the same shape and dtype as `data`, holding its spline
    coefficients. Pass this (not `data` itself) as the `coefficients`
    argument to `map_coordinates_spline`.
    """
    ndim = data.ndim
    for i in range(ndim):
        axis = ndim - i - 1
        # The tridiagonal operator's entries (`4.0`, `1.0`, `1.0`) are always
        # real, regardless of `data`'s dtype -- build it in the real dtype
        # underlying `data` (e.g. float32 for complex64) so that `data`
        # being complex doesn't force the elimination itself into complex
        # arithmetic (`_solve_coefficients` splits complex `data` into its
        # real and imaginary parts and reuses this one real operator for
        # both, which is ~2x cheaper than solving once with a complex
        # operator).
        A = _build_operator(data.shape[axis] - 2, dtype=_real_dtype(data.dtype))
        fn = lambda x: _solve_coefficients(x, A)
        for j in range(ndim - 2, -1, -1):
            ax = int(j >= axis)
            fn = vmap(fn, ax, ax)
        data = fn(data)
    return data


# ============================================================================
# Separable interpolation core, shared by linear and cubic-spline
# interpolation (`map_coordinates`/`map_coordinates_spline` above)
# ============================================================================


def _check_ndim(ndim: int, coordinate_arrs: Sequence[Array]) -> None:
    if len(coordinate_arrs) != ndim:
        raise ValueError(
            f"Coordinates must be a sequence of length {ndim}, but found that "
            f"it was equal to {len(coordinate_arrs)}."
        )
    if ndim not in (2, 3):
        raise NotImplementedError(
            f"map_coordinates/map_coordinates_spline only support 2D or 3D "
            f"inputs, but got an input of dimension {ndim}."
        )


def _interp_dispatch(
    array: Array,
    taps: Sequence[list[tuple[Array, ArrayLike]]],
    mode: str,
    fill_value: complex | float,
    gather_mode: Literal["loop", "single_gather"],
) -> Array:
    """Dispatch to the 2D/3D separable interpolator, using either the
    `"loop"` or `"single_gather"` gather strategy (see
    `map_coordinates_spline`'s `gather_mode` argument for the tradeoffs).
    `taps` holds, per axis, the (index, weight) pairs of that axis's
    interpolation kernel (2 taps for linear, 4 for cubic)."""
    if gather_mode == "loop":
        impl_2d, impl_3d = _interp_2d_loop, _interp_3d_loop
    elif gather_mode == "single_gather":
        impl_2d, impl_3d = _interp_2d_single_gather, _interp_3d_single_gather
    else:
        raise ValueError(
            f"`gather_mode` must be 'loop' or 'single_gather', but got '{gather_mode}'."
        )
    if len(taps) == 2:
        return impl_2d(array, taps[0], taps[1], mode, fill_value)
    else:
        return impl_3d(array, taps[0], taps[1], taps[2], mode, fill_value)


def _combine_taps(values: Sequence[Array], weights: Sequence[ArrayLike]) -> Array:
    """Weighted sum of already-gathered values along one axis's
    interpolation taps."""
    return functools.reduce(operator.add, (v * w for v, w in zip(values, weights)))


def _interp_2d_loop(
    array: Array,
    taps_y: list[tuple[Array, ArrayLike]],
    taps_x: list[tuple[Array, ArrayLike]],
    mode: str,
    fill_value: complex | float,
) -> Array:
    """Separable bilinear/bicubic interpolation: interpolate along x first,
    then along y, one tap combination at a time. This is algebraically
    equivalent to (but cheaper than) forming the outer product of per-axis
    weights for every tap combination independently: for `n` taps per axis,
    that approach spends an extra multiply combining per-axis weights for
    each of the `n^2` combinations on top of the multiply against the
    gathered value, whereas interpolating one axis at a time spends only
    one multiply per gathered value, with no separate weight-combining
    step. Never holds more than one tap's worth of gathered values at once
    (unlike `_interp_2d_single_gather`), at the cost of `n^2` separate
    gather calls instead of one.
    """
    row_results = []
    row_weights = []
    for iy, wy in taps_y:
        col_values = []
        col_weights = []
        for ix, wx in taps_x:
            col_values.append(array.at[iy, ix].get(mode=mode, fill_value=fill_value))
            col_weights.append(wx)
        row_results.append(_combine_taps(col_values, col_weights))
        row_weights.append(wy)
    return _combine_taps(row_results, row_weights)


def _interp_3d_loop(
    array: Array,
    taps_z: list[tuple[Array, ArrayLike]],
    taps_y: list[tuple[Array, ArrayLike]],
    taps_x: list[tuple[Array, ArrayLike]],
    mode: str,
    fill_value: complex | float,
) -> Array:
    """Separable trilinear/tricubic interpolation (see `_interp_2d_loop`):
    interpolate along x, then y, then z, rather than combining all three
    axes' weights independently for every tap combination."""
    z_results = []
    z_weights = []
    for iz, wz in taps_z:
        y_results = []
        y_weights = []
        for iy, wy in taps_y:
            x_values = []
            x_weights = []
            for ix, wx in taps_x:
                x_values.append(
                    array.at[iz, iy, ix].get(mode=mode, fill_value=fill_value)
                )
                x_weights.append(wx)
            y_results.append(_combine_taps(x_values, x_weights))
            y_weights.append(wy)
        z_results.append(_combine_taps(y_results, y_weights))
        z_weights.append(wz)
    return _combine_taps(z_results, z_weights)


def _interp_2d_single_gather(
    array: Array,
    taps_y: list[tuple[Array, ArrayLike]],
    taps_x: list[tuple[Array, ArrayLike]],
    mode: str,
    fill_value: complex | float,
) -> Array:
    """Separable bilinear/bicubic interpolation via one consolidated
    gather: stack all taps per axis and gather every tap combination in a
    single (broadcasted, fancy-indexed) call, then reduce one axis at a
    time (as in `_interp_2d_loop`, to avoid ever forming the outer product
    of weights). Materializes an `(*array.shape, n_y, n_x)`-shaped gathered
    array all at once, so peak memory scales with the number of query
    points -- see `map_coordinates_spline`'s `gather_mode` argument.
    """
    iy = jnp.stack([i for i, _ in taps_y], axis=-1)  # (*S, n_y)
    wy = jnp.stack([w for _, w in taps_y], axis=-1)
    ix = jnp.stack([i for i, _ in taps_x], axis=-1)  # (*S, n_x)
    wx = jnp.stack([w for _, w in taps_x], axis=-1)
    gathered = array.at[iy[..., :, None], ix[..., None, :]].get(
        mode=mode, fill_value=fill_value
    )  # (*S, n_y, n_x)
    after_x = jnp.sum(gathered * wx[..., None, :], axis=-1)  # (*S, n_y)
    return jnp.sum(after_x * wy, axis=-1)


def _interp_3d_single_gather(
    array: Array,
    taps_z: list[tuple[Array, ArrayLike]],
    taps_y: list[tuple[Array, ArrayLike]],
    taps_x: list[tuple[Array, ArrayLike]],
    mode: str,
    fill_value: complex | float,
) -> Array:
    """Separable trilinear/tricubic interpolation via one consolidated
    gather (see `_interp_2d_single_gather`)."""
    iz = jnp.stack([i for i, _ in taps_z], axis=-1)
    wz = jnp.stack([w for _, w in taps_z], axis=-1)
    iy = jnp.stack([i for i, _ in taps_y], axis=-1)
    wy = jnp.stack([w for _, w in taps_y], axis=-1)
    ix = jnp.stack([i for i, _ in taps_x], axis=-1)
    wx = jnp.stack([w for _, w in taps_x], axis=-1)
    gathered = array.at[
        iz[..., :, None, None], iy[..., None, :, None], ix[..., None, None, :]
    ].get(mode=mode, fill_value=fill_value)  # (*S, n_z, n_y, n_x)
    after_x = jnp.sum(gathered * wx[..., None, None, :], axis=-1)
    after_y = jnp.sum(after_x * wy[..., None, :], axis=-1)
    return jnp.sum(after_y * wz, axis=-1)


#
# Linear interpolation utilities
#
def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _linear_indices_and_weights(
    coordinate: Array,
) -> list[tuple[Array, ArrayLike]]:
    lower = jnp.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


#
# Spline interpolation utilities
#
def _real_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """The real dtype underlying `dtype` (e.g. `float32` for `complex64`;
    unchanged for already-real dtypes)."""
    return jnp.zeros((), dtype=dtype).real.dtype


def _build_operator(
    n: int, dtype: jnp.dtype | None = None, diag_value: float = 4.0
) -> lx.TridiagonalLinearOperator:
    diagonal = jnp.full((n,), diag_value, dtype=dtype)
    lower_diagonal = jnp.full((n - 1,), 1.0, dtype=dtype)
    upper_diagonal = jnp.full((n - 1,), 1.0, dtype=dtype)
    return lx.TridiagonalLinearOperator(diagonal, lower_diagonal, upper_diagonal)


def _construct_vector(data: Array, c2: Array, cnp2: Array) -> Array:
    yvec = data[1:-1]
    first = data[1] - c2
    last = data[-2] - cnp2
    yvec = yvec.at[0].set(first)
    yvec = yvec.at[-1].set(last)
    return yvec


def _solve_coefficients(
    data: Array, operator: lx.TridiagonalLinearOperator, h=1
) -> Array:
    # Calcualte second and second last coefficients
    c2 = 1 / 6 * data[0]
    cnp2 = 1 / 6 * data[-1]

    # Solve for internal cofficients. `operator` is always real (see
    # `compute_spline_coefficients`); if `data` (and so `yvec`) is complex,
    # solve the real and imaginary parts separately against that same real
    # operator and recombine, rather than solving once with a complex
    # right-hand side against a complex-cast copy of `operator` -- the
    # latter forces the whole elimination into (~2x costlier) complex
    # arithmetic even though the matrix itself never needed to be complex.
    yvec = _construct_vector(data, c2, cnp2)
    if jnp.iscomplexobj(yvec):
        cs_real = lx.linear_solve(operator, yvec.real).value
        cs_imag = lx.linear_solve(operator, yvec.imag).value
        cs = cs_real + 1j * cs_imag
    else:
        cs = lx.linear_solve(operator, yvec).value

    # Calculate first and last coefficients
    c1 = 2 * c2 - cs[0]
    cnp3 = 2 * cnp2 - cs[-1]
    return jnp.concatenate([jnp.array([c1, c2]), cs, jnp.array([cnp2, cnp3])])


def _spline_basis(t: Array) -> Array:
    at = jnp.abs(t)
    fn1 = lambda t: (2 - t) ** 3
    fn2 = lambda t: 4 - 6 * t**2 + 3 * t**3
    return jnp.where(
        at >= 1,
        jnp.where(at <= 2, fn1(at), 0),  # type: ignore
        jnp.where(at <= 1, fn2(at), 0),  # type: ignore
    )


def _cubic_indices_and_weights(
    coordinate: Array,
) -> list[tuple[Array, ArrayLike]]:
    """The four (index, weight) taps of the cubic B-spline convolution
    kernel at `coordinate`, i.e. `floor(coordinate) + [0, 1, 2, 3]` (matching
    the previous `_spline_point`/`_spline_value` implementation's indexing
    convention) weighted by `_spline_basis`."""
    floor = jnp.floor(coordinate)
    base_index = floor.astype(jnp.int32)
    frac = coordinate - floor
    return [
        (base_index + offset, _spline_basis(frac - offset + 1)) for offset in range(4)
    ]
