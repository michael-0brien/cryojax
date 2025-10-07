"""
These versions of the scipy function map_coordinates is modified from Louis Desdoigts's
version: https://github.com/LouisDesdoigts/jax/blob/cubic-spline-updated/jax/_src/scipy/ndimage.py.

This code was developed for the project [`dLux`](https://louisdesdoigts.github.io/dLux/).
"""

import functools
import itertools
import operator
from collections.abc import Sequence

import jax.numpy as jnp
import lineax as lx
from jax import lax, vmap
from jaxtyping import Array, ArrayLike


def map_coordinates(
    input: Array,
    coordinates: Sequence[Array],
    order: int,
    mode: str = "fill",
    cval: float | complex = 0.0,
):
    """
    Similar to `scipy.map_coordinates`, but diverges from the API.

    Adapted from Louis Desdoigts's [version of `jax.scipy.map_coordinates`](https://github.com/LouisDesdoigts/jax/blob/cubic-spline-updated/jax/_src/scipy/ndimage.py),
    which was developed for the project [`dLux`](https://louisdesdoigts.github.io/dLux/).

    Arguments
    ---------
    mode :
        Uses built-in JAX out-of-bounds indexing to determine how to
        extrapolate beyond boundaries.
        See https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html.
    """
    if order in [0, 1]:
        return _map_coordinates_nn_or_linear(input, coordinates, order, mode, cval)
    elif order == 3:
        coefficients = compute_spline_coefficients(input)
        return _map_coordinates_with_cubic_spline(coefficients, coordinates, mode, cval)
    else:
        raise NotImplementedError(f"map_coordinates does not support order={order}.")


def map_coordinates_spline(
    coefficients: Array,
    coordinates: Sequence[Array],
    mode: str = "fill",
    cval: float | complex = 0.0,
):
    """
    Similar to scipy.map_coordinates, but takes cubic spline coefficients as input.

    Adapted from https://github.com/LouisDesdoigts/jax/blob/cubic-spline-updated/jax/_src/scipy/ndimage.py,
    which was developed for the project [dLux](https://louisdesdoigts.github.io/dLux/).

    Arguments
    ---------
    mode :
        Uses built-in JAX out-of-bounds indexing to determine how to
        extrapolate beyond boundaries.
        See https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html.

    """
    return _map_coordinates_with_cubic_spline(coefficients, coordinates, mode, cval)


def compute_spline_coefficients(data: Array) -> Array:
    """Solve for the spline coefficients of an input array."""
    ndim = data.ndim
    for i in range(ndim):
        axis = ndim - i - 1
        A = _build_operator(data.shape[axis] - 2, dtype=data.dtype)
        fn = lambda x: _solve_coefficients(x, A)
        for j in range(ndim - 2, -1, -1):
            ax = int(j >= axis)
            fn = vmap(fn, ax, ax)
        data = fn(data)
    return data


def _map_coordinates_nn_or_linear(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str,
    cval: float | complex,
) -> Array:
    input_arr = jnp.asarray(input)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]

    if len(coordinates) != input_arr.ndim:
        raise ValueError(
            f"Coordinates must be a sequence of length {input_arr.ndim}, "
            f"but found that it was equal to {len(coordinates)}."
        )

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError(
            "_map_coordinates_nn_or_linear requires order = 0 or 1."
        )

    interpolations_1d = []
    for coordinate in coordinate_arrs:
        interp_nodes = interp_fun(coordinate)
        interpolations_1d.append(interp_nodes)
    outputs = []
    for items in itertools.product(*interpolations_1d):
        indices, weights = _unzip2(items)
        contribution = input_arr.at[indices].get(mode=mode, fill_value=cval)
        interpolated = _nonempty_prod(weights) * contribution
        outputs.append(interpolated)
    result = _nonempty_sum(outputs)
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def _map_coordinates_with_cubic_spline(
    coefficients: ArrayLike,
    coordinates: Sequence[ArrayLike],
    mode: str,
    cval: complex | float,
) -> Array:
    coefficients = jnp.asarray(coefficients)
    if len(coordinates) != coefficients.ndim:
        raise ValueError(
            "coordinates must be a sequence of length coefficients.ndim, but "
            f"{len(coordinates)} != {coefficients.ndim}"
        )
    # Stack coordinates along one axis
    coords = jnp.stack([jnp.asarray(c) for c in coordinates], axis=0)
    # vmap spline evaluate over coordinates and return
    fn = lambda coord: _spline_point(coefficients, coord, mode, cval)
    return vmap(fn)(coords.reshape(coefficients.ndim, -1).T).reshape(coords.shape[1:])


#
# Nearest neighbor and linear interpolation utilities
#
def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(
    coordinate: Array,
) -> list[tuple[Array, ArrayLike]]:
    index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


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

    # Solve for internal cofficients
    yvec = _construct_vector(data, c2, cnp2)
    solution = lx.linear_solve(operator, yvec)
    cs = solution.value

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


def _spline_value(
    coefficients: Array,
    coordinate: Array,
    index: Array,
    mode: str,
    cval: complex | float,
) -> Array:
    coefficient = coefficients.at[tuple(index)].get(mode=mode, fill_value=cval)
    fn = vmap(lambda x, i: _spline_basis(x - i + 1), (0, 0))
    return coefficient * fn(coordinate, index).prod()


def _spline_point(
    coefficients: Array,
    coordinate: Array,
    mode: str,
    cval: complex | float,
) -> Array:
    index_fn = lambda x: (jnp.arange(0, 4) + jnp.floor(x)).astype(int)
    index_vals = vmap(index_fn)(coordinate)
    indices = jnp.array(jnp.meshgrid(*index_vals, indexing="ij"))
    fn = lambda index: _spline_value(coefficients, coordinate, index, mode, cval)
    return vmap(fn)(indices.reshape(coefficients.ndim, -1).T).sum()


#
# Utilities
#
def _unzip2(xys):
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)
