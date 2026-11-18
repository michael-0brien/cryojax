"""Pure-JAX PSWF spreading and interpolation on a periodic fine grid, with analytic VJPs.

Coordinates `u` are `(x, y[, z])` in fine-grid pixels, matching the last, ..., first grid
axis. A point touches `l = ceil(u - w/2) + j`, `j = 0..w-1`, weighted by per-tap Horner
polynomials in `s = ceil(u - w/2) - (u - w/2)` (see `kernel.py`), evaluated per axis once
per point. All functions are plain JAX, so forward mode through the VJPs gives HVPs.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .kernel import PSWFKernel, derivative_coefficients


def spread(
    u: Float[Array, "M d"],
    c: Float[Array, "R M"],
    fine_shape: tuple[int, ...],
    kernel: PSWFKernel,
) -> Float[Array, "R ..."]:
    """Spread the `R` strength vectors `c` onto the periodic grid `fine_shape`."""
    indices, offsets = _taps(u, fine_shape, kernel.width)
    weights = [_horner(kernel.coefficients, s) for s in offsets]
    products = weights[0]
    for axis_weights in weights[1:]:
        products = products[..., None] * _expand(axis_weights, products.ndim)
    num = c.shape[0]
    contributions = products[..., None] * c.T.reshape(
        c.shape[1:] + (1,) * len(weights) + (num,)
    )
    grid = jax.ops.segment_sum(
        contributions.reshape(math.prod(products.shape), num),
        _flat_indices(indices, fine_shape).ravel(),
        num_segments=math.prod(fine_shape),
    )
    return grid.T.reshape(num, *fine_shape)


def interp(
    u: Float[Array, "M d"], g: Float[Array, "R ..."], kernel: PSWFKernel
) -> Float[Array, "R M"]:
    """Interpolate the `R` periodic grids `g` at `u`."""
    indices, offsets = _taps(u, g.shape[1:], kernel.width)
    weights = [_horner(kernel.coefficients, s) for s in offsets]
    axes = _AXES[-len(weights) :]
    spec = f"m{axes}r," + ",".join(f"m{a}" for a in axes) + "->rm"
    return _einsum(spec, _gather(g, indices), *weights)


def spread_bwd(
    u: Float[Array, "M d"],
    c: Float[Array, "R M"],
    g_bar: Float[Array, "R ..."],
    kernel: PSWFKernel,
) -> tuple[Float[Array, "M d"], Float[Array, "R M"]]:
    """VJP of `spread`: `c_bar = interp(u, g_bar)` and
    `u_bar = sum_r c_r interp'(u, g_bar_r)`."""
    c_bar, u_bar = _interp_and_gradient(u, g_bar, c, kernel)
    return u_bar, c_bar


def interp_bwd(
    u: Float[Array, "M d"],
    g: Float[Array, "R ..."],
    f_bar: Float[Array, "R M"],
    kernel: PSWFKernel,
) -> tuple[Float[Array, "M d"], Float[Array, "R ..."]]:
    """VJP of `interp`: `g_bar = spread(u, f_bar)` and
    `u_bar = sum_r f_bar_r interp'(u, g_r)`."""
    _, u_bar = _interp_and_gradient(u, g, f_bar, kernel)
    return u_bar, spread(u, f_bar, g.shape[1:], kernel)


_AXES = "zyx"
_einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


def _taps(u, fine_shape, width):
    """Per grid axis (first to last): wrapped tap indices `(M, w)` and offsets `s`."""
    indices, offsets = [], []
    for axis, m in enumerate(fine_shape):
        u_axis = u[:, len(fine_shape) - 1 - axis]
        start = jnp.ceil(u_axis - width / 2)
        indices.append((start.astype(jnp.int32)[:, None] + np.arange(width)) % m)
        offsets.append(start - (u_axis - width / 2))
    return indices, offsets


def _horner(coefficients: np.ndarray, s: Array) -> Array:
    """Every tap's polynomial (highest power first) at `s`, shape `(M, w)`."""
    coefficients = jnp.asarray(coefficients, dtype=s.dtype)
    out = jnp.broadcast_to(coefficients[:, 0], s.shape + coefficients.shape[:1])
    for k in range(1, coefficients.shape[1]):
        out = out * s[:, None] + coefficients[:, k]
    return out


def _expand(axis_values, ndim):
    """`(M, w)` to `(M, 1, ..., 1, w)`, to broadcast along an `ndim`-array's last axis."""
    num, width = axis_values.shape
    return axis_values.reshape((num,) + (1,) * (ndim - 1) + (width,))


def _flat_indices(indices, fine_shape):
    """Flat grid index of every tap, `(M, w, ..., w)`."""
    flat = indices[0]
    for axis_indices, m in zip(indices[1:], fine_shape[1:]):
        flat = flat[..., None] * m + _expand(axis_indices, flat.ndim)
    return flat


def _gather(g, indices):
    """The grids' values at every tap, `(M, w, ..., w, R)`."""
    return g.reshape(g.shape[0], -1).T[_flat_indices(indices, g.shape[1:])]


def _interp_and_gradient(u, g, weights_r, kernel):
    """`interp(u, g)` and `sum_r weights_r grad_u interp(u, g_r)`, shape `(M, d)`."""
    indices, offsets = _taps(u, g.shape[1:], kernel.width)
    weights = [_horner(kernel.coefficients, s) for s in offsets]
    first_derivative = derivative_coefficients(kernel.coefficients, 1)
    derivatives = [_horner(first_derivative, s) for s in offsets]
    gathered = _gather(g, indices)
    axes = _AXES[-len(weights) :]
    gradient = []
    for a, axis_derivatives in zip(axes, derivatives):
        others = [(b, w) for b, w in zip(axes, weights) if b != a]
        spec = f"m{axes}r," + ",".join(f"m{b}" for b, _ in others) + f"->m{a}r"
        along_axis = _einsum(spec, gathered, *(w for _, w in others))
        gradient.append(_einsum("mwr,mw,rm->m", along_axis, axis_derivatives, weights_r))
    # `along_axis` now leaves the last grid axis
    values = _einsum("mwr,mw->rm", along_axis, weights[-1])
    # Grid axes run (z, y, x); `u` runs (x, y, z)
    return values, jnp.stack(gradient[::-1], axis=-1)
