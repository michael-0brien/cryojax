"""Pallas/Triton kernels for the native NUFFT's PSWF spreading and interpolation, with
the tap convention and Horner coefficients of `spread.py`.

Each program takes `block_size` points. It evaluates every axis's `w` tap weights once
per point by Horner, with the coefficients baked in as constants from a static
`kernel_key`, and keeps them as `w` values per lane; the neighbourhood walk then selects
an outer axis's weight by its loop index and unrolls the innermost axis. The scatter
atomically adds into a zeros buffer aliased into its output; the gather is a masked
load. Masked lanes hold unspecified values and are never stored. Pallas-Triton rules as
in `_spreading/pallas_spread.py`.
"""

import math
from functools import cache

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap
from jaxtyping import Array, Float

from .._pallas_helpers import _choose_block_size, _CompilerParams, _valid_lane_mask
from .kernel import PSWFKernel


def make_kernel_key(kernel: PSWFKernel) -> tuple[int, int, bytes]:
    """The static `(width, degree, coefficient bytes)` the kernels are built from."""
    coefficients = np.ascontiguousarray(kernel.coefficients, np.float64)
    return kernel.width, coefficients.shape[1] - 1, coefficients.tobytes()


def pallas_nufft_spread(
    u: Float[Array, "M d"],
    c: Float[Array, "R M"],
    fine_shape: tuple[int, ...],
    kernel_key: tuple[int, int, bytes],
) -> Float[Array, "R ..."]:
    """Spread the `R` strength vectors `c` onto the periodic grid `fine_shape`, as
    `spread.spread`. A `jax.vmap` over `c` alone runs one launch."""

    def spread_one(u, c):
        return _spread(u, c, fine_shape, kernel_key)

    batchable = custom_vmap(spread_one)

    @batchable.def_vmap
    def _rule(axis_size, in_batched, u, c):
        if in_batched[0]:
            in_axes = (0, 0 if in_batched[1] else None)
            return jax.vmap(spread_one, in_axes=in_axes)(u, c), True
        grids = spread_one(u, c.reshape(-1, c.shape[-1]))
        return grids.reshape(axis_size, -1, *fine_shape), True

    return batchable(u, c)


def pallas_nufft_interp(
    u: Float[Array, "M d"],
    g: Float[Array, "R ..."],
    weights: Float[Array, "R M"] | None,
    fine_shape: tuple[int, ...],
    kernel_key: tuple[int, int, bytes],
    values: bool,
    positions: bool,
) -> tuple[Float[Array, "R M"] | None, Float[Array, "M d"] | None]:
    """Interpolate the `R` periodic grids `g` at `u` (`values`) and/or compute
    `sum_r weights_r grad_u interp(u, g_r)` (`positions`), as `spread.py`. A `jax.vmap`
    over `g` and `weights` with `u` unbatched runs one launch."""
    weights = (weights,) if positions else ()

    def interp_one(u, g, *weights):
        out = _interp(u, g, weights, fine_shape, kernel_key, values, positions, 1)
        return (*out[:-1], out[-1][0]) if positions else out

    batchable = custom_vmap(interp_one)

    @batchable.def_vmap
    def _rule(axis_size, in_batched, u, g, *weights):
        out_batched = (True,) * (values + positions)
        if in_batched[0]:
            in_axes = tuple(0 if b else None for b in in_batched)
            return jax.vmap(interp_one, in_axes=in_axes)(u, g, *weights), out_batched
        flat = []
        for x, b in zip((g, *weights), in_batched[1:]):
            x = x if b else jnp.broadcast_to(x, (axis_size, *x.shape))
            flat.append(x.reshape(-1, *x.shape[2:]))
        g, *weights = flat
        out = _interp(u, g, weights, fine_shape, kernel_key, values, positions, axis_size)
        if values:
            out = (out[0].reshape(axis_size, -1, u.shape[0]), *out[1:])
        return out, out_batched

    out = list(batchable(u, g, *weights))
    return (out.pop(0) if values else None), (out.pop(0) if positions else None)


#
# Calls
#
def _spread(u, c, fine_shape, kernel_key):
    (num_grids, num_points), dtype = c.shape, u.dtype
    size = math.prod(fine_shape)
    if num_grids == 0 or num_points == 0:
        return jnp.zeros((num_grids, *fine_shape), dtype)
    block_size = _choose_block_size(kernel_key[0], len(fine_shape))
    kernel = _make_spread_kernel(
        fine_shape, kernel_key, num_grids, block_size, num_points
    )
    grids = pl.pallas_call(
        kernel,
        grid=(pl.cdiv(num_points, block_size),),
        in_specs=[
            *[_per_point_spec((num_points,), block_size)] * len(fine_shape),
            _per_point_spec((num_grids, num_points), block_size),
            pl.BlockSpec((num_grids, size), lambda p: (0, 0)),
        ],
        out_specs=pl.BlockSpec((num_grids, size), lambda p: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((num_grids, size), dtype),
        input_output_aliases={len(fine_shape) + 1: 0},
        compiler_params=_CompilerParams(),
    )(*u.T, c.astype(dtype), jnp.zeros((num_grids, size), dtype))
    return grids.reshape(num_grids, *fine_shape)


def _interp(u, g, weights, fine_shape, kernel_key, values, positions, num_groups):
    """`(values (R, M), u_bar (num_groups, M, d))`, those requested; the `R` grids form
    `num_groups` consecutive groups, each reducing its own `u_bar`."""
    (num_points, ndim), num_grids, dtype = u.shape, g.shape[0], u.dtype
    out_shapes = [(num_grids, num_points)] if values else []
    if positions:
        out_shapes += [(num_groups, num_points)] * ndim
    if num_grids == 0 or num_points == 0:
        out = [jnp.zeros(shape, dtype) for shape in out_shapes]
    else:
        block_size = _choose_block_size(kernel_key[0], ndim)
        kernel = _make_interp_kernel(
            fine_shape,
            kernel_key,
            num_grids,
            num_groups,
            block_size,
            num_points,
            values,
            positions,
        )
        size = math.prod(fine_shape)
        out = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(num_points, block_size),),
            in_specs=[
                *[_per_point_spec((num_points,), block_size)] * ndim,
                pl.BlockSpec((num_grids, size), lambda p: (0, 0)),
                *[_per_point_spec((num_grids, num_points), block_size)] * len(weights),
            ],
            out_specs=[_per_point_spec(shape, block_size) for shape in out_shapes],
            out_shape=[jax.ShapeDtypeStruct(shape, dtype) for shape in out_shapes],
            compiler_params=_CompilerParams(),
        )(
            *u.T,
            g.reshape(num_grids, size).astype(dtype),
            *(w.astype(dtype) for w in weights),
        )
    if positions:
        return (*out[:-ndim], jnp.stack(out[-ndim:], axis=-1))
    return tuple(out)


def _per_point_spec(shape, block_size):
    """Blocks of `block_size` points along the last axis, whole along the others."""
    return pl.BlockSpec(
        (*shape[:-1], block_size), lambda p: (0,) * (len(shape) - 1) + (p,)
    )


#
# Kernels
#
def _tap_polynomials(kernel_key, *, derivative):
    """Per tap, the Horner coefficients (highest power first) of the weight, or of its
    derivative in `u` (`d/du = -d/ds`), as Python floats."""
    width, degree, data = kernel_key
    coefficients = np.frombuffer(data, np.float64).reshape(width, degree + 1)
    if derivative:
        coefficients = -coefficients[:, :-1] * np.arange(degree, 0, -1)
    return coefficients.tolist()


def _horner(coefficients, s):
    out = coefficients[0] * s + coefficients[1]
    for coefficient in coefficients[2:]:
        out = out * s + coefficient
    return out


def _select(per_tap, t):
    """`per_tap[t]` for a traced loop index `t`."""
    out = per_tap[0]
    for k, value in enumerate(per_tap[1:], start=1):
        out = jnp.where(t == k, value, out)
    return out


def _axis_taps(u_refs, mask, width, polynomials):
    """Per grid axis (first to last): the first tap's index and every tap's
    polynomials at the point's offset, each `(block_size,)`. `u_refs` run (x, y, z)."""
    starts, per_axis = [], []
    for u_ref in reversed(u_refs):
        half = pltriton.load(u_ref, mask=mask) - width / 2
        start = jnp.ceil(half)
        s = start - half
        starts.append(start.astype(jnp.int32))
        per_axis.append([[_horner(p, s) for p in polys] for polys in polynomials])
    return starts, per_axis


@cache
def _make_spread_kernel(fine_shape, kernel_key, num_grids, block_size, num_points):
    ndim, width = len(fine_shape), kernel_key[0]
    strides = [math.prod(fine_shape[axis + 1 :]) for axis in range(ndim)]
    polynomials = [_tap_polynomials(kernel_key, derivative=False)]

    def kernel(*refs):
        u_refs, (c_ref, _, out_ref) = refs[:ndim], refs[ndim:]
        # The zeros input is aliased into `out_ref`
        mask = (
            _valid_lane_mask(block_size, num_points) if num_points % block_size else None
        )
        starts, per_axis = _axis_taps(u_refs, mask, width, polynomials)
        weights = [w for (w,) in per_axis]
        inner = [(starts[-1] + j) % fine_shape[-1] for j in range(width)]

        def scatter(r, axis, base, value):
            if axis == ndim - 1:
                for j in range(width):
                    pltriton.atomic_add(
                        out_ref, (r, base + inner[j]), value * weights[-1][j], mask=mask
                    )
                return

            def body(t, carry):
                index = (starts[axis] + t) % fine_shape[axis]
                weight = _select(weights[axis], t)
                scatter(r, axis + 1, base + index * strides[axis], value * weight)
                return carry

            jax.lax.fori_loop(0, width, body, 0)

        def grid_body(r, carry):
            scatter(r, 0, 0, pltriton.load(c_ref.at[r], mask=mask))
            return carry

        jax.lax.fori_loop(0, num_grids, grid_body, 0)

    return kernel


@cache
def _make_interp_kernel(
    fine_shape,
    kernel_key,
    num_grids,
    num_groups,
    block_size,
    num_points,
    values,
    positions,
):
    ndim, width = len(fine_shape), kernel_key[0]
    strides = [math.prod(fine_shape[axis + 1 :]) for axis in range(ndim)]
    polynomials = [_tap_polynomials(kernel_key, derivative=False)]
    if positions:
        polynomials.append(_tap_polynomials(kernel_key, derivative=True))
    grids_per_group = num_grids // num_groups

    def kernel(*refs):
        u_refs, (g_ref, *refs) = refs[:ndim], refs[ndim:]
        weights_ref = refs.pop(0) if positions else None
        values_ref = refs.pop(0) if values else None
        u_bar_refs = refs[::-1]  # (z, y, x), as the grid axes
        mask = (
            _valid_lane_mask(block_size, num_points) if num_points % block_size else None
        )
        starts, per_axis = _axis_taps(u_refs, mask, width, polynomials)
        inner = [(starts[-1] + j) % fine_shape[-1] for j in range(width)]
        zero = jnp.zeros((block_size,), g_ref.dtype)

        def gather(r, axis, base):
            """The value and, for `positions`, the gradient along axes `axis..` of the
            neighbourhood under `base`."""
            if axis == ndim - 1:
                value, gradient = zero, zero
                for j in range(width):
                    g = pltriton.load(g_ref.at[r, base + inner[j]], mask=mask)
                    value = value + g * per_axis[-1][0][j]
                    if positions:
                        gradient = gradient + g * per_axis[-1][1][j]
                return value, [gradient] if positions else []

            def body(t, carry):
                value, gradients = carry
                index = (starts[axis] + t) % fine_shape[axis]
                sub_value, sub_gradients = gather(
                    r, axis + 1, base + index * strides[axis]
                )
                weight = _select(per_axis[axis][0], t)
                if positions:
                    derivative = _select(per_axis[axis][1], t)
                    gradients = [gradients[0] + derivative * sub_value] + [
                        gradient + weight * sub
                        for gradient, sub in zip(gradients[1:], sub_gradients)
                    ]
                return value + weight * sub_value, gradients

            init = (zero, [zero] * (ndim - axis) if positions else [])
            return jax.lax.fori_loop(0, width, body, init)

        def group_body(group, carry):
            def grid_body(k, u_bar):
                r = group * grids_per_group + k
                value, gradients = gather(r, 0, 0)
                if values:
                    pltriton.store(values_ref.at[r], value, mask=mask)
                if positions:
                    weight = pltriton.load(weights_ref.at[r], mask=mask)
                    u_bar = [
                        a + weight * gradient for a, gradient in zip(u_bar, gradients)
                    ]
                return u_bar

            u_bar = jax.lax.fori_loop(
                0, grids_per_group, grid_body, [zero] * ndim if positions else []
            )
            for u_bar_ref, gradient in zip(u_bar_refs, u_bar):
                pltriton.store(u_bar_ref.at[group], gradient, mask=mask)
            return carry

        jax.lax.fori_loop(0, num_groups, group_body, 0)

    return kernel
