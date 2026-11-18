"""Pallas/Triton kernels for the native NUFFT's PSWF spreading and interpolation, in
the tap convention of `spread.py`.

A program takes `block_size` points and evaluates their tap weights once by Horner, the
coefficients baked in from a static `kernel_key`; the neighbourhood walk unrolls the
innermost axis. The scatter atomically adds into an aliased zeros buffer, the gather is
a masked load. As in `_spreading/pallas_spread.py`, each wrapper's `jax.custom_jvp`
tangent is a tangent-only kernel, specialized on symbolic zeros and batched over
directions.
"""

import itertools
import math
from functools import cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap
from jaxtyping import Array, Float

from .._pallas_helpers import (
    _add,
    _batchable,
    _choose_block_size,
    _CompilerParams,
    _is_zero,
    _mul,
    _valid_lane_mask,
)
from .kernel import PSWFKernel, derivative_coefficients


def make_kernel_key(kernel: PSWFKernel) -> tuple[int, int, bytes]:
    """The static `(width, degree, coefficient bytes)` the kernels are built from."""
    coefficients = np.ascontiguousarray(kernel.coefficients, np.float64)
    return kernel.width, coefficients.shape[1] - 1, coefficients.tobytes()


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
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
        # Explicit sizes, not -1: any of them may be 0
        _, num_grids, num_points = c.shape
        grids = spread_one(u, c.reshape(axis_size * num_grids, num_points))
        return grids.reshape(axis_size, num_grids, *fine_shape), True

    return batchable(u, c)


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6))
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
        # Explicit sizes, not -1: any of them may be 0
        num_grids, flat = (g.shape[1] if in_batched[1] else g.shape[0]), []
        for x, b in zip((g, *weights), in_batched[1:]):
            x = x if b else jnp.broadcast_to(x, (axis_size, *x.shape))
            flat.append(x.reshape(axis_size * num_grids, *x.shape[2:]))
        g, *weights = flat
        out = _interp(u, g, weights, fine_shape, kernel_key, values, positions, axis_size)
        if values:
            out = (out[0].reshape(axis_size, num_grids, u.shape[0]), *out[1:])
        return out, out_batched

    out = list(batchable(u, g, *weights))
    return (out.pop(0) if values else None), (out.pop(0) if positions else None)


#
# Forward-mode rules: the primal kernel and a separate tangent-only kernel, never
# fused, so that `jax.linearize` keeps the primal (see `_spreading/pallas_spread.py`)
#
def _spread_jvp(fine_shape, kernel_key, primals, tangents):
    out = pallas_nufft_spread(*primals, fine_shape, kernel_key)
    if all(_is_zero(t) for t in tangents):
        return out, jnp.zeros_like(out)
    return out, pallas_nufft_spread_tangent(*primals, tangents, fine_shape, kernel_key)


pallas_nufft_spread.defjvp(_spread_jvp, symbolic_zeros=True)


def _interp_jvp(fine_shape, kernel_key, values, positions, primals, tangents):
    out = pallas_nufft_interp(*primals, fine_shape, kernel_key, values, positions)
    tangents = tangents[: 2 + positions]
    if all(_is_zero(t) for t in tangents):
        return out, jax.tree.map(jnp.zeros_like, out)
    return out, pallas_nufft_interp_tangent(
        *primals, tangents, fine_shape, kernel_key, values, positions
    )


pallas_nufft_interp.defjvp(_interp_jvp, symbolic_zeros=True)


def pallas_nufft_spread_tangent(u, c, tangents, fine_shape, kernel_key):
    """The tangent of `pallas_nufft_spread` for `tangents = (tu, tc)`, symbolic zeros
    dropped at trace time. Under `jax.vmap` over the tangents, one launch."""
    nonzero = tuple(not _is_zero(t) for t in tangents)
    fn = _spread_tangent_batchable(fine_shape, kernel_key, nonzero)
    return fn(u, c, *(t for t in tangents if not _is_zero(t)))


def pallas_nufft_interp_tangent(
    u, g, weights, tangents, fine_shape, kernel_key, values, positions
):
    """The tangent of `pallas_nufft_interp` for `tangents = (tu, tg[, tweights])`,
    symbolic zeros dropped at trace time. Under `jax.vmap` over the tangents, one
    launch."""
    nonzero = tuple(not _is_zero(t) for t in tangents)
    fn = _interp_tangent_batchable(fine_shape, kernel_key, values, positions, nonzero)
    primals = (u, g, weights) if positions else (u, g)
    out = list(fn(*primals, *(t for t in tangents if not _is_zero(t))))
    return (out.pop(0) if values else None), (out.pop(0) if positions else None)


@cache
def _spread_tangent_batchable(fine_shape, kernel_key, nonzero):
    def impl(primals, tangents):
        return _spread_tangent(*primals, tangents, nonzero, fine_shape, kernel_key)

    def single(u, c, *tangents):
        return impl((u, c), tuple(t[None] for t in tangents))[0]

    return _batchable(single, 2, impl)


@cache
def _interp_tangent_batchable(fine_shape, kernel_key, values, positions, nonzero):
    num_primals = 2 + positions

    def impl(primals, tangents):
        u, g, *weights = primals
        return _interp_tangent(
            u,
            g,
            weights[0] if positions else None,
            tangents,
            nonzero,
            fine_shape,
            kernel_key,
            values,
            positions,
        )

    def single(*args):
        primals, tangents = args[:num_primals], args[num_primals:]
        return tuple(x[0] for x in impl(primals, tuple(t[None] for t in tangents)))

    return _batchable(single, num_primals, impl)


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
        name="nufft_spread",
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
            name="nufft_interp",
        )(
            *u.T,
            g.reshape(num_grids, size).astype(dtype),
            *(w.astype(dtype) for w in weights),
        )
    if positions:
        return (*out[:-ndim], jnp.stack(out[-ndim:], axis=-1))
    return tuple(out)


def _spread_tangent(u, c, tangents, nonzero, fine_shape, kernel_key):
    """`(D, R, *fine_shape)` for the `D` directions of the nonzero `tangents`: `tu`
    `(D, M, d)` and/or `tc` `(D, R, M)`."""
    tangents = iter(tangents)
    tu, tc = (next(tangents) if is_nonzero else None for is_nonzero in nonzero)
    (num_grids, num_points), dtype, ndim = c.shape, u.dtype, len(fine_shape)
    num_directions = (tc if tu is None else tu).shape[0]
    if num_grids == 0 or num_points == 0:
        return jnp.zeros((num_directions, num_grids, *fine_shape), dtype)
    block_size = _choose_block_size(kernel_key[0], ndim)
    per_point = partial(_per_point_spec, block_size=block_size)
    operands, in_specs = [*u.T], [per_point((num_points,))] * ndim
    if tu is not None:
        operands += [c.astype(dtype), *_per_axis_and_direction(tu.astype(dtype))]
        in_specs += [per_point((num_grids, num_points))]
        in_specs += [per_point((num_points,))] * (ndim * num_directions)
    if tc is not None:
        shape = (num_directions * num_grids, num_points)
        operands.append(tc.astype(dtype).reshape(shape))
        in_specs.append(per_point(shape))
    out_shape = (num_directions * num_grids, math.prod(fine_shape))
    out_spec = pl.BlockSpec(out_shape, lambda p: (0, 0))
    kernel = _make_spread_tangent_kernel(
        fine_shape, kernel_key, num_grids, block_size, num_points, nonzero, num_directions
    )
    grids = pl.pallas_call(
        kernel,
        grid=(pl.cdiv(num_points, block_size),),
        in_specs=[*in_specs, out_spec],
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct(out_shape, dtype),
        input_output_aliases={len(operands): 0},
        compiler_params=_CompilerParams(),
        name="nufft_spread_tangent",
    )(*operands, jnp.zeros(out_shape, dtype))
    return grids.reshape(num_directions, num_grids, *fine_shape)


def _interp_tangent(
    u, g, weights, tangents, nonzero, fine_shape, kernel_key, values, positions
):
    """The requested outputs' tangents, `(D, R, M)` for the values and `(D, M, d)` for
    the position gradient, for the `D` directions of the nonzero `tangents`: `tu`
    `(D, M, d)`, `tg` `(D, R, ...)`, `tweights` `(D, R, M)`."""
    nonzero = (*nonzero, False)[:3]
    tangents = iter(tangents)
    tu, tg, tw = (next(tangents) if is_nonzero else None for is_nonzero in nonzero)
    (num_points, ndim), num_grids, dtype = u.shape, g.shape[0], u.dtype
    num_directions = next(t for t in (tu, tg, tw) if t is not None).shape[0]
    out_values, g_order, uses_weights = _interp_tangent_plan(values, positions, nonzero)
    values_shape = (num_directions * num_grids, num_points)
    out_shapes = [values_shape] if out_values else []
    out_shapes += [(num_points,)] * (ndim * num_directions) if positions else []
    if num_grids == 0 or num_points == 0:
        out = [jnp.zeros(shape, dtype) for shape in out_shapes]
    else:
        block_size = _choose_block_size(kernel_key[0], ndim)
        per_point = partial(_per_point_spec, block_size=block_size)
        size = math.prod(fine_shape)
        operands, in_specs = [*u.T], [per_point((num_points,))] * ndim
        if g_order is not None:
            operands.append(g.reshape(num_grids, size).astype(dtype))
            in_specs.append(pl.BlockSpec((num_grids, size), lambda p: (0, 0)))
        if uses_weights:
            operands.append(weights.astype(dtype))
            in_specs.append(per_point((num_grids, num_points)))
        if tu is not None:
            operands += _per_axis_and_direction(tu.astype(dtype))
            in_specs += [per_point((num_points,))] * (ndim * num_directions)
        if tg is not None:
            shape = (num_directions * num_grids, size)
            operands.append(tg.astype(dtype).reshape(shape))
            in_specs.append(pl.BlockSpec(shape, lambda p: (0, 0)))
        if tw is not None:
            operands.append(tw.astype(dtype).reshape(values_shape))
            in_specs.append(per_point(values_shape))
        kernel = _make_interp_tangent_kernel(
            fine_shape,
            kernel_key,
            num_grids,
            block_size,
            num_points,
            values,
            positions,
            nonzero,
            num_directions,
        )
        out = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(num_points, block_size),),
            in_specs=in_specs,
            out_specs=[per_point(shape) for shape in out_shapes],
            out_shape=[jax.ShapeDtypeStruct(shape, dtype) for shape in out_shapes],
            compiler_params=_CompilerParams(),
            name="nufft_interp_tangent",
        )(*operands)
    result = []
    if values:
        shape = (num_directions, num_grids, num_points)
        result.append(out[0].reshape(shape) if out_values else jnp.zeros(shape, dtype))
    if positions:
        u_bar = jnp.stack(out[len(out) - ndim * num_directions :])
        u_bar = u_bar.reshape(ndim, num_directions, num_points)
        result.append(jnp.moveaxis(u_bar, 0, -1))
    return tuple(result)


def _per_axis_and_direction(tu):
    """`(D, M, d)` as `d * D` arrays `(M,)`, axis-major. Per-direction operands are
    separate refs: interpret mode cannot index a ref by a Python int."""
    num_directions, num_points, ndim = tu.shape
    return list(jnp.moveaxis(tu, -1, 0).reshape(ndim * num_directions, num_points))


def _by_grid_axis(per_axis_and_direction, num_directions):
    """Axis-major `(x, y, z)` by direction, as nested lists per grid axis `(z, y, x)`
    and direction."""
    per_axis = [
        per_axis_and_direction[start : start + num_directions]
        for start in range(0, len(per_axis_and_direction), num_directions)
    ]
    return per_axis[::-1]


def _per_point_spec(shape, block_size):
    """Blocks of `block_size` points along the last axis, whole along the others."""
    return pl.BlockSpec(
        (*shape[:-1], block_size), lambda p: (0,) * (len(shape) - 1) + (p,)
    )


#
# Kernels
#
def _tap_polynomials(kernel_key, order):
    """Per tap, the Horner coefficients (highest power first) of the weight's
    `order`-th derivative in `u`, as Python floats."""
    width, degree, data = kernel_key
    coefficients = np.frombuffer(data, np.float64).reshape(width, degree + 1)
    return derivative_coefficients(coefficients, order).tolist()


def _horner(coefficients, z):
    if len(coefficients) == 1:
        return 0 * z + coefficients[0]
    out = coefficients[0] * z + coefficients[1]
    for coefficient in coefficients[2:]:
        out = out * z + coefficient
    return out


def _select(per_tap, t):
    """`per_tap[t]` for a traced loop index `t`."""
    out = per_tap[0]
    for k, value in enumerate(per_tap[1:], start=1):
        out = jnp.where(t == k, value, out)
    return out


def _axis_taps(u_refs, mask, width, polynomials):
    """Per grid axis (first to last): the first tap's index and every tap's
    polynomials at `z = 2s - 1`, `s` the point's offset, each `(block_size,)`. `u_refs`
    run (x, y, z)."""
    starts, per_axis = [], []
    for u_ref in reversed(u_refs):
        half = pltriton.load(u_ref, mask=mask) - width / 2
        start = jnp.ceil(half)
        z = 2 * (start - half) - 1
        starts.append(start.astype(jnp.int32))
        per_axis.append([[_horner(p, z) for p in polys] for polys in polynomials])
    return starts, per_axis


def _gather_jets(loads, orders, starts, per_axis, inner, fine_shape, zero):
    """Per source `k`, `sum_l x_k[l] prod_a phi_a(l)` and its `u`-derivatives up to
    `orders[k]`, keyed by sorted grid-axis tuples (`()` the value)."""
    ndim, width = len(fine_shape), len(inner)
    strides = [math.prod(fine_shape[axis + 1 :]) for axis in range(ndim)]

    def walk(axis, base):
        if axis == ndim - 1:
            jets = [{(axis,) * m: zero for m in range(order + 1)} for order in orders]
            for j in range(width):
                for jet, load, order in zip(jets, loads, orders):
                    x = load(base + inner[j])
                    for m in range(order + 1):
                        jet[(axis,) * m] = jet[(axis,) * m] + x * per_axis[axis][m][j]
            return jets

        def body(t, jets):
            index = (starts[axis] + t) % fine_shape[axis]
            subs = walk(axis + 1, base + index * strides[axis])
            factors = [_select(p, t) for p in per_axis[axis][: max(orders) + 1]]
            for jet, sub, order in zip(jets, subs, orders):
                for key, value in sub.items():
                    for m in range(order - len(key) + 1):
                        k = (axis,) * m + key
                        jet[k] = jet[k] + factors[m] * value
            return jets

        init = [
            {
                key: zero
                for n in range(order + 1)
                for key in itertools.combinations_with_replacement(range(axis, ndim), n)
            }
            for order in orders
        ]
        return jax.lax.fori_loop(0, width, body, init)

    return walk(0, 0)


@cache
def _make_spread_kernel(fine_shape, kernel_key, num_grids, block_size, num_points):
    ndim, width = len(fine_shape), kernel_key[0]
    strides = [math.prod(fine_shape[axis + 1 :]) for axis in range(ndim)]
    polynomials = [_tap_polynomials(kernel_key, 0)]

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
    polynomials = [_tap_polynomials(kernel_key, m) for m in range(1 + positions)]
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

        def group_body(group, carry):
            def grid_body(k, u_bar):
                r = group * grids_per_group + k
                (jet,) = _gather_jets(
                    [lambda flat: pltriton.load(g_ref.at[r, flat], mask=mask)],
                    [int(positions)],
                    starts,
                    per_axis,
                    inner,
                    fine_shape,
                    zero,
                )
                if values:
                    pltriton.store(values_ref.at[r], jet[()], mask=mask)
                if positions:
                    weight = pltriton.load(weights_ref.at[r], mask=mask)
                    u_bar = [a + weight * jet[(axis,)] for axis, a in enumerate(u_bar)]
                return u_bar

            u_bar = jax.lax.fori_loop(
                0, grids_per_group, grid_body, [zero] * ndim if positions else []
            )
            for u_bar_ref, gradient in zip(u_bar_refs, u_bar):
                pltriton.store(u_bar_ref.at[group], gradient, mask=mask)
            return carry

        jax.lax.fori_loop(0, num_groups, group_body, 0)

    return kernel


@cache
def _make_spread_tangent_kernel(
    fine_shape, kernel_key, num_grids, block_size, num_points, nonzero, num_directions
):
    """The tangent of `_make_spread_kernel` in `num_directions` directions, by the
    product rule along each axis: `(value, t) -> (value phi, t phi + value tu phi')`."""
    ndim, width = len(fine_shape), kernel_key[0]
    nonzero_u, nonzero_c = nonzero
    strides = [math.prod(fine_shape[axis + 1 :]) for axis in range(ndim)]
    polynomials = [_tap_polynomials(kernel_key, m) for m in range(1 + nonzero_u)]
    directions = range(num_directions)

    def kernel(*refs):
        u_refs, refs = refs[:ndim], list(refs[ndim:])
        c_ref = refs.pop(0) if nonzero_u else None
        tu_refs = [refs.pop(0) for _ in range(ndim * num_directions) if nonzero_u]
        tc_ref = refs.pop(0) if nonzero_c else None
        # The zeros input is aliased into `out_ref`
        _, out_ref = refs
        mask = (
            _valid_lane_mask(block_size, num_points) if num_points % block_size else None
        )
        starts, per_axis = _axis_taps(u_refs, mask, width, polynomials)
        tu = _by_grid_axis(
            [pltriton.load(ref, mask=mask) for ref in tu_refs], num_directions
        )
        inner = [(starts[-1] + j) % fine_shape[-1] for j in range(width)]

        def scatter(r, axis, base, value, tangents):
            if axis == ndim - 1:
                moved = [
                    _mul(value, tu[axis][d]) if nonzero_u else None for d in directions
                ]
                for j in range(width):
                    weight = per_axis[axis][0][j]
                    derivative = per_axis[axis][1][j] if nonzero_u else None
                    for d in directions:
                        tangent = _add(
                            _mul(tangents[d], weight), _mul(moved[d], derivative)
                        )
                        pltriton.atomic_add(
                            out_ref,
                            (d * num_grids + r, base + inner[j]),
                            tangent,
                            mask=mask,
                        )
                return

            def body(t, carry):
                index = (starts[axis] + t) % fine_shape[axis]
                weight = _select(per_axis[axis][0], t)
                moved = _mul(value, _select(per_axis[axis][1], t)) if nonzero_u else None
                scatter(
                    r,
                    axis + 1,
                    base + index * strides[axis],
                    _mul(value, weight),
                    [
                        _add(
                            _mul(tangents[d], weight),
                            _mul(moved, tu[axis][d]) if nonzero_u else None,
                        )
                        for d in directions
                    ],
                )
                return carry

            jax.lax.fori_loop(0, width, body, 0)

        def grid_body(r, carry):
            value = pltriton.load(c_ref.at[r], mask=mask) if nonzero_u else None
            tangents = [
                pltriton.load(tc_ref.at[d * num_grids + r], mask=mask)
                if nonzero_c
                else None
                for d in directions
            ]
            scatter(r, 0, 0, value, tangents)
            return carry

        jax.lax.fori_loop(0, num_grids, grid_body, 0)

    return kernel


def _interp_tangent_plan(values, positions, nonzero):
    """What the gather tangent kernel needs: whether it computes the values' tangent
    (else zero), `g`'s derivative order (`None`: `g` unused), and whether it reads
    `weights`."""
    nonzero_u, nonzero_g, nonzero_w = nonzero
    if positions and nonzero_u:
        g_order = 2
    elif (values and nonzero_u) or (positions and nonzero_w):
        g_order = 1
    else:
        g_order = None
    return (
        values and (nonzero_u or nonzero_g),
        g_order,
        positions and (nonzero_u or nonzero_g),
    )


@cache
def _make_interp_tangent_kernel(
    fine_shape,
    kernel_key,
    num_grids,
    block_size,
    num_points,
    values,
    positions,
    nonzero,
    num_directions,
):
    """The tangent of `_make_interp_kernel` (one group) in `num_directions` directions:
    values `interp(tg) + tu . grad`, positions `tweights grad + weights (grad(tg) + hess
    tu)`."""
    ndim, width = len(fine_shape), kernel_key[0]
    nonzero_u, nonzero_g, nonzero_w = nonzero
    out_values, g_order, uses_weights = _interp_tangent_plan(values, positions, nonzero)
    directions = range(num_directions)
    orders = [] if g_order is None else [g_order]
    orders += [int(positions)] * num_directions if nonzero_g else []
    polynomials = [_tap_polynomials(kernel_key, m) for m in range(max(orders) + 1)]

    def contract(tu, d, jet, key):
        """`sum_b tu_b jet[key + (b,)]` in direction `d`."""
        return _add(*(tu[b][d] * jet[tuple(sorted(key + (b,)))] for b in range(ndim)))

    def kernel(*refs):
        u_refs, refs = refs[:ndim], list(refs[ndim:])
        g_ref = refs.pop(0) if g_order is not None else None
        weights_ref = refs.pop(0) if uses_weights else None
        tu_refs = [refs.pop(0) for _ in range(ndim * num_directions) if nonzero_u]
        tg_ref = refs.pop(0) if nonzero_g else None
        tw_ref = refs.pop(0) if nonzero_w else None
        tvalues_ref = refs.pop(0) if out_values else None
        tu_bar_refs = _by_grid_axis(refs, num_directions)
        mask = (
            _valid_lane_mask(block_size, num_points) if num_points % block_size else None
        )
        starts, per_axis = _axis_taps(u_refs, mask, width, polynomials)
        tu = _by_grid_axis(
            [pltriton.load(ref, mask=mask) for ref in tu_refs], num_directions
        )
        inner = [(starts[-1] + j) % fine_shape[-1] for j in range(width)]
        zero = jnp.zeros((block_size,), u_refs[0].dtype)

        def grid_body(r, tu_bar):
            loads = []
            if g_order is not None:
                loads.append(lambda flat: pltriton.load(g_ref.at[r, flat], mask=mask))
            if nonzero_g:
                loads += [
                    lambda flat, row=d * num_grids + r: pltriton.load(
                        tg_ref.at[row, flat], mask=mask
                    )
                    for d in directions
                ]
            jets = _gather_jets(loads, orders, starts, per_axis, inner, fine_shape, zero)
            g_jet = jets.pop(0) if g_order is not None else None
            weight = pltriton.load(weights_ref.at[r], mask=mask) if uses_weights else None
            for d in directions:
                tg_jet = jets[d] if nonzero_g else {}
                row = d * num_grids + r
                if out_values:
                    tvalue = _add(
                        tg_jet.get(()), contract(tu, d, g_jet, ()) if nonzero_u else None
                    )
                    pltriton.store(tvalues_ref.at[row], tvalue, mask=mask)
                if not positions:
                    continue
                tw = pltriton.load(tw_ref.at[row], mask=mask) if nonzero_w else None
                for a in range(ndim):
                    tu_bar[d][a] = tu_bar[d][a] + _add(
                        _mul(tw, g_jet[(a,)]) if nonzero_w else None,
                        _mul(
                            weight,
                            _add(
                                tg_jet.get((a,)),
                                contract(tu, d, g_jet, (a,)) if nonzero_u else None,
                            ),
                        ),
                    )
            return tu_bar

        tu_bar = jax.lax.fori_loop(
            0,
            num_grids,
            grid_body,
            [[zero] * ndim for _ in directions] if positions else [],
        )
        for d, gradients in enumerate(tu_bar):
            for axis_refs, gradient in zip(tu_bar_refs, gradients):
                pltriton.store(axis_refs[d], gradient, mask=mask)

    return kernel
