"""The native NUFFT's Pallas kernels against its pure-JAX backend.

The kernel tests run anywhere in Pallas's interpret mode. The scatter's run with one
point per program: interpret mode applies an `atomic_add` with repeated indices as a
non-accumulating set and refuses a mask, so a program must hold one point and no
ragged lanes. The dispatch tests compile the kernels and need a GPU.
"""

import itertools
import math
from functools import partial

import cryojax.ndimage as im
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage import _pallas_helpers
from cryojax.ndimage._nufft import api as nufft_api, pallas_spread as nufft_pallas
from cryojax.ndimage._nufft.kernel import make_pswf_kernel
from cryojax.ndimage._nufft.spread import interp, interp_bwd, spread, spread_bwd
from jax._src import core as jax_core  # jaxpr inspection only


jax.config.update("jax_enable_x64", True)

requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="Pallas/Triton kernels require a CUDA GPU"
)

FINE_SHAPES = [(40, 36), (18, 16, 20)]
PIXEL_SIZE = 0.7


def _relative_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _pallas_calls(fn, *args):
    """Every `pallas_call` in `fn`'s traced program, as `(out_shapes, n_grid_axes)`."""
    return [
        ([v.aval.shape for v in eqn.outvars], len(eqn.params["grid_mapping"].grid))
        for eqn in _pallas_eqns(fn, *args)
    ]


def _pallas_names(fn, *args):
    return sorted(eqn.params["name"] for eqn in _pallas_eqns(fn, *args))


def _pallas_eqns(fn, *args):
    """Every `pallas_call` equation in `fn`'s traced program."""
    found = []

    def walk(v):
        if isinstance(v, jax_core.ClosedJaxpr):
            visit(v.jaxpr)
        elif isinstance(v, jax_core.Jaxpr):
            visit(v)
        elif isinstance(v, (tuple, list)):
            for w in v:
                walk(w)
        elif isinstance(v, dict):
            for w in v.values():
                walk(w)

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                found.append(eqn)
            for v in eqn.params.values():
                walk(v)

    visit(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


@pytest.fixture
def interpret(monkeypatch):
    real_pallas_call = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *a, **kw: real_pallas_call(*a, **{**kw, "interpret": True}),
    )


def _fine_problem(fine_shape, num_points, num_grids, seed, batch=()):
    """Fine-grid coordinates over twice the box, so that points wrap, with strengths and
    grids."""
    rng = np.random.default_rng(seed)
    m = np.array(fine_shape[::-1])
    u = rng.uniform(-m / 2, 1.5 * m, (*batch, num_points, m.size))
    c = rng.normal(size=(*batch, num_grids, num_points))
    g = rng.normal(size=(*batch, num_grids, *fine_shape))
    return jnp.asarray(u), jnp.asarray(c), jnp.asarray(g)


def _kernel(fine_shape):
    return make_pswf_kernel(1e-6, 2.0, len(fine_shape))


#
# Kernels, interpret mode
#
@pytest.mark.parametrize("fine_shape", FINE_SHAPES)
def test_spread_matches_pure_jax_interpret(interpret, monkeypatch, fine_shape):
    monkeypatch.setattr(_pallas_helpers, "CRYOJAX_PALLAS_BLOCK_SIZE", 1)
    kernel = _kernel(fine_shape)
    u, c, _ = _fine_problem(fine_shape, 40, 2, seed=0)
    actual = nufft_pallas.pallas_nufft_spread(
        u, c, fine_shape, nufft_pallas.make_kernel_key(kernel)
    )
    expected = spread(u, c, fine_shape, kernel)
    assert actual.shape == expected.shape
    assert _relative_error(actual, expected) <= 1e-12


@pytest.mark.parametrize(
    "values, positions", [(True, False), (True, True), (False, True)]
)
@pytest.mark.parametrize("fine_shape", FINE_SHAPES)
def test_interp_matches_pure_jax_interpret(interpret, fine_shape, values, positions):
    kernel = _kernel(fine_shape)
    # Not a multiple of the block size: the last program has masked lanes
    u, weights, g = _fine_problem(fine_shape, 150, 2, seed=1)
    actual_values, actual_u_bar = nufft_pallas.pallas_nufft_interp(
        u,
        g,
        weights if positions else None,
        fine_shape,
        nufft_pallas.make_kernel_key(kernel),
        values=values,
        positions=positions,
    )
    if values:
        assert _relative_error(actual_values, interp(u, g, kernel)) <= 1e-12
    else:
        assert actual_values is None
    if positions:
        expected_u_bar, _ = interp_bwd(u, g, weights, kernel)
        assert _relative_error(actual_u_bar, expected_u_bar) <= 1e-12
    else:
        assert actual_u_bar is None


def test_spread_vjp_gather_matches_pure_jax_interpret(interpret):
    fine_shape = FINE_SHAPES[0]
    kernel = _kernel(fine_shape)
    u, c, g_bar = _fine_problem(fine_shape, 150, 3, seed=2)
    c_bar, u_bar = nufft_pallas.pallas_nufft_interp(
        u, g_bar, c, fine_shape, nufft_pallas.make_kernel_key(kernel), True, True
    )
    expected_u_bar, expected_c_bar = spread_bwd(u, c, g_bar, kernel)
    assert _relative_error(c_bar, expected_c_bar) <= 1e-12
    assert _relative_error(u_bar, expected_u_bar) <= 1e-12


#
# Batching
#
@pytest.mark.parametrize("fine_shape", FINE_SHAPES)
def test_vmap_over_strengths_is_one_launch(fine_shape):
    key = nufft_pallas.make_kernel_key(_kernel(fine_shape))
    u, c, g = _fine_problem(fine_shape, 150, 2, seed=3)
    batched_c, batched_g = jnp.stack([c, 2 * c, -c]), jnp.stack([g, 2 * g, -g])

    def spread_fn(c):
        return nufft_pallas.pallas_nufft_spread(u, c, fine_shape, key)

    def interp_fn(g, weights):
        return nufft_pallas.pallas_nufft_interp(
            u, g, weights, fine_shape, key, True, True
        )

    size = int(np.prod(fine_shape))
    assert _pallas_calls(jax.vmap(spread_fn), batched_c) == [([(6, size)], 1)]
    # Values per grid, the position gradient per batch member and axis
    assert _pallas_calls(jax.vmap(interp_fn), batched_g, batched_c) == [
        ([(6, 150)] + [(3, 150)] * len(fine_shape), 1)
    ]


def test_vmap_over_grids_matches_unbatched_interpret(interpret, monkeypatch):
    """The batched kernel reduces the position gradient within each batch member."""
    fine_shape = FINE_SHAPES[0]
    kernel = _kernel(fine_shape)
    key = nufft_pallas.make_kernel_key(kernel)
    u, weights, g = _fine_problem(fine_shape, 150, 2, seed=4)
    _, batched_weights, batched_g = _fine_problem(fine_shape, 150, 2, seed=5, batch=(3,))

    def interp_fn(g, weights):
        return nufft_pallas.pallas_nufft_interp(
            u, g, weights, fine_shape, key, True, True
        )

    for in_axes, args in [
        ((0, 0), (batched_g, batched_weights)),
        ((None, 0), (g, batched_weights)),
        ((0, None), (batched_g, weights)),
    ]:
        values, u_bar = jax.vmap(interp_fn, in_axes=in_axes)(*args)
        for b in range(3):
            g_b, weights_b = (
                x if axis is None else x[b] for x, axis in zip(args, in_axes)
            )
            expected_values, expected_u_bar = spread_bwd(u, weights_b, g_b, kernel)[::-1]
            assert _relative_error(values[b], expected_values) <= 1e-12
            assert _relative_error(u_bar[b], expected_u_bar) <= 1e-12

    monkeypatch.setattr(_pallas_helpers, "CRYOJAX_PALLAS_BLOCK_SIZE", 1)
    u, batched_c, _ = _fine_problem(fine_shape, 30, 2, seed=6, batch=(3,))
    grids = jax.vmap(
        lambda c: nufft_pallas.pallas_nufft_spread(u[0], c, fine_shape, key)
    )(batched_c)
    for b in range(3):
        expected = spread(u[0], batched_c[b], fine_shape, kernel)
        assert _relative_error(grids[b], expected) <= 1e-12


def test_empty_points_under_vmap():
    """A vmap over strengths or grids with no points launches nothing and gives zeros."""
    fine_shape = FINE_SHAPES[0]
    key = nufft_pallas.make_kernel_key(_kernel(fine_shape))
    u, c, g = _fine_problem(fine_shape, 0, 2, seed=7, batch=(3,))

    def spread_fn(c):
        return nufft_pallas.pallas_nufft_spread(u[0], c, fine_shape, key)

    def interp_fn(g, weights):
        return nufft_pallas.pallas_nufft_interp(
            u[0], g, weights, fine_shape, key, True, True
        )

    assert _pallas_calls(jax.vmap(spread_fn), c) == []
    assert _pallas_calls(jax.vmap(interp_fn), g, c) == []
    grids = jax.vmap(spread_fn)(c)
    values, u_bar = jax.vmap(interp_fn)(g, c)
    assert grids.shape == (3, 2, *fine_shape) and not np.any(np.asarray(grids))
    assert values.shape == (3, 2, 0) and u_bar.shape == (3, 0, 2)


#
# Forward mode, interpret mode
#
def _directions(primals, n_directions, seed):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.normal(size=(n_directions, *p.shape))) for p in primals)


def _jvp_in(fn, primals, moving, directions):
    """`jax.vmap` over directions of `jax.jvp(fn)` in the `moving` primals; the others
    are closed over, so their tangents are symbolic zeros."""

    def fn_of_moving(*args):
        args = iter(args)
        return fn(*(next(args) if m else p for p, m in zip(primals, moving)))

    def pick(xs):
        return tuple(x for x, m in zip(xs, moving) if m)

    return jax.vmap(lambda *t: jax.jvp(fn_of_moving, pick(primals), t))(*pick(directions))


def _assert_trees_close(actual, expected, tolerance):
    actual, expected = jax.tree.leaves(actual), jax.tree.leaves(expected)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == e.dtype
        if np.any(np.asarray(e)):
            assert _relative_error(a, e) <= tolerance
        else:
            assert not np.any(np.asarray(a))


@pytest.mark.parametrize("moving", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fine_shape", FINE_SHAPES)
def test_spread_jvp_matches_pure_jax_interpret(
    interpret, monkeypatch, fine_shape, moving
):
    monkeypatch.setattr(_pallas_helpers, "CRYOJAX_PALLAS_BLOCK_SIZE", 1)
    kernel = _kernel(fine_shape)
    key = nufft_pallas.make_kernel_key(kernel)
    u, c, _ = _fine_problem(fine_shape, 30, 2, seed=12)
    directions = _directions((u, c), 3, seed=13)
    actual = _jvp_in(
        lambda u, c: nufft_pallas.pallas_nufft_spread(u, c, fine_shape, key),
        (u, c),
        moving,
        directions,
    )
    expected = _jvp_in(
        lambda u, c: spread(u, c, fine_shape, kernel), (u, c), moving, directions
    )
    _assert_trees_close(actual, expected, 1e-12)


INTERP_JVP_CASES = [
    (values, positions, moving)
    for values, positions in [(True, False), (True, True), (False, True)]
    for moving in itertools.product((True, False), repeat=2 + positions)
    if any(moving)
]


@pytest.mark.parametrize("values, positions, moving", INTERP_JVP_CASES)
@pytest.mark.parametrize("fine_shape", FINE_SHAPES)
def test_interp_jvp_matches_pure_jax_interpret(
    interpret, fine_shape, values, positions, moving
):
    kernel = _kernel(fine_shape)
    key = nufft_pallas.make_kernel_key(kernel)
    # Not a multiple of the block size: the last program has masked lanes
    u, weights, g = _fine_problem(fine_shape, 150, 2, seed=14)
    primals = (u, g, weights) if positions else (u, g)
    directions = _directions(primals, 3, seed=15)

    def reference(u, g, weights=None):
        if not positions:
            return interp(u, g, kernel), None
        u_bar, values_out = spread_bwd(u, weights, g, kernel)
        return (values_out if values else None), u_bar

    def pallas(u, g, weights=None):
        return nufft_pallas.pallas_nufft_interp(
            u, g, weights, fine_shape, key, values, positions
        )

    actual = _jvp_in(pallas, primals, moving, directions)
    expected = _jvp_in(reference, primals, moving, directions)
    _assert_trees_close(actual, expected, 1e-12)


def test_tangent_kernels_specialized_on_symbolic_zeros():
    """A symbolic-zero tangent never reaches a tangent kernel, nor does a primal operand
    that only it would multiply."""
    fine_shape, ndim = FINE_SHAPES[0], 2
    key = nufft_pallas.make_kernel_key(_kernel(fine_shape))
    u, c, g = _fine_problem(fine_shape, 150, 2, seed=16)

    def tangent_inputs(fn, primals, moving):
        directions = _directions(primals, 1, seed=17)
        (eqn,) = [
            eqn
            for eqn in _pallas_eqns(lambda: _jvp_in(fn, primals, moving, directions))
            if eqn.params["name"].endswith("_tangent")
        ]
        return len(eqn.invars)

    def spread_fn(u, c):
        return nufft_pallas.pallas_nufft_spread(u, c, fine_shape, key)

    def interp_fn(u, g, weights):
        return nufft_pallas.pallas_nufft_interp(
            u, g, weights, fine_shape, key, True, True
        )

    # Per axis u and tu; c, tc and the output buffer
    assert tangent_inputs(spread_fn, (u, c), (True, True)) == 2 * ndim + 3
    assert tangent_inputs(spread_fn, (u, c), (True, False)) == 2 * ndim + 2
    assert tangent_inputs(spread_fn, (u, c), (False, True)) == ndim + 2
    # Per axis u and tu; g, weights, tg, tw
    assert tangent_inputs(interp_fn, (u, g, c), (True, True, True)) == 2 * ndim + 4
    assert tangent_inputs(interp_fn, (u, g, c), (False, True, False)) == ndim + 2
    assert tangent_inputs(interp_fn, (u, g, c), (False, False, True)) == ndim + 2


def test_jvp_and_vmap_over_strengths_compose_interpret(interpret, monkeypatch):
    """With the positions shared, a `jax.vmap` over strengths and grids still folds the
    primal into one launch inside `jax.jvp`, and a `jax.jvp` inside `jax.vmap` matches
    pure JAX."""
    monkeypatch.setattr(_pallas_helpers, "CRYOJAX_PALLAS_BLOCK_SIZE", 1)
    fine_shape = FINE_SHAPES[0]
    kernel = _kernel(fine_shape)
    key = nufft_pallas.make_kernel_key(kernel)
    u, c, g = _fine_problem(fine_shape, 30, 2, seed=18, batch=(3,))
    u = u[0]
    (tu,) = _directions((u,), 1, seed=19)
    tu = tu[0]
    tc, tg = _directions((c, g), 1, seed=20)
    tc, tg = tc[0], tg[0]

    pallas_spread = partial(nufft_pallas.pallas_nufft_spread, fine_shape=fine_shape)

    def spread_fns(spread_fn):
        jvp_of_vmap = jax.jvp(jax.vmap(spread_fn, in_axes=(None, 0)), (u, c), (tu, tc))
        vmap_of_jvp = jax.vmap(
            lambda c, tc: jax.jvp(spread_fn, (u, c), (tu, tc)), in_axes=(0, 0)
        )(c, tc)
        return jvp_of_vmap, vmap_of_jvp

    actual = spread_fns(lambda u, c: pallas_spread(u, c, kernel_key=key))
    expected = spread_fns(lambda u, c: spread(u, c, fine_shape, kernel))
    _assert_trees_close(actual, expected, 1e-12)
    size = int(np.prod(fine_shape))
    primal_calls = [
        [v.aval.shape for v in eqn.outvars]
        for eqn in _pallas_eqns(
            lambda: jax.jvp(
                jax.vmap(
                    lambda u, c: pallas_spread(u, c, kernel_key=key), in_axes=(None, 0)
                ),
                (u, c),
                (tu, tc),
            )
        )
        if eqn.params["name"] == "nufft_spread"
    ]
    assert primal_calls == [[(6, size)]]

    def interp_fns(interp_fn):
        jvp_of_vmap = jax.jvp(
            jax.vmap(interp_fn, in_axes=(None, 0, 0)), (u, g, c), (tu, tg, tc)
        )
        vmap_of_jvp = jax.vmap(
            lambda g, c, tg, tc: jax.jvp(interp_fn, (u, g, c), (tu, tg, tc))
        )(g, c, tg, tc)
        return jvp_of_vmap, vmap_of_jvp

    def reference(u, g, weights):
        return spread_bwd(u, weights, g, kernel)[::-1]

    def pallas(u, g, weights):
        return nufft_pallas.pallas_nufft_interp(
            u, g, weights, fine_shape, key, True, True
        )

    _assert_trees_close(interp_fns(pallas), interp_fns(reference), 1e-12)


#
# Dispatch, compiled
#
# At eps 1e-9 and sigma 1.25 in 2D the deconvolution lifts summation-order rounding to
# 2e-12, in the pure-JAX backend against itself too, so float64 runs there at sigma 2
DISPATCH_CASES = [
    ((64, 64), np.float64, 1e-9, 2.0),
    ((64, 64), np.float64, 1e-6, 1.25),
    ((7, 6), np.float64, 1e-9, 2.0),
    ((16, 15, 14), np.float64, 1e-9, 1.25),
    ((64, 64), np.float32, 1e-4, 1.25),
    ((16, 15, 14), np.float32, 1e-4, 1.25),
]
ENABLE_PALLAS = {"pure": False, "full": True, "bwd": {"bwd": True}}


def _points(shape, num, seed, dtype, batch=()):
    """Positions over twice the box, so that points wrap, in units of `PIXEL_SIZE`."""
    n = np.array(shape[::-1])
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-n, n, (*batch, num, n.size)) * PIXEL_SIZE, dtype)


def _spectrum(shape, seed, dtype, batch=()):
    rng = np.random.default_rng(seed)
    half = (*batch, *shape[:-1], shape[-1] // 2 + 1)
    return jnp.asarray(rng.normal(size=half) + 1j * rng.normal(size=half)).astype(
        jnp.result_type(dtype, 1j)
    )


@partial(jax.jit, static_argnames=("shape", "eps", "sigma", "mode"))
def _nufft1_vjp(positions, strengths, pixel_size, cotangent, shape, eps, sigma, mode):
    def fn(positions, strengths, pixel_size):
        return im.dispatch_nufft1(
            positions,
            strengths,
            shape,
            pixel_size=pixel_size,
            eps=eps,
            upsample_factor=sigma,
            enable_pallas=ENABLE_PALLAS[mode],
        )

    spectrum, vjp = jax.vjp(fn, positions, strengths, pixel_size)
    return spectrum, vjp(cotangent)


@partial(jax.jit, static_argnames=("shape", "eps", "sigma", "mode"))
def _nufft2_vjp(positions, spectrum, pixel_size, cotangent, shape, eps, sigma, mode):
    def fn(positions, spectrum, pixel_size):
        return im.dispatch_nufft2(
            positions,
            spectrum,
            shape,
            pixel_size=pixel_size,
            eps=eps,
            upsample_factor=sigma,
            enable_pallas=ENABLE_PALLAS[mode],
        )

    values, vjp = jax.vjp(fn, positions, spectrum, pixel_size)
    return values, vjp(cotangent)


def _assert_matches(actual, expected, positions, dtype):
    """Primal and VJP within 1e-5 (float32) or 1e-12 (float64). The pixel-size
    cotangent sums cancelling per-point terms, so it is measured against their size."""
    tolerance = 1e-5 if dtype == np.float32 else 1e-12
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.shape == e.shape and a.dtype == e.dtype
    primal, (d_positions, d_data, d_pixel_size) = actual
    expected_primal, (expected_positions, expected_data, expected_pixel_size) = expected
    assert _relative_error(primal, expected_primal) <= tolerance
    assert _relative_error(d_positions, expected_positions) <= tolerance
    assert _relative_error(d_data, expected_data) <= tolerance
    terms = np.linalg.norm(np.asarray(positions * expected_positions) / PIXEL_SIZE)
    assert abs(d_pixel_size - expected_pixel_size) <= tolerance * terms


@requires_gpu
@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("mode", ["full", "bwd"])
@pytest.mark.parametrize("shape, dtype, eps, sigma", DISPATCH_CASES)
def test_pallas_nufft1_matches_pure_jax(shape, dtype, eps, sigma, mode, batch):
    positions = _points(shape, 300, seed=0, dtype=dtype)
    strengths = jnp.asarray(np.random.default_rng(1).normal(size=(*batch, 300)), dtype)
    pixel_size = jnp.asarray(PIXEL_SIZE, dtype)
    cotangent = _spectrum(shape, seed=2, dtype=dtype, batch=batch)
    args = (positions, strengths, pixel_size, cotangent)
    kwargs = dict(shape=shape, eps=eps, sigma=sigma)
    expected = _nufft1_vjp(*args, **kwargs, mode="pure")
    actual = _nufft1_vjp(*args, **kwargs, mode=mode)
    _assert_matches(actual, expected, positions, dtype)


@requires_gpu
@pytest.mark.parametrize("batch", [(), (3,)])
@pytest.mark.parametrize("mode", ["full", "bwd"])
@pytest.mark.parametrize("shape, dtype, eps, sigma", DISPATCH_CASES)
def test_pallas_nufft2_matches_pure_jax(shape, dtype, eps, sigma, mode, batch):
    positions = _points(shape, 300, seed=3, dtype=dtype)
    spectrum = _spectrum(shape, seed=4, dtype=dtype, batch=batch)
    pixel_size = jnp.asarray(PIXEL_SIZE, dtype)
    cotangent = jnp.asarray(np.random.default_rng(5).normal(size=(*batch, 300)), dtype)
    args = (positions, spectrum, pixel_size, cotangent)
    kwargs = dict(shape=shape, eps=eps, sigma=sigma)
    expected = _nufft2_vjp(*args, **kwargs, mode="pure")
    actual = _nufft2_vjp(*args, **kwargs, mode=mode)
    _assert_matches(actual, expected, positions, dtype)


@requires_gpu
@pytest.mark.parametrize("shape", [(64, 64), (16, 15, 14)])
def test_vmap_over_point_sets(shape):
    """Different point sets with batched strengths, values and VJPs."""
    positions = _points(shape, 200, seed=6, dtype=np.float64, batch=(4,))
    strengths = jnp.asarray(np.random.default_rng(7).normal(size=(4, 2, 200)))
    spectrum = _spectrum(shape, seed=8, dtype=np.float64, batch=(4, 2))
    pixel_size = jnp.asarray(PIXEL_SIZE)
    cotangent1 = _spectrum(shape, seed=9, dtype=np.float64, batch=(4, 2))
    cotangent2 = jnp.asarray(np.random.default_rng(10).normal(size=(4, 2, 200)))
    for vjp_fn, data, cotangent in [
        (_nufft1_vjp, strengths, cotangent1),
        (_nufft2_vjp, spectrum, cotangent2),
    ]:
        results = {
            mode: jax.vmap(
                partial(vjp_fn, shape=shape, eps=1e-9, sigma=2.0, mode=mode),
                in_axes=(0, 0, None, 0),
            )(positions, data, pixel_size, cotangent)
            for mode in ("pure", "full")
        }
        for a, e in zip(*(jax.tree.leaves(r) for r in results.values())):
            assert _relative_error(e, a) <= 1e-12


def _loss(positions, strengths, spectrum, pixel_size, shape, enable_pallas):
    kwargs = dict(
        pixel_size=pixel_size, eps=1e-9, upsample_factor=2.0, enable_pallas=enable_pallas
    )
    type1 = im.dispatch_nufft1(positions, strengths, shape, **kwargs)
    type2 = im.dispatch_nufft2(positions, spectrum, shape, **kwargs)
    # Linear in `type1`: a single point's `|type1|**2` does not depend on its position
    return jnp.vdot(spectrum, type1).real + jnp.sum(type2**2), (type1, type2)


@requires_gpu
@pytest.mark.parametrize("shape", [(8, 6), (5, 7, 6)])
def test_pallas_empty_points(shape):
    ndim = len(shape)
    spectrum = _spectrum(shape, seed=11, dtype=np.float64)
    value_and_grad = {
        enable_pallas: jax.value_and_grad(
            partial(_loss, shape=shape, enable_pallas=enable_pallas),
            argnums=(0, 1, 2, 3),
            has_aux=True,
        )
        for enable_pallas in (False, True)
    }

    empty = (jnp.zeros((0, ndim)), jnp.zeros(0), spectrum, jnp.asarray(PIXEL_SIZE))
    assert _pallas_calls(value_and_grad[True], *empty) == []
    (value, (type1, type2)), grads = value_and_grad[True](*empty)
    assert type1.shape == (*shape[:-1], shape[-1] // 2 + 1) and type2.shape == (0,)
    assert value == 0 and np.all(np.asarray(type1) == 0)
    assert [g.shape for g in grads] == [(0, ndim), (0,), spectrum.shape, ()]
    assert all(np.all(np.asarray(g) == 0) for g in grads)

    def hvp(*args):
        grad = lambda *p: value_and_grad[True](*p)[1]
        return jax.jvp(grad, args, args)

    assert _pallas_calls(hvp, *empty) == []
    for g in jax.tree.leaves(hvp(*empty)):
        assert not np.any(np.asarray(g))

    one = (jnp.asarray([[0.3, -1.2, 0.7][:ndim]]), jnp.asarray([2.5]), *empty[2:])
    (actual, _), actual_grads = value_and_grad[True](*one)
    (expected, _), expected_grads = value_and_grad[False](*one)
    assert _relative_error(actual, expected) <= 1e-12
    for a, e in zip(actual_grads, expected_grads):
        assert _relative_error(a, e) <= 1e-12

    # Under a vmap over strengths and spectra, the positions shared
    batched = (empty[0], jnp.zeros((3, 0)), jnp.stack([spectrum] * 3), empty[3])
    vmapped = {
        enable_pallas: jax.vmap(fn, in_axes=(None, 0, 0, None))
        for enable_pallas, fn in value_and_grad.items()
    }
    assert _pallas_calls(vmapped[True], *batched) == []
    actual, expected = (jax.tree.leaves(vmapped[k](*batched)) for k in (True, False))
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and np.array_equal(a, e)


@requires_gpu
@pytest.mark.parametrize(
    "enable_pallas, pallas",
    [
        (False, False),
        (True, True),
        ({"fwd": True, "bwd": False}, True),
        ({"fwd": False, "bwd": True}, True),
    ],
)
def test_auto_prices_pallas_when_either_pass_uses_it(monkeypatch, enable_pallas, pallas):
    backends = []

    def resolve_upsample_factor(upsample_factor, **kwargs):
        backends.append(kwargs["pallas"])
        return resolve_upsample_factor_unpatched(upsample_factor, **kwargs)

    resolve_upsample_factor_unpatched = nufft_api.resolve_upsample_factor
    monkeypatch.setattr(nufft_api, "resolve_upsample_factor", resolve_upsample_factor)
    shape = (16, 15)
    positions = _points(shape, 20, seed=43, dtype=np.float32)
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=1e-4, enable_pallas=enable_pallas)
    im.dispatch_nufft1(positions, jnp.ones(20, np.float32), shape, **kwargs)
    spectrum = _spectrum(shape, seed=44, dtype=np.float32)
    im.dispatch_nufft2(positions, spectrum, shape, **kwargs)
    assert backends == [pallas, pallas]


#
# Forward-over-reverse, compiled
#
HVP_CASES = [
    ((64, 64), np.float64, 1e-9, 2.0),
    ((16, 15, 14), np.float64, 1e-9, 1.25),
    ((64, 64), np.float32, 1e-4, 1.25),
    ((16, 15, 14), np.float32, 1e-4, 1.25),
]
# One HVP: per `custom_vjp`, each primal kernel and its tangent kernel
HVP_KERNELS = {
    1: ["nufft_interp", "nufft_interp_tangent", "nufft_spread", "nufft_spread_tangent"],
    2: ["nufft_interp"] * 2
    + ["nufft_interp_tangent"] * 2
    + ["nufft_spread", "nufft_spread_tangent"],
}


def _hvp_problem(transform, shape, dtype, seed, batch=()):
    """Positions, strengths (type 1) or spectrum (type 2), and pixel size; and the
    target of `_hvp_loss`."""
    positions = _points(shape, 300, seed=seed, dtype=dtype)
    rng = np.random.default_rng(seed + 1)
    strengths = jnp.asarray(rng.normal(size=(*batch, 300)), dtype)
    spectrum = _spectrum(shape, seed=seed + 2, dtype=dtype, batch=batch)
    data, target = (strengths, spectrum) if transform == 1 else (spectrum, strengths)
    return (positions, data, jnp.asarray(PIXEL_SIZE, dtype)), target


def _hvp_directions(primals, seed, n_directions=None):
    rng = np.random.default_rng(seed)
    lead = () if n_directions is None else (n_directions,)

    def draw(p):
        t = rng.normal(size=(*lead, *p.shape))
        if jnp.iscomplexobj(p):
            t = t + 1j * rng.normal(size=t.shape)
        return jnp.asarray(t).astype(p.dtype)

    return tuple(draw(p) for p in primals)


def _hvp_loss(transform, shape, eps, sigma, target, enable_pallas):
    dispatch = im.dispatch_nufft1 if transform == 1 else im.dispatch_nufft2

    def loss(positions, data, pixel_size):
        out = dispatch(
            positions,
            data,
            shape,
            pixel_size=pixel_size,
            eps=eps,
            upsample_factor=sigma,
            enable_pallas=enable_pallas,
        )
        return jnp.vdot(out - target, out - target).real

    return loss


@partial(jax.jit, static_argnames=("transform", "shape", "eps", "sigma", "enable_pallas"))
def _hvp(primals, directions, target, transform, shape, eps, sigma, enable_pallas):
    loss = _hvp_loss(transform, shape, eps, sigma, target, enable_pallas)
    return jax.jvp(jax.grad(loss, argnums=(0, 1, 2)), primals, directions)


@requires_gpu
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("transform", [1, 2])
@pytest.mark.parametrize("shape, dtype, eps, sigma", HVP_CASES)
def test_pallas_hvp_matches_pure_jax(shape, dtype, eps, sigma, transform, batch):
    primals, target = _hvp_problem(transform, shape, dtype, seed=30, batch=batch)
    directions = _hvp_directions(primals, seed=31)
    kwargs = dict(transform=transform, shape=shape, eps=eps, sigma=sigma)
    expected = _hvp(primals, directions, target, **kwargs, enable_pallas=False)
    actual = _hvp(primals, directions, target, **kwargs, enable_pallas=True)
    tolerance = 1e-4 if dtype == np.float32 else 1e-10
    (grads, hvps), (expected_grads, expected_hvps) = actual, expected
    _assert_trees_close(
        (grads[:2], hvps[:2]), (expected_grads[:2], expected_hvps[:2]), tolerance
    )
    # The pixel-size entries sum cancelling per-point terms (`u` is linear in
    # `positions / pixel_size`), so they are measured against those terms' size
    (positions, _, pixel_size), (t_positions, _, t_pixel_size) = primals, directions
    norm = lambda x: np.linalg.norm(np.asarray(x))
    grad_terms = norm(positions * expected_grads[0]) / pixel_size
    hvp_terms = (
        norm(t_positions * expected_grads[0]) + norm(positions * expected_hvps[0])
    ) / pixel_size + abs(t_pixel_size) * grad_terms / pixel_size
    assert abs(grads[2] - expected_grads[2]) <= tolerance * grad_terms
    assert abs(hvps[2] - expected_hvps[2]) <= tolerance * hvp_terms


@requires_gpu
@pytest.mark.parametrize("transform", [1, 2])
@pytest.mark.parametrize("argnum", [0, 1, 2])
def test_pallas_hvp_in_one_argument(transform, argnum):
    """The other arguments are constants: their tangents reach the rules as symbolic
    zeros."""
    shape, dtype, eps, sigma = HVP_CASES[0]
    primals, target = _hvp_problem(transform, shape, dtype, seed=32)
    (direction,) = _hvp_directions(primals[argnum : argnum + 1], seed=33)

    def hvp(enable_pallas):
        loss = _hvp_loss(transform, shape, eps, sigma, target, enable_pallas)

        def loss_in_one(x):
            return loss(*primals[:argnum], x, *primals[argnum + 1 :])

        return jax.jit(jax.jvp, static_argnums=0)(
            jax.grad(loss_in_one), (primals[argnum],), (direction,)
        )

    _assert_trees_close(hvp(True), hvp(False), 1e-10)


@requires_gpu
@pytest.mark.parametrize("transform", [1, 2])
def test_pallas_linearize_value_and_grad(transform):
    """`jax.linearize` keeps the primal: its tangent map runs the tangent kernels only."""
    shape, dtype, eps, sigma = HVP_CASES[0]
    primals, target = _hvp_problem(transform, shape, dtype, seed=34)
    directions = _hvp_directions(primals, seed=35)
    results = {}
    for enable_pallas in (False, True):
        loss = _hvp_loss(transform, shape, eps, sigma, target, enable_pallas)
        value_and_grad = jax.value_and_grad(loss, argnums=(0, 1, 2))
        out, tangent_map = jax.linearize(value_and_grad, *primals)
        results[enable_pallas] = (out, tangent_map(*directions))
    names = _pallas_names(tangent_map, *directions)
    assert names == [n for n in HVP_KERNELS[transform] if n.endswith("_tangent")]
    _assert_trees_close(results[True], results[False], 1e-10)


@requires_gpu
@pytest.mark.parametrize("transform", [1, 2])
def test_pallas_vmap_hvp_one_launch(transform):
    """Five HVPs at one point: each tangent kernel launches once with all five
    directions, on one grid axis."""
    shape, dtype, eps, sigma = HVP_CASES[0]
    primals, target = _hvp_problem(transform, shape, dtype, seed=36)
    directions = _hvp_directions(primals, seed=37, n_directions=5)

    def hvp(enable_pallas):
        loss = _hvp_loss(transform, shape, eps, sigma, target, enable_pallas)
        grad = jax.grad(loss, argnums=(0, 1, 2))
        return lambda v: jax.jvp(grad, primals, v)[1]

    def tangent_sizes(fn, *args):
        return sorted(
            (eqn.params["name"], sum(math.prod(v.aval.shape) for v in eqn.outvars))
            for eqn in _pallas_eqns(fn, *args)
            if eqn.params["name"].endswith("_tangent")
        )

    eqns = _pallas_eqns(jax.vmap(hvp(True)), directions)
    assert sorted(eqn.params["name"] for eqn in eqns) == HVP_KERNELS[transform]
    assert all(len(eqn.params["grid_mapping"].grid) == 1 for eqn in eqns)
    one = jax.tree.map(lambda t: t[0], directions)
    assert tangent_sizes(jax.vmap(hvp(True)), directions) == [
        (name, 5 * size) for name, size in tangent_sizes(hvp(True), one)
    ]
    actual = jax.jit(jax.vmap(hvp(True)))(directions)
    expected = jax.jit(jax.vmap(hvp(False)))(directions)
    _assert_trees_close(actual, expected, 1e-10)


@requires_gpu
@pytest.mark.parametrize("transform", [1, 2])
@pytest.mark.parametrize("n_directions, n_chunks", [(None, 1), (12, 1), (13, 2)])
def test_pallas_hvp_launch_count(transform, n_directions, n_chunks):
    """One HVP launches each primal kernel and its tangent kernel once; a batch of
    directions beyond `_MAX_JVP_DIRECTIONS` splits into chunks."""
    assert _pallas_helpers._MAX_JVP_DIRECTIONS == 12
    shape, dtype, eps, sigma = HVP_CASES[0]
    primals, target = _hvp_problem(transform, shape, dtype, seed=38)
    directions = _hvp_directions(primals, seed=39, n_directions=n_directions)
    loss = _hvp_loss(transform, shape, eps, sigma, target, True)
    grad = jax.grad(loss, argnums=(0, 1, 2))
    hvp = lambda v: jax.jvp(grad, primals, v)[1]
    names = _pallas_names(hvp if n_directions is None else jax.vmap(hvp), directions)
    expected = [
        name
        for name in HVP_KERNELS[transform]
        for _ in range(n_chunks if name.endswith("_tangent") else 1)
    ]
    assert names == sorted(expected)


@requires_gpu
@pytest.mark.parametrize("transform", [1, 2])
def test_pallas_hvp_under_vmap(transform):
    """HVPs under a `jax.vmap` over point sets, and over data with the positions
    shared."""
    shape, dtype, eps, sigma = HVP_CASES[0]
    primals, target = _hvp_problem(transform, shape, dtype, seed=40, batch=(3,))
    positions = _points(shape, 300, seed=41, dtype=dtype, batch=(3,))
    directions = _hvp_directions((positions, *primals[1:]), seed=42)

    def hvps(enable_pallas):
        loss = _hvp_loss(transform, shape, eps, sigma, target[0], enable_pallas)
        grad = jax.grad(loss, argnums=(0, 1, 2))
        hvp = lambda p, v: jax.jvp(grad, p, v)
        point_sets = jax.vmap(hvp, in_axes=((0, 0, None), (0, 0, None)))(
            (positions, *primals[1:]), directions
        )
        shared = jax.vmap(hvp, in_axes=((None, 0, None), (None, 0, None)))(
            primals, (directions[0][0], *directions[1:])
        )
        return point_sets, shared

    _assert_trees_close(jax.jit(hvps, static_argnums=0)(True), hvps(False), 1e-10)
