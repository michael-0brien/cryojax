"""The native NUFFT's Pallas kernels against its pure-JAX backend.

The kernel tests run anywhere in Pallas's interpret mode. The scatter's run with one
point per program: interpret mode applies an `atomic_add` with repeated indices as a
non-accumulating set and refuses a mask, so a program must hold one point and no
ragged lanes. The dispatch tests compile the kernels and need a GPU.
"""

from functools import partial

import cryojax.ndimage as im
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage import _pallas_helpers
from cryojax.ndimage._nufft import pallas_spread as nufft_pallas
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
                grid = eqn.params["grid_mapping"].grid
                found.append(([v.aval.shape for v in eqn.outvars], len(grid)))
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

    one = (jnp.asarray([[0.3, -1.2, 0.7][:ndim]]), jnp.asarray([2.5]), *empty[2:])
    (actual, _), actual_grads = value_and_grad[True](*one)
    (expected, _), expected_grads = value_and_grad[False](*one)
    assert _relative_error(actual, expected) <= 1e-12
    for a, e in zip(actual_grads, expected_grads):
        assert _relative_error(a, e) <= 1e-12
