"""Tests for the Pallas/Triton GPU backend of `cryojax.ndimage.spread_gaussians_2d/3d`.

Numerical correctness/gradient checks for the *forward* (atomic-add scatter)
kernel only run on a real CUDA GPU: `interpret=True` does not faithfully
simulate `atomic_add` for duplicate indices within a single vectorized call
(silently drops the duplicate rather than accumulating -- see the
`pallas-triton-gotchas` project note), and realistic inputs (points with
overlapping kernel footprints) hit this constantly. The *backward*
(gather/interpolation) kernel has no such restriction -- gathers interpret
faithfully -- so it additionally gets `interpret=True` coverage that runs
anywhere (no GPU required).

Gradient checks compare directly against the pure-JAX reference (itself
already validated by `test_ndimage.py`'s own `check_grads` checks) rather
than using `check_grads` here too: `check_grads`' finite differences are a
weaker, higher-tolerance proxy for exactly this comparison, and add real
wall-clock cost for no stronger a guarantee. Direct comparison is both
tighter (exact-vs-exact) and cheaper.

The `jax.jit`-wrapped functions below are deliberately defined once at module
level, not inside each test: defining `@jax.jit` fresh inside a test function
creates a new function object (and forces a fresh trace/compile) on every
parametrized call, even when the underlying kernel is identical -- the same
"fresh closure defeats caching" trap as the (separately fixed) Pallas kernel
factories in `pallas_spread.py`. Module-level jitted functions let JAX's own
compilation cache actually do its job across the `scalar_variance` /
`enable_pallas` parametrizations that share a compiled kernel.
"""

import jax
import jax.numpy as jnp
import pytest
from cryojax.ndimage import spread_gaussians_2d, spread_gaussians_3d
from cryojax.ndimage._spreading.pallas_spread import (
    pallas_spread_vjp_2d,
    resolve_enable_pallas,
)


requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="Pallas/Triton kernels require a CUDA GPU",
)


@pytest.fixture
def points_2d():
    # Deliberately larger than the kernel's internal block size, and
    # crossing a block boundary with a ragged trailing block, to exercise
    # the padding-lane masking as well as heavy overlap between points'
    # kernel footprints (the case `interpret=True` can't simulate for the
    # scatter kernel; see module docstring).
    key = jax.random.PRNGKey(0)
    m = 150
    ny, nx = 48, 40
    pixel_size = jnp.asarray(1.3)
    x = jax.random.uniform(key, (m,), minval=-15, maxval=15) * pixel_size
    y = (
        jax.random.uniform(jax.random.fold_in(key, 1), (m,), minval=-15, maxval=15)
        * pixel_size
    )
    amplitude = jax.random.normal(jax.random.fold_in(key, 2), (m,)) * 2 + 3
    variance = jnp.abs(jax.random.normal(jax.random.fold_in(key, 3), (m,))) * 0.3 + 0.4
    return x, y, amplitude, variance, pixel_size, (ny, nx)


@pytest.fixture
def points_3d():
    key = jax.random.PRNGKey(1)
    m = 150
    nz, ny, nx = 24, 28, 32
    voxel_size = jnp.asarray(1.1)
    x = jax.random.uniform(key, (m,), minval=-12, maxval=12) * voxel_size
    y = (
        jax.random.uniform(jax.random.fold_in(key, 1), (m,), minval=-12, maxval=12)
        * voxel_size
    )
    z = (
        jax.random.uniform(jax.random.fold_in(key, 2), (m,), minval=-12, maxval=12)
        * voxel_size
    )
    amplitude = jax.random.normal(jax.random.fold_in(key, 3), (m,)) * 2 + 3
    variance = jnp.abs(jax.random.normal(jax.random.fold_in(key, 4), (m,))) * 0.3 + 0.4
    return x, y, z, amplitude, variance, voxel_size, (nz, ny, nx)


# ── Module-level jitted wrappers (see module docstring for why) ─────────────


def _make_jit_spread_2d(enable_pallas):
    def raw(x, y, amplitude, variance, pixel_size, shape, n_spread, use_erf):
        return spread_gaussians_2d(
            x,
            y,
            amplitude,
            variance,
            shape,
            pixel_size=pixel_size,
            n_spread=n_spread,
            use_erf=use_erf,
            enable_pallas=enable_pallas,
        )

    return jax.jit(raw, static_argnames=("shape", "n_spread", "use_erf"))


def _make_jit_spread_3d(enable_pallas):
    def raw(x, y, z, amplitude, variance, voxel_size, shape, n_spread, use_erf):
        return spread_gaussians_3d(
            x,
            y,
            z,
            amplitude,
            variance,
            shape,
            voxel_size=voxel_size,
            n_spread=n_spread,
            use_erf=use_erf,
            enable_pallas=enable_pallas,
        )

    return jax.jit(raw, static_argnames=("shape", "n_spread", "use_erf"))


_JIT_SPREAD_2D = {
    "pure_jax": _make_jit_spread_2d(False),
    "full": _make_jit_spread_2d(True),
    "fwd_only": _make_jit_spread_2d({"fwd": True}),
    "bwd_only": _make_jit_spread_2d({"bwd": True}),
}
_JIT_SPREAD_3D = {
    "pure_jax": _make_jit_spread_3d(False),
    "full": _make_jit_spread_3d(True),
    "fwd_only": _make_jit_spread_3d({"fwd": True}),
    "bwd_only": _make_jit_spread_3d({"bwd": True}),
}


# ── `resolve_enable_pallas`: pure config-resolution logic, no GPU needed ────


@pytest.fixture
def force_gpu_backend(monkeypatch):
    # `resolve_enable_pallas` both resolves the fwd/bwd flags *and* checks
    # GPU availability (fail-fast, rather than deferring to whatever error
    # `pallas_call` itself raises). These tests exercise the resolution
    # logic in isolation, so they mock `jax.default_backend()` -> "gpu"
    # (this repo's own `.venv` is CPU-only) rather than actually requiring
    # one; GPU-availability *rejection* is covered separately below.
    import cryojax.ndimage._spreading.pallas_spread as pallas_spread

    monkeypatch.setattr(pallas_spread.jax, "default_backend", lambda: "gpu")


def test_resolve_enable_pallas_none_defers_to_env(monkeypatch, force_gpu_backend):
    import cryojax.ndimage._spreading.pallas_spread as pallas_spread

    monkeypatch.setattr(pallas_spread, "CRYOJAX_ENABLE_PALLAS", True)
    assert resolve_enable_pallas(None) == (True, True)
    monkeypatch.setattr(pallas_spread, "CRYOJAX_ENABLE_PALLAS", False)
    assert resolve_enable_pallas(None) == (False, False)


def test_resolve_enable_pallas_bool(force_gpu_backend):
    assert resolve_enable_pallas(True) == (True, True)
    assert resolve_enable_pallas(False) == (False, False)


def test_resolve_enable_pallas_bool_overrides_env(monkeypatch, force_gpu_backend):
    # `enable_pallas=<bool>` must win outright over `CRYOJAX_ENABLE_PALLAS`,
    # in both directions -- not just take effect when the env var agrees.
    import cryojax.ndimage._spreading.pallas_spread as pallas_spread

    monkeypatch.setattr(pallas_spread, "CRYOJAX_ENABLE_PALLAS", True)
    assert resolve_enable_pallas(False) == (False, False)
    monkeypatch.setattr(pallas_spread, "CRYOJAX_ENABLE_PALLAS", False)
    assert resolve_enable_pallas(True) == (True, True)


def test_resolve_enable_pallas_dict_partial(monkeypatch, force_gpu_backend):
    import cryojax.ndimage._spreading.pallas_spread as pallas_spread

    monkeypatch.setattr(pallas_spread, "CRYOJAX_ENABLE_PALLAS", False)
    assert resolve_enable_pallas({"fwd": True}) == (True, False)
    assert resolve_enable_pallas({"bwd": True}) == (False, True)
    assert resolve_enable_pallas({"fwd": True, "bwd": True}) == (True, True)
    # A dict overrides per-key; a key it doesn't mention still falls back
    # to the env var, not to some other default.
    monkeypatch.setattr(pallas_spread, "CRYOJAX_ENABLE_PALLAS", True)
    assert resolve_enable_pallas({"fwd": False}) == (False, True)


def test_resolve_enable_pallas_requires_gpu_when_no_gpu_backend(monkeypatch):
    import cryojax.ndimage._spreading.pallas_spread as pallas_spread

    monkeypatch.setattr(pallas_spread.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        resolve_enable_pallas(True)


def test_resolve_enable_pallas_invalid_type():
    with pytest.raises(TypeError):
        resolve_enable_pallas("not-a-valid-value")  # type: ignore[arg-type]


# ── Backward (gather): safe to check numerically under `interpret=True` ─────
#
# `pallas_spread_vjp_{2,3}d` don't expose an `interpret` argument (the real
# module always compiles for real), so these tests call `pl.pallas_call`
# with `interpret=True` directly via a small local monkeypatch of
# `pl.pallas_call`'s default, rather than threading an `interpret` kwarg
# through the whole public dispatch chain just for testing.


@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_bwd_2d_matches_pure_jax_interpret(
    monkeypatch, points_2d, use_erf, scalar_variance
):
    import jax.experimental.pallas as pl

    real_pallas_call = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *a, **kw: real_pallas_call(*a, **{**kw, "interpret": True}),
    )

    x, y, amplitude, variance, pixel_size, shape = points_2d
    if scalar_variance:
        variance = variance[0]
    n_spread = 7
    ny, nx = shape

    def f(x, y, amplitude, variance, pixel_size):
        return spread_gaussians_2d(
            x,
            y,
            amplitude,
            variance,
            shape,
            pixel_size=pixel_size,
            n_spread=n_spread,
            use_erf=use_erf,
        )

    out_ref, vjp_fn = jax.vjp(f, x, y, amplitude, variance, pixel_size)
    g = jax.random.normal(jax.random.PRNGKey(42), out_ref.shape)
    grads_ref = vjp_fn(g)

    i = x / pixel_size + nx // 2
    j = y / pixel_size + ny // 2
    res = (i, j, amplitude, variance, pixel_size)
    di, dj, damplitude, dvariance, dpixel_size = pallas_spread_vjp_2d(
        ny, nx, n_spread, use_erf, res, g
    )
    # `grads_ref` is w.r.t. physical `x`/`y`; `di`/`dj` are w.r.t. grid-index
    # `i`/`j` (`di/dx = 1/pixel_size`), so compare after the same rescaling
    # `spread_gaussians_2d` itself applies internally.
    dx_ref, dy_ref, damp_ref, dvar_ref, dpix_ref = grads_ref
    assert jnp.allclose(di / pixel_size, dx_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(dj / pixel_size, dy_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(damplitude, damp_ref, atol=1e-4, rtol=1e-4)


# ── Forward-only value match: real GPU only ──────────────────────────────────


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_fwd_2d_matches_pure_jax(points_2d, use_erf, scalar_variance):
    x, y, amplitude, variance, pixel_size, shape = points_2d
    if scalar_variance:
        variance = variance[0]
    n_spread = 7

    ref = _JIT_SPREAD_2D["pure_jax"](
        x, y, amplitude, variance, pixel_size, shape, n_spread, use_erf
    )
    out = _JIT_SPREAD_2D["fwd_only"](
        x, y, amplitude, variance, pixel_size, shape, n_spread, use_erf
    )
    assert jnp.allclose(out, ref, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_fwd_3d_matches_pure_jax(points_3d, use_erf, scalar_variance):
    x, y, z, amplitude, variance, voxel_size, shape = points_3d
    if scalar_variance:
        variance = variance[0]
    n_spread = 7

    ref = _JIT_SPREAD_3D["pure_jax"](
        x, y, z, amplitude, variance, voxel_size, shape, n_spread, use_erf
    )
    out = _JIT_SPREAD_3D["fwd_only"](
        x, y, z, amplitude, variance, voxel_size, shape, n_spread, use_erf
    )
    assert jnp.allclose(out, ref, atol=1e-4, rtol=1e-4)


# ── Full gradient match vs. pure-JAX reference: real GPU only ────────────────


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
@pytest.mark.parametrize("enable_pallas_key", ["full", "fwd_only", "bwd_only"])
def test_pallas_2d_grads_match_pure_jax(
    points_2d, use_erf, scalar_variance, enable_pallas_key
):
    # Regression coverage for two real bugs caught here (both now fixed,
    # see `pallas_spread.py` and the `pallas-triton-gotchas` project note):
    # (1) with `M` not a multiple of the kernel's internal block size
    # (`points_2d`'s M=150 is deliberately ragged), the backward kernel
    # produced NaN in specific entries of an *earlier, fully-valid* block --
    # not even the ragged block itself -- traced to unmasked output stores
    # letting the ragged block's internal NaN (from zero-padded `variance`)
    # leak across blocks; (2) the `use_erf=True` branch's
    # `2.0 / jnp.sqrt(jnp.pi)` constant (no traced array anywhere in that
    # expression) silently resolved to float32 in Pallas-Triton regardless
    # of the surrounding computation's actual dtype, crashing under
    # `jax_enable_x64` (this repo's test `conftest.py` enables it globally).
    x, y, amplitude, variance, pixel_size, shape = points_2d
    if scalar_variance:
        variance = variance[0]
    n_spread = 7

    def call(fn, *args):
        return fn(*args, shape, n_spread, use_erf)

    out_ref, vjp_ref = jax.vjp(
        lambda x, y, amp, var, pix: call(_JIT_SPREAD_2D["pure_jax"], x, y, amp, var, pix),
        x,
        y,
        amplitude,
        variance,
        pixel_size,
    )
    out_pallas, vjp_pallas = jax.vjp(
        lambda x, y, amp, var, pix: call(
            _JIT_SPREAD_2D[enable_pallas_key], x, y, amp, var, pix
        ),
        x,
        y,
        amplitude,
        variance,
        pixel_size,
    )
    assert jnp.allclose(out_pallas, out_ref, atol=1e-4, rtol=1e-4)
    g = jax.random.normal(jax.random.PRNGKey(42), out_ref.shape)
    for ref, pallas in zip(vjp_ref(g), vjp_pallas(g)):
        assert not jnp.any(jnp.isnan(pallas))
        assert jnp.allclose(pallas, ref, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
@pytest.mark.parametrize("enable_pallas_key", ["full", "fwd_only", "bwd_only"])
def test_pallas_3d_grads_match_pure_jax(
    points_3d, use_erf, scalar_variance, enable_pallas_key
):
    # See `test_pallas_2d_grads_match_pure_jax` for context.
    x, y, z, amplitude, variance, voxel_size, shape = points_3d
    if scalar_variance:
        variance = variance[0]
    n_spread = 7

    def call(fn, *args):
        return fn(*args, shape, n_spread, use_erf)

    out_ref, vjp_ref = jax.vjp(
        lambda x, y, z, amp, var, vox: call(
            _JIT_SPREAD_3D["pure_jax"], x, y, z, amp, var, vox
        ),
        x,
        y,
        z,
        amplitude,
        variance,
        voxel_size,
    )
    out_pallas, vjp_pallas = jax.vjp(
        lambda x, y, z, amp, var, vox: call(
            _JIT_SPREAD_3D[enable_pallas_key], x, y, z, amp, var, vox
        ),
        x,
        y,
        z,
        amplitude,
        variance,
        voxel_size,
    )
    assert jnp.allclose(out_pallas, out_ref, atol=1e-4, rtol=1e-4)
    g = jax.random.normal(jax.random.PRNGKey(42), out_ref.shape)
    for ref, pallas in zip(vjp_ref(g), vjp_pallas(g)):
        assert not jnp.any(jnp.isnan(pallas))
        assert jnp.allclose(pallas, ref, atol=1e-4, rtol=1e-4)


@requires_gpu
def test_pallas_requires_gpu_error_message_not_triggered_on_gpu(points_2d):
    # Sanity check that `enable_pallas=True` doesn't spuriously raise the
    # "requires a CUDA GPU" error when a GPU actually is available.
    x, y, amplitude, variance, pixel_size, shape = points_2d
    spread_gaussians_2d(
        x, y, amplitude, variance, shape, pixel_size=pixel_size, enable_pallas=True
    )


# ── Forward-mode: explicit JVP rules on the pallas wrappers ─────────────────
#
# JAX cannot differentiate the kernels themselves: the scatter aliases a zeros
# input into its output (refused by `_pallas_call_jvp_rule`), and the generic
# rule crashes on any kernel whose block index maps read `program_id`. Both
# wrappers therefore carry a `jax.custom_jvp` with a hand-written tangent, and
# the pure-JAX backend -- fully differentiable -- is the reference.


def _index_points_2d(key, m, ny, nx):
    i = jax.random.uniform(key, (m,), minval=-2.0, maxval=nx + 2.0)
    j = jax.random.uniform(jax.random.fold_in(key, 1), (m,), minval=-2.0, maxval=ny + 2.0)
    amplitude = jax.random.normal(jax.random.fold_in(key, 2), (m,)) * 2 + 3
    variance = jnp.abs(jax.random.normal(jax.random.fold_in(key, 3), (m,))) * 0.3 + 0.4
    return i, j, amplitude, variance


def _random_tangents(key, primals):
    return tuple(
        jax.random.normal(jax.random.fold_in(key, k), jnp.shape(p))
        for k, p in enumerate(primals)
    )


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_fwd_2d_jvp_matches_pure_jax(use_erf, scalar_variance):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_2d
    from cryojax.ndimage._spreading.spread import spread_2d_impl

    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(3), 150, ny, nx)
    if scalar_variance:
        variance = variance[0]
    pixel_size = jnp.asarray(1.3)
    primals = (i, j, amplitude, variance, pixel_size)
    tangents = _random_tangents(jax.random.PRNGKey(4), primals)

    ref = lambda i, j, a, v, p: spread_2d_impl(
        i, j, a, v, ny, nx, pixel_size=p, n_spread=n_spread, use_erf=use_erf
    )
    pallas = lambda i, j, a, v, p: pallas_spread_2d(
        i, j, a, v, p, ny, nx, n_spread, use_erf
    )
    out_ref, tan_ref = jax.jvp(ref, primals, tangents)
    out_pallas, tan_pallas = jax.jvp(pallas, primals, tangents)
    assert jnp.allclose(out_pallas, out_ref, atol=1e-4, rtol=1e-4)
    assert not jnp.any(jnp.isnan(tan_pallas))
    assert jnp.allclose(tan_pallas, tan_ref, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_2d_hvp_matches_pure_jax(points_2d, use_erf, scalar_variance):
    """Forward-over-reverse through the public API: what a Hessian-vector product
    in an optimizer actually asks for."""
    x, y, amplitude, variance, pixel_size, shape = points_2d
    if scalar_variance:
        variance = variance[0]
    n_spread = 7
    target = jax.random.normal(jax.random.PRNGKey(5), shape)

    def make_loss(key):
        def loss(x, y, amp, var, pix):
            image = _JIT_SPREAD_2D[key](x, y, amp, var, pix, shape, n_spread, use_erf)
            return jnp.sum((image - target) ** 2)

        return loss

    primals = (x, y, amplitude, variance, pixel_size)
    tangents = _random_tangents(jax.random.PRNGKey(6), primals)
    grad_ref, hvp_ref = jax.jvp(
        jax.grad(make_loss("pure_jax"), argnums=(0, 1, 2, 3, 4)), primals, tangents
    )
    grad_pallas, hvp_pallas = jax.jvp(
        jax.grad(make_loss("full"), argnums=(0, 1, 2, 3, 4)), primals, tangents
    )
    for ref, pallas in zip(grad_ref, grad_pallas):
        assert jnp.allclose(pallas, ref, atol=1e-4, rtol=1e-4)
    for ref, pallas in zip(hvp_ref, hvp_pallas):
        assert not jnp.any(jnp.isnan(pallas))
        assert jnp.allclose(pallas, ref, atol=1e-3, rtol=1e-3)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_bwd_2d_jvp_matches_pure_jax(use_erf, scalar_variance):
    """Tangents on both the residuals and the cotangent, so the linear-in-`g` half
    and the second-derivative half of the rule are exercised together."""
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_vjp_2d
    from cryojax.ndimage._spreading.spread import spread_2d_bwd

    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(8), 150, ny, nx)
    if scalar_variance:
        variance = variance[0]
    pixel_size = jnp.asarray(1.3)
    res = (i, j, amplitude, variance, pixel_size)
    g = jax.random.normal(jax.random.PRNGKey(9), (ny, nx))
    tangents = (_random_tangents(jax.random.PRNGKey(10), res), jnp.ones_like(g) * 0.3)

    ref = lambda res, g: spread_2d_bwd(ny, nx, n_spread, use_erf, res, g)
    pallas = lambda res, g: pallas_spread_vjp_2d(ny, nx, n_spread, use_erf, res, g)
    out_ref, tan_ref = jax.jvp(ref, (res, g), tangents)
    out_pallas, tan_pallas = jax.jvp(pallas, (res, g), tangents)
    for r, p in zip(out_ref, out_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    for r, p in zip(tan_ref, tan_pallas):
        assert not jnp.any(jnp.isnan(p))
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
def test_pallas_2d_jvp_specialized_on_symbolic_zeros(use_erf):
    """The two nonzero patterns an optimizer's shared-parameter HVP produces: a scatter
    tangent only in `pixel_size`, and a gather tangent only in the cotangent `g`."""
    from cryojax.ndimage._spreading.pallas_spread import (
        pallas_spread_2d,
        pallas_spread_vjp_2d,
    )
    from cryojax.ndimage._spreading.spread import spread_2d_bwd, spread_2d_impl

    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(11), 150, ny, nx)
    pixel_size = jnp.asarray(1.3)
    tpix = jnp.asarray(0.7)

    ref = lambda p: spread_2d_impl(
        i,
        j,
        amplitude,
        variance,
        ny,
        nx,
        pixel_size=p,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    pallas = lambda p: pallas_spread_2d(
        i, j, amplitude, variance, p, ny, nx, n_spread, use_erf
    )
    _, tan_ref = jax.jvp(ref, (pixel_size,), (tpix,))
    _, tan_pallas = jax.jvp(pallas, (pixel_size,), (tpix,))
    assert jnp.allclose(tan_pallas, tan_ref, atol=1e-4, rtol=1e-4)

    res = (i, j, amplitude, variance, pixel_size)
    g = jax.random.normal(jax.random.PRNGKey(12), (ny, nx))
    tg = jax.random.normal(jax.random.PRNGKey(13), (ny, nx))
    ref_g = lambda g: spread_2d_bwd(ny, nx, n_spread, use_erf, res, g)
    pallas_g = lambda g: pallas_spread_vjp_2d(ny, nx, n_spread, use_erf, res, g)
    _, tan_ref = jax.jvp(ref_g, (g,), (tg,))
    _, tan_pallas = jax.jvp(pallas_g, (g,), (tg,))
    for r, p in zip(tan_ref, tan_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
def test_pallas_2d_hvp_float32(points_2d, use_erf):
    """Production runs the kernels in float32; the file's history has a dtype-lowering
    bug, so the rules get their own float32 run."""
    x, y, amplitude, variance, pixel_size, shape = points_2d
    f32 = lambda a: jnp.asarray(a, dtype=jnp.float32)
    primals = tuple(map(f32, (x, y, amplitude, variance, pixel_size)))
    n_spread = 7
    target = f32(jax.random.normal(jax.random.PRNGKey(14), shape))

    def make_loss(key):
        def loss(x, y, amp, var, pix):
            image = _JIT_SPREAD_2D[key](x, y, amp, var, pix, shape, n_spread, use_erf)
            return jnp.sum((image - target) ** 2)

        return loss

    tangents = tuple(map(f32, _random_tangents(jax.random.PRNGKey(15), primals)))
    _, hvp_ref = jax.jvp(
        jax.grad(make_loss("pure_jax"), argnums=(0, 1, 2, 3, 4)), primals, tangents
    )
    _, hvp_pallas = jax.jvp(
        jax.grad(make_loss("full"), argnums=(0, 1, 2, 3, 4)), primals, tangents
    )
    for ref, pallas in zip(hvp_ref, hvp_pallas):
        assert pallas.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(pallas))
        assert jnp.allclose(pallas, ref, atol=1e-2, rtol=1e-2)


def _index_points_3d(key, m, nz, ny, nx):
    i = jax.random.uniform(key, (m,), minval=-2.0, maxval=nx + 2.0)
    j = jax.random.uniform(jax.random.fold_in(key, 1), (m,), minval=-2.0, maxval=ny + 2.0)
    k = jax.random.uniform(jax.random.fold_in(key, 2), (m,), minval=-2.0, maxval=nz + 2.0)
    amplitude = jax.random.normal(jax.random.fold_in(key, 3), (m,)) * 2 + 3
    variance = jnp.abs(jax.random.normal(jax.random.fold_in(key, 4), (m,))) * 0.3 + 0.4
    return i, j, k, amplitude, variance


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_fwd_3d_jvp_matches_pure_jax(use_erf, scalar_variance):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_3d
    from cryojax.ndimage._spreading.spread import spread_3d_impl

    nz, ny, nx, n_spread = 20, 24, 28, 5
    i, j, k, amplitude, variance = _index_points_3d(
        jax.random.PRNGKey(16), 150, nz, ny, nx
    )
    if scalar_variance:
        variance = variance[0]
    voxel_size = jnp.asarray(1.1)
    primals = (i, j, k, amplitude, variance, voxel_size)
    tangents = _random_tangents(jax.random.PRNGKey(17), primals)

    ref = lambda i, j, k, a, v, p: spread_3d_impl(
        i, j, k, a, v, nz, ny, nx, voxel_size=p, n_spread=n_spread, use_erf=use_erf
    )
    pallas = lambda i, j, k, a, v, p: pallas_spread_3d(
        i, j, k, a, v, p, nz, ny, nx, n_spread, use_erf
    )
    out_ref, tan_ref = jax.jvp(ref, primals, tangents)
    out_pallas, tan_pallas = jax.jvp(pallas, primals, tangents)
    assert jnp.allclose(out_pallas, out_ref, atol=1e-4, rtol=1e-4)
    assert not jnp.any(jnp.isnan(tan_pallas))
    assert jnp.allclose(tan_pallas, tan_ref, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
def test_pallas_bwd_3d_jvp_matches_pure_jax(use_erf, scalar_variance):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_vjp_3d
    from cryojax.ndimage._spreading.spread import spread_3d_bwd

    nz, ny, nx, n_spread = 20, 24, 28, 5
    i, j, k, amplitude, variance = _index_points_3d(
        jax.random.PRNGKey(18), 150, nz, ny, nx
    )
    if scalar_variance:
        variance = variance[0]
    voxel_size = jnp.asarray(1.1)
    res = (i, j, k, amplitude, variance, voxel_size)
    g = jax.random.normal(jax.random.PRNGKey(19), (nz, ny, nx))
    tangents = (_random_tangents(jax.random.PRNGKey(20), res), jnp.ones_like(g) * 0.3)

    ref = lambda res, g: spread_3d_bwd(nz, ny, nx, n_spread, use_erf, res, g)
    pallas = lambda res, g: pallas_spread_vjp_3d(nz, ny, nx, n_spread, use_erf, res, g)
    out_ref, tan_ref = jax.jvp(ref, (res, g), tangents)
    out_pallas, tan_pallas = jax.jvp(pallas, (res, g), tangents)
    for r, p in zip(out_ref, out_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    for r, p in zip(tan_ref, tan_pallas):
        assert not jnp.any(jnp.isnan(p))
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
def test_pallas_3d_hvp_matches_pure_jax(points_3d, use_erf):
    x, y, z, amplitude, variance, voxel_size, shape = points_3d
    n_spread = 5
    target = jax.random.normal(jax.random.PRNGKey(21), shape)

    def make_loss(key):
        def loss(x, y, z, amp, var, pix):
            volume = _JIT_SPREAD_3D[key](x, y, z, amp, var, pix, shape, n_spread, use_erf)
            return jnp.sum((volume - target) ** 2)

        return loss

    primals = (x, y, z, amplitude, variance, voxel_size)
    tangents = _random_tangents(jax.random.PRNGKey(22), primals)
    argnums = (0, 1, 2, 3, 4, 5)
    grad_ref, hvp_ref = jax.jvp(
        jax.grad(make_loss("pure_jax"), argnums=argnums), primals, tangents
    )
    grad_pallas, hvp_pallas = jax.jvp(
        jax.grad(make_loss("full"), argnums=argnums), primals, tangents
    )
    for ref, pallas in zip(grad_ref, grad_pallas):
        assert jnp.allclose(pallas, ref, atol=1e-4, rtol=1e-4)
    for ref, pallas in zip(hvp_ref, hvp_pallas):
        assert not jnp.any(jnp.isnan(pallas))
        assert jnp.allclose(pallas, ref, atol=1e-3, rtol=1e-3)


# ── Several tangent directions at one primal point ───────────────────────────
#
# `jax.vmap` over tangents (a batch of Hessian-vector products, `jax.jacfwd`) asks for
# the JVP in several directions at the same primal point. Pallas's generic batching
# would run one program per direction, recomputing every weight and the primal each
# time; the JVP wrappers instead route a tangent batch to a kernel that evaluates the
# weights once and accumulates all directions. The pure-JAX backend, batched by JAX
# itself, is the reference throughout.

from jax._src import core as _jax_core  # noqa: E402  (jaxpr inspection only)


def _pallas_calls(fn, *args):
    """Every `pallas_call` in `fn`'s traced program, as
    `(n_outputs, out_shape, n_grid_axes)`."""
    found = []

    def walk(v):
        if isinstance(v, _jax_core.ClosedJaxpr):
            visit(v.jaxpr)
        elif isinstance(v, _jax_core.Jaxpr):
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
                found.append((len(eqn.outvars), eqn.outvars[-1].aval.shape, len(grid)))
            for v in eqn.params.values():
                walk(v)

    visit(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


# Layer 0: the JAX contracts the routing relies on, pinned on toy functions (CPU).


def test_contract_custom_vmap_fires_inside_a_custom_jvp_rule():
    from jax.custom_batching import custom_vmap

    seen = []

    @custom_vmap
    def tangent_map(x, t):
        return x**3, 3 * x**2 * t

    @tangent_map.def_vmap
    def _(axis_size, in_batched, x, t):
        seen.append((axis_size, tuple(in_batched)))
        return (x**3, 3 * x**2 * t), (False, True)

    @jax.custom_jvp
    def f(x):
        return x**3

    @f.defjvp
    def _(primals, tangents):
        (x,), (t,) = primals, tangents
        return tangent_map(x, t)

    x = jnp.arange(1.0, 4.0)
    dirs = jnp.eye(3)[:2]
    for wrap in (lambda g: g, jax.jit):
        seen.clear()
        out = wrap(lambda d: jax.vmap(lambda dd: jax.jvp(f, (x,), (dd,)))(d))(dirs)
        assert seen == [(2, (False, True))]
        assert out[0].shape == (2, 3)  # an unbatched primal is broadcast for the caller
        assert jnp.allclose(out[1], 3 * x**2 * dirs)


def test_contract_symbolic_zeros_reach_the_rule():
    from jax.custom_derivatives import SymbolicZero

    seen = []

    @jax.custom_jvp
    def f(x, y):
        return x * y

    def rule(primals, tangents):
        seen.append(tuple(isinstance(t, SymbolicZero) for t in tangents))
        x, y = primals
        tx, ty = tangents
        return x * y, (0.0 if isinstance(tx, SymbolicZero) else tx * y) + (
            0.0 if isinstance(ty, SymbolicZero) else x * ty
        )

    f.defjvp(rule, symbolic_zeros=True)
    jax.jvp(lambda x: f(x, jnp.asarray(2.0)), (jnp.asarray(3.0),), (jnp.asarray(1.0),))
    assert seen == [(False, True)]


# Layer 2/4: several directions, against the pure-JAX backend and the single-direction
# kernel.


def _direction_tangents(key, primals, n_directions):
    return tuple(
        jax.random.normal(jax.random.fold_in(key, k), (n_directions, *jnp.shape(p)))
        for k, p in enumerate(primals)
    )


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
@pytest.mark.parametrize("n_directions", [1, 2, 3, 11])
def test_pallas_spread_jvp_2d_over_directions_matches_pure_jax(
    use_erf, scalar_variance, n_directions
):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_2d
    from cryojax.ndimage._spreading.spread import spread_2d_impl

    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(23), 150, ny, nx)
    if scalar_variance:
        variance = variance[0]
    primals = (i, j, amplitude, variance, jnp.asarray(1.3))
    dirs = _direction_tangents(jax.random.PRNGKey(24), primals, n_directions)

    ref = lambda i, j, a, v, p: spread_2d_impl(
        i, j, a, v, ny, nx, pixel_size=p, n_spread=n_spread, use_erf=use_erf
    )
    pallas = lambda i, j, a, v, p: pallas_spread_2d(
        i, j, a, v, p, ny, nx, n_spread, use_erf
    )
    batched_jvp = lambda f: jax.vmap(lambda t: jax.jvp(f, primals, t))(dirs)
    out_ref, tan_ref = batched_jvp(ref)
    out_pallas, tan_pallas = batched_jvp(pallas)
    assert jnp.allclose(out_pallas, out_ref, atol=1e-4, rtol=1e-4)
    assert not jnp.any(jnp.isnan(tan_pallas))
    assert jnp.allclose(tan_pallas, tan_ref, atol=1e-4, rtol=1e-4)
    # ...and each direction equals the single-direction kernel. Not bit for bit: the
    # scatter's atomic adds land in hardware order, so two launches on identical inputs
    # differ at the ULP level (the gather, which owns its outputs, is exact -- see its
    # test).
    for d in range(n_directions):
        single = jax.jvp(pallas, primals, tuple(t[d] for t in dirs))[1]
        assert jnp.allclose(tan_pallas[d], single, atol=1e-9, rtol=1e-9)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("scalar_variance", [False, True])
@pytest.mark.parametrize("n_directions", [1, 3, 11])
def test_pallas_spread_vjp_jvp_2d_over_directions_matches_pure_jax(
    use_erf, scalar_variance, n_directions
):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_vjp_2d
    from cryojax.ndimage._spreading.spread import spread_2d_bwd

    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(25), 150, ny, nx)
    if scalar_variance:
        variance = variance[0]
    res = (i, j, amplitude, variance, jnp.asarray(1.3))
    g = jax.random.normal(jax.random.PRNGKey(26), (ny, nx))
    dirs = (
        _direction_tangents(jax.random.PRNGKey(27), res, n_directions),
        jax.random.normal(jax.random.PRNGKey(28), (n_directions, ny, nx)),
    )
    ref = lambda res, g: spread_2d_bwd(ny, nx, n_spread, use_erf, res, g)
    pallas = lambda res, g: pallas_spread_vjp_2d(ny, nx, n_spread, use_erf, res, g)
    batched_jvp = lambda f: jax.vmap(lambda t: jax.jvp(f, (res, g), t))(dirs)
    out_ref, tan_ref = batched_jvp(ref)
    out_pallas, tan_pallas = batched_jvp(pallas)
    for r, p in zip(out_ref, out_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    for r, p in zip(tan_ref, tan_pallas):
        assert not jnp.any(jnp.isnan(p))
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    # Per-point outputs bit for bit (the gather owns its outputs; no atomics). The two
    # scalar outputs are `jnp.sum` reductions outside the kernel, over `(n, m)` here and
    # `(m,)` there, and XLA's GPU reductions are not bitwise reproducible across shapes.
    for d in range(n_directions):
        single = jax.jvp(pallas, (res, g), jax.tree.map(lambda t: t[d], dirs))[1]
        for s, p in zip(single, tan_pallas):
            if jnp.ndim(s) == 0:
                assert jnp.allclose(p[d], s, rtol=1e-12, atol=0.0)
            else:
                assert jnp.array_equal(p[d], s)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
def test_pallas_2d_directions_specialized_on_the_nonzero_pattern(use_erf):
    """Directions that move only the cotangent (an `ac`/`bf`-like border) and only
    `pixel_size` (a `ps`-like border)."""
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_vjp_2d
    from cryojax.ndimage._spreading.spread import spread_2d_bwd

    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(29), 150, ny, nx)
    res = (i, j, amplitude, variance, jnp.asarray(1.3))
    g = jax.random.normal(jax.random.PRNGKey(30), (ny, nx))
    tg = jax.random.normal(jax.random.PRNGKey(31), (3, ny, nx))
    tpix = jnp.asarray([1.0, 0.5, -0.25])
    ref = lambda res, g: spread_2d_bwd(ny, nx, n_spread, use_erf, res, g)
    pallas = lambda res, g: pallas_spread_vjp_2d(ny, nx, n_spread, use_erf, res, g)
    for f_ref, f_pallas in (
        (lambda gg: ref(res, gg), lambda gg: pallas(res, gg)),
        (lambda p: ref((*res[:4], p), g), lambda p: pallas((*res[:4], p), g)),
    ):
        pass
    tan_ref = jax.vmap(lambda t: jax.jvp(lambda gg: ref(res, gg), (g,), (t,))[1])(tg)
    tan_pallas = jax.vmap(lambda t: jax.jvp(lambda gg: pallas(res, gg), (g,), (t,))[1])(
        tg
    )
    for r, p in zip(tan_ref, tan_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    tan_ref = jax.vmap(
        lambda t: jax.jvp(lambda p: ref((*res[:4], p), g), (res[4],), (t,))[1]
    )(tpix)
    tan_pallas = jax.vmap(
        lambda t: jax.jvp(lambda p: pallas((*res[:4], p), g), (res[4],), (t,))[1]
    )(tpix)
    for r, p in zip(tan_ref, tan_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)


# Layer 3: routing. One launch per kernel kind, one grid axis, tangent outputs batched.


def _hvp_loss_2d(use_erf, enable_pallas):
    ny, nx, n_spread = 48, 40, 7
    i, j, amplitude, variance = _index_points_2d(jax.random.PRNGKey(32), 150, ny, nx)
    pixel_size = jnp.asarray(1.3)
    target = jax.random.normal(jax.random.PRNGKey(33), (ny, nx))
    x, y = i * pixel_size - nx / 2 * pixel_size, j * pixel_size - ny / 2 * pixel_size

    def loss(p):
        xx, yy, a, v, pix = p
        image = spread_gaussians_2d(
            xx,
            yy,
            a,
            v,
            (ny, nx),
            pixel_size=pix,
            n_spread=n_spread,
            use_erf=use_erf,
            enable_pallas=enable_pallas,
        )
        return jnp.sum((image - target) ** 2)

    return loss, (x, y, amplitude, variance, pixel_size)


@requires_gpu
@pytest.mark.parametrize("n_directions, n_launches", [(1, 2), (3, 2), (12, 2), (13, 4)])
def test_tangent_map_launches_one_multi_direction_kernel_per_kind(
    n_directions, n_launches
):
    """`jax.linearize` records the tangent map; applying it in `n_directions` runs the
    tangent scatter and tangent gather once each per chunk, on one grid axis, and no
    primal kernel at all -- the property an optimizer holding the gradient relies on."""
    from cryojax.ndimage._spreading.pallas_spread import _MAX_JVP_DIRECTIONS

    assert _MAX_JVP_DIRECTIONS == 12  # the (13, 4) row above assumes two chunks
    loss, p = _hvp_loss_2d(use_erf=True, enable_pallas=True)
    dirs = _direction_tangents(jax.random.PRNGKey(34), p, n_directions)
    _, tangent_map = jax.linearize(jax.value_and_grad(loss), p)
    calls = _pallas_calls(lambda d: jax.vmap(tangent_map)(d), dirs)
    assert len(calls) == n_launches
    cap = _MAX_JVP_DIRECTIONS
    n_pixels, chunk_sizes = 48 * 40, {min(n_directions, cap), n_directions % cap or cap}
    for n_outputs, out_shape, n_grid_axes in calls:
        assert n_grid_axes == 1  # generic batching would add a second grid axis
        # tangent scatter: one flattened `(directions * pixels,)` image; tangent gather:
        # five `(directions, points)` outputs
        assert n_outputs in (1, 5)
        directions = out_shape[0] // n_pixels if n_outputs == 1 else out_shape[0]
        assert directions in chunk_sizes


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("n_directions", [1, 3])
def test_linearized_hvp_2d_matches_pure_jax(use_erf, n_directions):
    """The optimizer's composition: gradient once, then the tangent map in the needed
    directions, against the pure-JAX backend doing the same."""
    loss_p, p = _hvp_loss_2d(use_erf, enable_pallas=True)
    loss_r, _ = _hvp_loss_2d(use_erf, enable_pallas=False)
    dirs = _direction_tangents(jax.random.PRNGKey(44), p, n_directions)

    def step(loss):
        (f, g), tangent_map = jax.linearize(jax.value_and_grad(loss), p)
        return f, g, jax.vmap(tangent_map)(dirs)[1]

    f_r, g_r, hv_r = jax.jit(lambda: step(loss_r))()
    f_p, g_p, hv_p = jax.jit(lambda: step(loss_p))()
    assert jnp.allclose(f_p, f_r, rtol=1e-6)
    for r, q in zip(g_r, g_p):
        assert jnp.allclose(q, r, atol=1e-4, rtol=1e-4)
    for r, q in zip(hv_r, hv_p):
        assert not jnp.any(jnp.isnan(q))
        assert jnp.allclose(q, r, atol=1e-3, rtol=1e-3)


@requires_gpu
def test_reverse_mode_never_enters_the_direction_path():
    loss, p = _hvp_loss_2d(use_erf=True, enable_pallas=True)
    calls = _pallas_calls(jax.grad(loss), p)
    assert [c[0] for c in calls] == [1, 5]  # plain scatter, plain gather


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("n_directions", [2, 3])
def test_public_vmapped_hvp_2d_matches_pure_jax(use_erf, n_directions):
    loss_p, p = _hvp_loss_2d(use_erf, enable_pallas=True)
    loss_r, _ = _hvp_loss_2d(use_erf, enable_pallas=False)
    dirs = _direction_tangents(jax.random.PRNGKey(35), p, n_directions)
    hvp = lambda loss: jax.jit(
        lambda d: jax.vmap(lambda dd: jax.jvp(jax.grad(loss), (p,), (dd,))[1])(d)
    )(dirs)
    for r, q in zip(hvp(loss_r), hvp(loss_p)):
        assert not jnp.any(jnp.isnan(q))
        assert jnp.allclose(q, r, atol=1e-3, rtol=1e-3)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("n_directions", [1, 3, 11])
def test_pallas_spread_jvp_3d_over_directions_matches_pure_jax(use_erf, n_directions):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_3d
    from cryojax.ndimage._spreading.spread import spread_3d_impl

    nz, ny, nx, n_spread = 20, 24, 28, 5
    i, j, k, amplitude, variance = _index_points_3d(
        jax.random.PRNGKey(36), 150, nz, ny, nx
    )
    primals = (i, j, k, amplitude, variance, jnp.asarray(1.1))
    dirs = _direction_tangents(jax.random.PRNGKey(37), primals, n_directions)
    ref = lambda i, j, k, a, v, p: spread_3d_impl(
        i, j, k, a, v, nz, ny, nx, voxel_size=p, n_spread=n_spread, use_erf=use_erf
    )
    pallas = lambda i, j, k, a, v, p: pallas_spread_3d(
        i, j, k, a, v, p, nz, ny, nx, n_spread, use_erf
    )
    batched_jvp = lambda f: jax.vmap(lambda t: jax.jvp(f, primals, t))(dirs)
    out_ref, tan_ref = batched_jvp(ref)
    out_pallas, tan_pallas = batched_jvp(pallas)
    assert jnp.allclose(out_pallas, out_ref, atol=1e-4, rtol=1e-4)
    assert not jnp.any(jnp.isnan(tan_pallas))
    assert jnp.allclose(tan_pallas, tan_ref, atol=1e-4, rtol=1e-4)
    for d in range(n_directions):
        single = jax.jvp(pallas, primals, tuple(t[d] for t in dirs))[1]
        assert jnp.allclose(tan_pallas[d], single, atol=1e-9, rtol=1e-9)


@requires_gpu
@pytest.mark.parametrize("use_erf", [False, True])
@pytest.mark.parametrize("n_directions", [1, 3, 11])
def test_pallas_spread_vjp_jvp_3d_over_directions_matches_pure_jax(use_erf, n_directions):
    from cryojax.ndimage._spreading.pallas_spread import pallas_spread_vjp_3d
    from cryojax.ndimage._spreading.spread import spread_3d_bwd

    nz, ny, nx, n_spread = 20, 24, 28, 5
    i, j, k, amplitude, variance = _index_points_3d(
        jax.random.PRNGKey(38), 150, nz, ny, nx
    )
    res = (i, j, k, amplitude, variance, jnp.asarray(1.1))
    g = jax.random.normal(jax.random.PRNGKey(39), (nz, ny, nx))
    dirs = (
        _direction_tangents(jax.random.PRNGKey(40), res, n_directions),
        jax.random.normal(jax.random.PRNGKey(41), (n_directions, nz, ny, nx)),
    )
    ref = lambda res, g: spread_3d_bwd(nz, ny, nx, n_spread, use_erf, res, g)
    pallas = lambda res, g: pallas_spread_vjp_3d(nz, ny, nx, n_spread, use_erf, res, g)
    batched_jvp = lambda f: jax.vmap(lambda t: jax.jvp(f, (res, g), t))(dirs)
    out_ref, tan_ref = batched_jvp(ref)
    out_pallas, tan_pallas = batched_jvp(pallas)
    for r, p in zip(out_ref, out_pallas):
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    for r, p in zip(tan_ref, tan_pallas):
        assert not jnp.any(jnp.isnan(p))
        assert jnp.allclose(p, r, atol=1e-4, rtol=1e-4)
    for d in range(n_directions):
        single = jax.jvp(pallas, (res, g), jax.tree.map(lambda t: t[d], dirs))[1]
        for s, p in zip(single, tan_pallas):
            if jnp.ndim(s) == 0:
                assert jnp.allclose(p[d], s, rtol=1e-12, atol=0.0)
            else:
                assert jnp.array_equal(p[d], s)


@requires_gpu
@pytest.mark.parametrize("n_directions", [2, 3])
def test_public_vmapped_hvp_3d_matches_pure_jax(points_3d, n_directions):
    x, y, z, amplitude, variance, voxel_size, shape = points_3d
    n_spread = 5
    target = jax.random.normal(jax.random.PRNGKey(42), shape)

    def make_loss(key):
        def loss(p):
            volume = _JIT_SPREAD_3D[key](*p, shape, n_spread, True)
            return jnp.sum((volume - target) ** 2)

        return loss

    p = (x, y, z, amplitude, variance, voxel_size)
    dirs = _direction_tangents(jax.random.PRNGKey(43), p, n_directions)
    hvp = lambda loss: jax.jit(
        lambda d: jax.vmap(lambda dd: jax.jvp(jax.grad(loss), (p,), (dd,))[1])(d)
    )(dirs)
    for r, q in zip(hvp(make_loss("pure_jax")), hvp(make_loss("full"))):
        assert not jnp.any(jnp.isnan(q))
        assert jnp.allclose(q, r, atol=1e-3, rtol=1e-3)


@requires_gpu
@pytest.mark.parametrize("n_directions", [1, 3])
def test_linearized_hvp_3d_matches_pure_jax(points_3d, n_directions):
    x, y, z, amplitude, variance, voxel_size, shape = points_3d
    n_spread = 5
    target = jax.random.normal(jax.random.PRNGKey(45), shape)

    def make_loss(key):
        def loss(p):
            volume = _JIT_SPREAD_3D[key](*p, shape, n_spread, True)
            return jnp.sum((volume - target) ** 2)

        return loss

    p = (x, y, z, amplitude, variance, voxel_size)
    dirs = _direction_tangents(jax.random.PRNGKey(46), p, n_directions)

    def step(loss):
        (f, g), tangent_map = jax.linearize(jax.value_and_grad(loss), p)
        return f, g, jax.vmap(tangent_map)(dirs)[1]

    f_r, g_r, hv_r = jax.jit(lambda: step(make_loss("pure_jax")))()
    f_p, g_p, hv_p = jax.jit(lambda: step(make_loss("full")))()
    assert jnp.allclose(f_p, f_r, rtol=1e-6)
    for r, q in zip(g_r, g_p):
        assert jnp.allclose(q, r, atol=1e-4, rtol=1e-4)
    for r, q in zip(hv_r, hv_p):
        assert not jnp.any(jnp.isnan(q))
        assert jnp.allclose(q, r, atol=1e-3, rtol=1e-3)
