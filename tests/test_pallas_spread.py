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
    pallas_interp_bwd_2d,
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
# `pallas_interp_bwd_{2,3}d` don't expose an `interpret` argument (the real
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
    di, dj, damplitude, dvariance, dpixel_size = pallas_interp_bwd_2d(
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
