import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage import nufft_accuracy_limits
from cryojax.ndimage._nufft.kernel import (
    _auto_upsample_factor,
    evaluate_pswf,
    fine_grid_size,
    kernel_fourier_series,
    kernel_width,
    make_pswf_kernel,
    resolve_upsample_factor,
    tap_weights,
)
from scipy.special import pro_ang1


SIGMAS = (1.25, 1.5, 2.0)


@pytest.mark.parametrize(
    "eps, sigma, ndim, dtype, width",
    [
        (1e-6, 1.25, 3, np.float32, 8),  # float32 guard: 8 at sigma 1.25
        (1e-12, 1.5, 2, np.float32, 11),  # guard 8 + floor(0.275 / 0.075)
        (1e-12, 1.5, 3, np.float32, 10),  # guard 8 + floor(0.15 / 0.075)
        (1e-12, 2.0, 2, np.float32, 12),  # float32 maximum
        (1e-15, 2.0, 2, np.float64, 16),  # float64 maximum, clipped not raised
        (1e-6, 1.25, 2, np.float64, 10),  # rule 1 unchanged
    ],
)
def test_kernel_width_follows_finufft_26(eps, sigma, ndim, dtype, width):
    assert kernel_width(eps, sigma, ndim, dtype) == width


def test_kernel_width_raises_at_most_one():
    with pytest.raises(ValueError, match=r"kernel_width.*upsample_factor.*1.0"):
        kernel_width(1e-4, 1.0, 2, np.float64)


@pytest.mark.parametrize("width", range(4, 17))
def test_pswf_eigen_residual(width):
    sigma = 1.25
    kernel = make_pswf_kernel(_eps_for_width(width, sigma), sigma, 2, np.float64)
    assert kernel.width == width
    x = np.linspace(-1, 1, 501)
    t, v = np.polynomial.legendre.leggauss(400)
    values = evaluate_pswf(kernel, t)
    lhs = np.cos(kernel.shape_parameter * np.outer(x, t)) @ (v * values)
    rhs = kernel.eigenvalue * evaluate_pswf(kernel, x)
    assert np.max(np.abs(lhs - rhs)) / kernel.eigenvalue < 1e-12


def _eps_for_width(width, sigma):
    """An `eps` whose rule gives exactly `width` in 2D, float64."""
    for exponent in np.arange(1.0, 12.0, 0.05):
        eps = 10.0 ** (-exponent)
        if kernel_width(eps, sigma, 2, np.float64) == width:
            return eps
    raise AssertionError(width)


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", (1e-2, 1e-4, 1e-6))
def test_pswf_matches_scipy_pro_ang1(eps, sigma):
    kernel = make_pswf_kernel(eps, sigma, 2, np.float64)
    assert kernel.width <= 10
    x = np.linspace(-1, 1, 401)[1:-1]
    c = kernel.shape_parameter
    ref = pro_ang1(0, 0, c, x)[0] / pro_ang1(0, 0, c, 0.0)[0]
    assert np.max(np.abs(evaluate_pswf(kernel, x) - ref)) < 1e-6


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", [10.0 ** (-k) for k in range(2, 12)])
def test_pswf_is_positive_and_decreasing(eps, sigma):
    """psi_0 has no zero in [-1, 1]; at large `c` the other even eigenfunctions share its
    eigenvalue to rounding, and only this tells them apart."""
    kernel = make_pswf_kernel(eps, sigma, 2, np.float64)
    values = evaluate_pswf(kernel, np.linspace(0, 1, 401))
    assert np.all(values > 0)
    assert np.all(np.diff(values) < 0)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", (1e-2, 1e-4, 1e-6, 1e-9, 1e-12))
def test_coefficient_count_within_finufft_bounds(eps, sigma, ndim, dtype):
    kernel = make_pswf_kernel(eps, sigma, ndim, dtype)
    w, nc = kernel.coefficients.shape
    assert w == kernel.width
    assert max(4, w - 4) <= nc <= min(19, w + 3)


def test_coefficient_count_drops_below_width():
    """finufft keeps the coefficients above `0.05 eps`: wide kernels at low `sigma` need
    fewer than they have taps."""
    kernel = make_pswf_kernel(1e-8, 1.25, 2, np.float64)
    assert kernel.coefficients.shape == (14, 12)


@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", (1e-2, 1e-4, 1e-6, 1e-9))
def test_tap_polynomials_match_pswf(eps, sigma, ndim):
    kernel = make_pswf_kernel(eps, sigma, ndim, np.float64)
    w = kernel.width
    s = np.linspace(0, 1, 1001)
    exact = np.array([evaluate_pswf(kernel, 2 * (s + j - w / 2) / w) for j in range(w)])
    assert tap_weights(kernel, s).shape == (w, 1001)
    assert np.max(np.abs(tap_weights(kernel, s) - exact)) <= eps


@pytest.mark.parametrize("sigma", (1.25, 1.4713, 2.0))
@pytest.mark.parametrize("eps", (1e-3, 1e-6, 1e-9))
def test_fourier_series_matches_analytic_pswf(eps, sigma):
    """Rule 6's quadrature of the fitted taps against the PSWF's self-Fourier transform
    `(w / 2) lambda psi(pi k w / (c m))`, in `fftfreq` order."""
    kernel = make_pswf_kernel(eps, sigma, 3, np.float64)
    w, c = kernel.width, kernel.shape_parameter
    n = 33
    m = fine_grid_size(n, sigma, w)
    got = kernel_fourier_series(kernel, n, m, rfft=False, dtype=np.float64)
    k = np.fft.fftfreq(n, 1 / n)
    ref = 0.5 * w * kernel.eigenvalue * evaluate_pswf(kernel, np.pi * k * w / (c * m))
    assert np.max(np.abs(got - ref)) <= eps * ref[0]
    half = kernel_fourier_series(kernel, n, m, rfft=True, dtype=np.float64)
    np.testing.assert_array_equal(half, got[: n // 2 + 1])


@pytest.mark.parametrize("eps", (1e-4, 1e-6))
def test_float32_taps_fitted_in_float32(eps):
    """finufft fits float32 taps in float32 (include/finufft/makeplan.hpp:245); in
    float32 they still reach `eps` (measured: 5.1 eps32 at 1e-4, 1.1 at 1e-6)."""
    kernel = make_pswf_kernel(eps, 2.0, 2, np.float32)
    kernel64 = make_pswf_kernel(eps, 2.0, 2, np.float64)
    assert kernel.coefficients.shape == kernel64.coefficients.shape
    assert not np.array_equal(kernel.coefficients, kernel64.coefficients)
    w = kernel.width
    s = np.linspace(0, 1, 1001, dtype=np.float32)
    taps = np.zeros((w, s.size), np.float32)
    for coefficient in kernel.coefficients.astype(np.float32).T:
        taps = taps * s + coefficient[:, None]
    exact = np.array([evaluate_pswf(kernel, 2 * (s + j - w / 2) / w) for j in range(w)])
    assert np.max(np.abs(taps - exact)) <= max(eps, 8 * np.finfo(np.float32).eps)


def test_fourier_series_in_transform_precision():
    kernel = make_pswf_kernel(1e-4, 1.25, 2, np.float32)
    phi_hat = kernel_fourier_series(kernel, 16, 20, rfft=True, dtype=np.float32)
    assert phi_hat.dtype == np.float32


def test_fine_grid_size():
    assert fine_grid_size(64, 1.25, 7) == 80
    assert fine_grid_size(5, 1.25, 7) == 16
    for n, sigma, w in [(7, 2.0, 16), (100, 1.5, 9), (33, 1.25, 12), (1024, 1.4713, 9)]:
        m = fine_grid_size(n, sigma, w)
        assert m % 2 == 0 and m >= max(np.ceil(sigma * n), 2 * w)
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        assert r == 1


def test_make_pswf_kernel_is_cached():
    kernel = make_pswf_kernel(1e-4, 1.25, 2, np.float32)
    assert kernel is make_pswf_kernel(1e-4, 1.25, 2, np.dtype("float32"))
    assert kernel is make_pswf_kernel(1e-4, 1.25, 2, jnp.float32)
    assert kernel is not make_pswf_kernel(1e-4, 1.25, 2, np.float64)
    for array in (kernel.nodes, kernel.node_values, kernel.coefficients):
        assert not array.flags.writeable


def test_rounding_floor():
    limits = nufft_accuracy_limits(
        (4096, 4096), eps=1e-3, upsample_factor=2.0, dtype=np.float32
    )
    assert limits.rounding_floor == pytest.approx(2.4169e-4, rel=1e-3)
    limits = nufft_accuracy_limits(
        (1024, 1024), eps=1e-3, upsample_factor=2.0, dtype=np.float32
    )
    assert limits.rounding_floor == pytest.approx(6.0422e-5, rel=1e-3)


def test_min_upsample_factor():
    limits = nufft_accuracy_limits(
        (64, 64), eps=1e-10, upsample_factor=1.25, dtype=np.float64
    )
    assert (limits.width, limits.min_upsample_factor) == (
        16,
        pytest.approx(1.2675, abs=1e-4),
    )
    limits = nufft_accuracy_limits(
        (32, 32, 32), eps=1e-6, upsample_factor=1.25, dtype=np.float32
    )
    assert (limits.width, limits.min_upsample_factor) == (
        8,
        pytest.approx(2.0397, abs=1e-4),
    )
    # Unreachable at any sigma up to 2.5, though above the floor (2.4e-7)
    limits = nufft_accuracy_limits(
        (4, 4), eps=3e-7, upsample_factor=2.5, dtype=np.float32
    )
    assert limits.rounding_floor < 3e-7
    assert limits.min_upsample_factor > 2.5


def test_complex_dtypes_read_as_real():
    """At 3D, sigma 1.25 and eps 1e-6 the float32 guard narrows 11 to 8."""
    shape, kwargs = (32, 32, 32), dict(eps=1e-6, upsample_factor=1.25)
    for real, complex_ in ((jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)):
        assert nufft_accuracy_limits(
            shape, dtype=complex_, **kwargs
        ) == nufft_accuracy_limits(shape, dtype=real, **kwargs)
        assert make_pswf_kernel(1e-6, 1.25, 3, complex_) is make_pswf_kernel(
            1e-6, 1.25, 3, real
        )
        assert _auto(10, shape, 1e-6, complex_, True) == _auto(
            10, shape, 1e-6, real, True
        )
    assert nufft_accuracy_limits(shape, dtype=jnp.complex128, **kwargs).width == 11


def _auto(n_points, shape, eps, dtype, pallas):
    return resolve_upsample_factor(
        "auto", n_points=n_points, shape=shape, eps=eps, dtype=dtype, pallas=pallas
    )


@pytest.mark.parametrize(
    "pallas, shape, eps, dtype, n_points, sigma, width",
    [
        (True, (1024, 1024), 1e-6, np.float64, 10_000, 1.251, 10),
        (True, (1024, 1024), 1e-6, np.float64, 104_858, 1.251, 10),
        (True, (1024, 1024), 1e-6, np.float64, 1_048_576, 1.772, 7),
        (True, (256, 256, 256), 1e-4, np.float64, 1_677_722, 1.3717, 6),
        (False, (1024, 1024), 1e-6, np.float64, 104_858, 1.772, 7),
        (False, (256, 256, 256), 1e-4, np.float64, 1_677_722, 1.7327, 5),
    ],
)
def test_auto_follows_finufft_search(pallas, shape, eps, dtype, n_points, sigma, width):
    resolved = _auto(n_points, shape, eps, dtype, pallas)
    assert resolved == pytest.approx(sigma, abs=1e-3)
    assert kernel_width(eps, resolved, len(shape), dtype) == width


def test_auto_unreachable_returns_upper_limit():
    assert _auto(10, (4, 4), 3e-7, np.float32, True) == 2.5


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("shape", ((8, 8), (1024, 1024), (8, 8, 8), (256, 256, 256)))
def test_auto_fails_sigma_limit_only_when_none_meets_it(shape, dtype):
    """The dispatch warnings rely on this: `"auto"` fails the sigma limit only when no
    factor up to 2.5 meets it, and then it is 2.5."""
    for eps in np.logspace(-16, -2, 113):
        if eps < np.finfo(dtype).eps:
            continue
        kwargs = dict(eps=float(eps), dtype=dtype)
        reachable = (
            nufft_accuracy_limits(
                shape, upsample_factor=2.5, **kwargs
            ).min_upsample_factor
            <= 2.5
        )
        for n_points, pallas in ((10, True), (10_000_000, True), (10_000_000, False)):
            sigma = _auto(n_points, shape, float(eps), dtype, pallas)
            limits = nufft_accuracy_limits(shape, upsample_factor=sigma, **kwargs)
            assert (sigma >= limits.min_upsample_factor) == reachable
            assert reachable or sigma == 2.5


def test_auto_is_cached():
    sigma = _auto(10_000, (64, 64), 1e-6, np.float32, True)
    hits = _auto_upsample_factor.cache_info().hits
    assert type(sigma) is float
    assert sigma == _auto(10_000, (64, 64), 1e-6, jnp.float32, True)
    assert _auto_upsample_factor.cache_info().hits == hits + 1


def test_explicit_upsample_factor_passes_through():
    kwargs = dict(n_points=10, shape=(8, 8), eps=1e-4, dtype=np.float32, pallas=True)
    assert resolve_upsample_factor(1.6, **kwargs) == 1.6
    for bad in (1.0, "fast"):
        with pytest.raises(
            ValueError, match=rf"resolve_upsample_factor[\s\S]*upsample_factor.*{bad}"
        ):
            resolve_upsample_factor(bad, **kwargs)
