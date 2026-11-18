import numpy as np
import pytest
from cryojax.ndimage._nufft.kernel import (
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
    "eps, sigma, ndim, width",
    [(1e-4, 1.25, 2, 7), (1e-4, 2.0, 2, 5), (1e-6, 1.25, 2, 10), (1e-6, 1.25, 3, 11),
     (1e-3, 1.25, 3, 6)],
)  # fmt: skip
def test_kernel_width_matches_finufft_rule(eps, sigma, ndim, width):
    assert kernel_width(eps, sigma, ndim) == width


def test_kernel_width_raises():
    with pytest.raises(
        ValueError, match=r"kernel_width.*eps.*1e-10.*upsample_factor.*1.25"
    ):
        kernel_width(1e-10, 1.25, 2)
    with pytest.raises(ValueError, match=r"kernel_width.*upsample_factor.*1.0"):
        kernel_width(1e-4, 1.0, 2)


@pytest.mark.parametrize("width", range(4, 17))
def test_pswf_eigen_residual(width):
    sigma = 1.25
    kernel = make_pswf_kernel(*_eps_for_width(width, sigma), 2)
    assert kernel.width == width
    x = np.linspace(-1, 1, 501)
    t, v = np.polynomial.legendre.leggauss(400)
    values = evaluate_pswf(kernel, t)
    lhs = np.cos(kernel.shape_parameter * np.outer(x, t)) @ (v * values)
    rhs = kernel.eigenvalue * evaluate_pswf(kernel, x)
    assert np.max(np.abs(lhs - rhs)) / kernel.eigenvalue < 1e-12


def _eps_for_width(width, sigma):
    """An `eps` (and `sigma`) whose rule gives exactly `width`."""
    for exponent in np.arange(1.0, 9.6, 0.05):
        eps = 10.0 ** (-exponent)
        try:
            if kernel_width(eps, sigma, 2) == width:
                return eps, sigma
        except ValueError:
            break
    raise AssertionError(width)


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", (1e-2, 1e-4, 1e-6))
def test_pswf_matches_scipy_pro_ang1(eps, sigma):
    kernel = make_pswf_kernel(eps, sigma, 2)
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
    try:
        kernel = make_pswf_kernel(eps, sigma, 2)
    except ValueError:
        pytest.skip("width above 16")
    values = evaluate_pswf(kernel, np.linspace(0, 1, 401))
    assert np.all(values > 0)
    assert np.all(np.diff(values) < 0)


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", (1e-3, 1e-6))
def test_self_fourier_identity(eps, sigma):
    kernel = make_pswf_kernel(eps, sigma, 3)
    w = kernel.width
    n, m = 32, int(np.ceil(sigma * 32))
    got = kernel_fourier_series(kernel, n, m, rfft=False)
    z, v = np.polynomial.legendre.leggauss(300)
    z, v = 0.5 * w * z, 0.5 * w * v
    xi = np.fft.fftfreq(n) * n / m
    ref = np.cos(2 * np.pi * np.outer(xi, z)) @ (v * evaluate_pswf(kernel, 2 * z / w))
    assert np.max(np.abs(got - ref) / np.abs(ref)) < 1e-12
    half = kernel_fourier_series(kernel, n, m, rfft=True)
    np.testing.assert_allclose(half, got[: n // 2 + 1], rtol=1e-14)


@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("eps", [10.0 ** (-k) for k in range(2, 10)])
def test_tap_fit_error_below_eps(eps, sigma, ndim):
    try:
        kernel = make_pswf_kernel(eps, sigma, ndim)
    except ValueError:
        pytest.skip("width above 16")
    w = kernel.width
    s = np.linspace(0, 1, 1001, endpoint=False)
    exact = np.array([evaluate_pswf(kernel, 2 * (s + j - w / 2) / w) for j in range(w)])
    assert tap_weights(kernel, s).shape == (w, 1001)
    assert np.max(np.abs(tap_weights(kernel, s) - exact)) <= eps / 10


def test_fine_grid_size():
    assert fine_grid_size(64, 1.25, 7) == 80
    assert fine_grid_size(5, 1.25, 7) == 16
    for n, sigma, w in [(7, 2.0, 16), (100, 1.5, 9), (33, 1.25, 12)]:
        m = fine_grid_size(n, sigma, w)
        assert m % 2 == 0 and m >= max(np.ceil(sigma * n), 2 * w)
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        assert r == 1


def test_make_pswf_kernel_is_cached():
    assert make_pswf_kernel(1e-4, 1.25, 2) is make_pswf_kernel(1e-4, 1.25, 2)


def _auto(n_points, shape, eps, dtype=np.float32):
    return resolve_upsample_factor(
        "auto", n_points=n_points, n_batch=1, shape=shape, eps=eps, dtype=dtype
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_auto_picks_low_sigma_when_sparse(dtype):
    assert _auto(10_000, (4096, 4096), 1e-4, dtype) == 1.25


def test_auto_picks_high_sigma_when_dense():
    assert _auto(100_000_000, (4096, 4096), 1e-6) == 2.0


@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("eps", (1e-9, 1e-10, 1e-11, 1e-12))
def test_auto_respects_width_cap(eps, ndim):
    sigma = _auto(10, (256,) * ndim, eps, np.float64)
    assert kernel_width(eps, sigma, ndim) <= 16


def test_auto_rounding_steers_float32_3d_from_low_sigma():
    """Sparse points favour 1.25, but in float32 its deconvolution amplifies rounding
    beyond `eps` (the benchmark's 1.4e-3 at eps 1e-6, w = 11)."""
    shape = (256, 256, 256)
    assert _auto(10_000, shape, 1e-6, np.float64) == 1.25
    assert _auto(10_000, shape, 1e-6, np.float32) == 1.5
    assert _auto(10_000, shape, 1e-4, np.float32) == 1.25


def test_auto_takes_most_accurate_when_eps_unreachable():
    assert _auto(10, (64, 64, 64), 1e-9, np.float32) == 2.0


def test_explicit_upsample_factor_passes_through():
    kwargs = dict(n_points=10, n_batch=1, shape=(8, 8), eps=1e-4, dtype=np.float32)
    assert resolve_upsample_factor(1.6, **kwargs) == 1.6
    for bad in (1.0, "fast"):
        with pytest.raises(
            ValueError, match=rf"resolve_upsample_factor[\s\S]*upsample_factor.*{bad}"
        ):
            resolve_upsample_factor(bad, **kwargs)
