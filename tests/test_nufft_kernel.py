import numpy as np
import pytest
from cryojax.ndimage._nufft.kernel import (
    evaluate_pswf,
    fine_grid_size,
    kernel_fourier_series,
    kernel_width,
    make_pswf_kernel,
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
