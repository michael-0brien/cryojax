"""Host-side PSWF kernel table for the native NUFFT (numpy, float64)."""

import functools
import math
from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np


_MAX_WIDTH = 16
_NUM_NODES = 256
_NUM_LEGENDRE = 100
_MAX_EXTRA_DEGREE = 6

_AUTO_UPSAMPLE_FACTORS = (1.25, 1.5, 1.75, 2.0)
# Value + gradient ms = a R M w^d + b R prod(m) log2(prod(m)); rounding error
# k eps_mach r_dyn. Fitted 2026-11-17 on an RTX PRO 5000 (Blackwell), float32, Pallas:
# benchmarks/benchmark_nufft.py, 2026-11-17-native-nufft.benchmark.json.
_SPREAD_COST, _FFT_COST, _ROUNDING_FACTOR = 1.331e-8, 3.782e-9, 3e-3


class PSWFKernel(NamedTuple):
    width: int
    shape_parameter: float
    eigenvalue: float
    nodes: np.ndarray
    node_values: np.ndarray
    coefficients: np.ndarray


def kernel_width(eps: float, upsample_factor: float, ndim: int) -> int:
    """Kernel width in fine-grid pixels, from finufft's rule."""
    if not upsample_factor > 1:
        raise ValueError(
            "Found invalid value for `kernel_width(eps, upsample_factor, ndim)`. "
            f"`upsample_factor` must be greater than 1, got upsample_factor="
            f"{upsample_factor}."
        )
    tolfac = 0.18 * 1.4 ** (ndim - 1)
    width = math.ceil(
        math.log(tolfac / eps) / (math.pi * math.sqrt(1 - 1 / upsample_factor)) + 1
    )
    if width > _MAX_WIDTH:
        raise ValueError(
            "Found invalid value for `kernel_width(eps, upsample_factor, ndim)`. "
            f"`eps`={eps} with `upsample_factor`={upsample_factor} needs width "
            f"{width}, above the maximum of {_MAX_WIDTH}."
        )
    return max(width, 2)


def _pswf_at_nodes(c: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """`psi_0` at Gauss-Legendre nodes and its eigenvalue, from the Legendre expansion
    of the prolate differential equation (Bouwkamp): the integral operator's top
    eigenvalues coincide to rounding at large `c`, the differential operator's do not."""
    k = 2 * np.arange(_NUM_LEGENDRE)
    diagonal = k * (k + 1) + c**2 * (2 * k * (k + 1) - 1) / ((2 * k + 3) * (2 * k - 1))
    k = k[:-1]
    off_diagonal = (
        c**2 * (k + 2) * (k + 1) / ((2 * k + 3) * np.sqrt((2 * k + 1) * (2 * k + 5)))
    )
    matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    _, vectors = np.linalg.eigh(matrix)
    legendre = np.zeros(2 * _NUM_LEGENDRE)
    legendre[::2] = vectors[:, 0] * np.sqrt(2 * np.arange(0, 2 * _NUM_LEGENDRE, 2) + 1)
    legendre /= np.polynomial.legendre.legval(0.0, legendre)
    nodes, weights = np.polynomial.legendre.leggauss(_NUM_NODES)
    # psi(0) = 1, so the eigenvalue is the integral of psi
    return 2 * legendre[0], nodes, weights, np.polynomial.legendre.legval(nodes, legendre)


def evaluate_pswf(kernel: PSWFKernel, x) -> np.ndarray:
    """PSWF `psi(x)` with `psi(0) = 1` for `|x| <= 1`, by Nyström interpolation."""
    x = np.asarray(x, dtype=np.float64)
    _, weights = np.polynomial.legendre.leggauss(_NUM_NODES)
    cos = np.cos(kernel.shape_parameter * x[..., None] * kernel.nodes)
    return cos @ (weights * kernel.node_values) / kernel.eigenvalue


def _fit_taps(kernel: PSWFKernel, degree: int) -> np.ndarray:
    w = kernel.width
    k = np.arange(4 * degree)
    s = 0.5 - 0.5 * np.cos(np.pi * (k + 0.5) / (4 * degree))
    return np.array(
        [
            np.polyfit(s, evaluate_pswf(kernel, 2 * (s + j - w / 2) / w), degree)
            for j in range(w)
        ]
    )


def _horner(coefficients: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.zeros(coefficients.shape[:1] + s.shape)
    for k in range(coefficients.shape[1]):
        out = out * s + coefficients[:, k].reshape((-1,) + (1,) * s.ndim)
    return out


@functools.cache
def _pswf(width: int, upsample_factor: float) -> PSWFKernel:
    """The PSWF kernel without its tap coefficients."""
    c = math.pi * width * (1 - 1 / (2 * upsample_factor)) - 0.05
    eigenvalue, nodes, _, node_values = _pswf_at_nodes(c)
    return PSWFKernel(width, c, float(eigenvalue), nodes, node_values, np.empty((0, 0)))


@functools.cache
def make_pswf_kernel(eps: float, upsample_factor: float, ndim: int) -> PSWFKernel:
    """Build the PSWF kernel and its per-tap Horner coefficients (highest power first)."""
    width = kernel_width(eps, upsample_factor, ndim)
    kernel = _pswf(width, upsample_factor)
    s = np.linspace(0.0, 1.0, 2001)
    exact = np.array(
        [evaluate_pswf(kernel, 2 * (s + j - width / 2) / width) for j in range(width)]
    )
    for degree in range(width, width + _MAX_EXTRA_DEGREE + 1):
        coefficients = _fit_taps(kernel, degree)
        if np.max(np.abs(_horner(coefficients, s) - exact)) <= eps / 10:
            coefficients.setflags(write=False)
            return kernel._replace(coefficients=coefficients)
    raise ValueError(
        "Found invalid value for `make_pswf_kernel(eps, upsample_factor, ndim)`. "
        f"The tap fit did not reach `eps`/10 for eps={eps}, "
        f"upsample_factor={upsample_factor}, ndim={ndim} with degree up to "
        f"{width + _MAX_EXTRA_DEGREE}."
    )


def derivative_coefficients(coefficients: np.ndarray, order: int) -> np.ndarray:
    """The per-tap Horner coefficients (highest power first) of the `order`-th
    derivative in `u`. The offset `s` falls as `u` rises, so each order flips the
    sign."""
    for _ in range(order):
        degree = coefficients.shape[1] - 1
        coefficients = -coefficients[:, :-1] * np.arange(degree, 0, -1)
    return coefficients


def tap_weights(kernel: PSWFKernel, s) -> np.ndarray:
    """Tap weights for fractional offsets `s`, shape `(width, *s.shape)`."""
    return _horner(kernel.coefficients, np.asarray(s, dtype=np.float64))


def kernel_fourier_series(
    kernel: PSWFKernel, n: int, m: int, *, rfft: bool
) -> np.ndarray:
    """`phi_hat(k / m)` for `n` modes in `fftfreq` order, or `0..n // 2` if `rfft`."""
    k = np.arange(n // 2 + 1) if rfft else np.fft.fftfreq(n, 1 / n)
    x = np.clip(math.pi * (k / m) * kernel.width / kernel.shape_parameter, -1, 1)
    return 0.5 * kernel.width * kernel.eigenvalue * evaluate_pswf(kernel, x)


def fine_grid_size(n: int, upsample_factor: float, width: int) -> int:
    """Smallest even 2,3,5-smooth size >= `max(ceil(upsample_factor n), 2 width)`."""
    candidate = max(math.ceil(upsample_factor * n), 2 * width)
    while candidate % 2 or not _is_smooth(candidate):
        candidate += 1
    return candidate


def _is_smooth(x: int) -> bool:
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def resolve_upsample_factor(
    upsample_factor: float | Literal["auto"],
    *,
    n_points: int,
    n_batch: int,
    shape: tuple[int, ...],
    eps: float,
    dtype,
) -> float:
    """`upsample_factor`, or for `"auto"` the cheapest of 1.25, 1.5, 1.75 and 2 with a
    kernel at most 16 wide whose predicted error `max(eps, k eps_mach r_dyn)` meets
    `eps`, else the one of least predicted error. `r_dyn` is the deconvolution's dynamic
    range, `phi_hat(0) / phi_hat(k)` at the corner of the half grid."""
    if isinstance(upsample_factor, str) and upsample_factor == "auto":
        return _auto_upsample_factor(
            n_points, n_batch, tuple(shape), eps, float(jnp.finfo(dtype).eps)
        )
    if isinstance(upsample_factor, str) or not upsample_factor > 1:
        raise ValueError(
            "Found invalid value for `resolve_upsample_factor(upsample_factor, ...)`. "
            '`upsample_factor` must be `"auto"` or greater than 1, got '
            f"upsample_factor={upsample_factor}."
        )
    return upsample_factor


@functools.cache
def _auto_upsample_factor(n_points, n_batch, shape, eps, eps_mach) -> float:
    ndim = len(shape)
    candidates = []
    for sigma in _AUTO_UPSAMPLE_FACTORS:
        try:
            width = kernel_width(eps, sigma, ndim)
        except ValueError:
            continue
        kernel = _pswf(width, sigma)
        fine_shape = tuple(fine_grid_size(n, sigma, width) for n in shape)
        r_dyn = 1.0
        for n, m in zip(shape, fine_shape):
            phi_hat = kernel_fourier_series(kernel, n, m, rfft=True)
            r_dyn *= phi_hat[0] / phi_hat[-1]
        error = max(eps, _ROUNDING_FACTOR * eps_mach * r_dyn)
        size = math.prod(fine_shape)
        cost = n_batch * (
            _SPREAD_COST * n_points * width**ndim + _FFT_COST * size * math.log2(size)
        )
        candidates.append((sigma, cost, error))
    if not candidates:
        # No kernel fits: the narrowest, for the caller's width check to report
        return _AUTO_UPSAMPLE_FACTORS[-1]
    meeting = [(cost, sigma) for sigma, cost, error in candidates if error <= eps]
    if meeting:
        return min(meeting)[1]
    return min(candidates, key=lambda candidate: candidate[2])[0]
