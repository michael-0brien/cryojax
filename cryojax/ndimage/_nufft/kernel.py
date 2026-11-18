"""Host-side PSWF kernel table and accuracy rules for the native NUFFT (numpy), as in
finufft 2.6. Citations are to finufft master `d2c1ed2`."""

import functools
import math
from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np


_NUM_NODES = 256
_NUM_LEGENDRE = 100

# finufft include/finufft_common/constants.h:10, 15, 28-29, 35-36, 46, 50, 54
_MIN_WIDTH = 2
_MAX_WIDTH_SINGLE, _MAX_WIDTH_DOUBLE = 12, 16
_MIN_NC, _MAX_NC = 4, 19
_MIN_AUTO_UPSAMPLE_FACTOR, _MAX_AUTO_UPSAMPLE_FACTOR = 1.25, 2.5
_UPSAMPLE_FACTOR_EDGE_PAD = 1e-3
_NSOFF = 1
_ROUND_FAC = 0.35

# "auto"'s FFT cost per fine-grid point and log2 of the grid size, in spread taps: the
# ratio of the fitted constants of the 11-17 benchmark (RTX PRO 5000)
_PALLAS_FFT_COST = 0.284
_PURE_JAX_FFT_COST = 0.0317


class PSWFKernel(NamedTuple):
    width: int
    shape_parameter: float
    eigenvalue: float
    nodes: np.ndarray
    node_values: np.ndarray
    coefficients: np.ndarray


class NUFFTAccuracyLimits(NamedTuple):
    rounding_floor: float
    width: int
    min_upsample_factor: float


def kernel_width(eps: float, upsample_factor: float, ndim: int, dtype) -> int:
    """Kernel width in fine-grid pixels, from finufft's rule and limits."""
    if not upsample_factor > 1:
        raise ValueError(
            "Found invalid value for `kernel_width(eps, upsample_factor, ndim, dtype)`. "
            f"`upsample_factor` must be greater than 1, got upsample_factor="
            f"{upsample_factor}."
        )
    # finufft src/common/kernel.cpp:72
    width = math.ceil(
        math.log(_tolfac(ndim) / eps) / (math.pi * math.sqrt(1 - 1 / upsample_factor))
        + _NSOFF
    )
    # finufft src/common/kernel.cpp:98, 106
    is_single = jnp.finfo(dtype).dtype != np.float64
    width = max(width, _MIN_WIDTH)
    width = min(width, _MAX_WIDTH_SINGLE if is_single else _MAX_WIDTH_DOUBLE)
    if is_single:
        step = (upsample_factor - 0.975 - 0.125 * ndim) * (1.0 / 0.075)
        width = min(width, 8 + math.floor(max(0.0, step)))
    return width


def _tolfac(ndim: int) -> float:
    # finufft src/common/kernel.cpp:61
    return 0.18 * 1.4 ** (ndim - 1)


def _pswf_at_nodes(c: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """`psi_0` at Gauss-Legendre nodes and its eigenvalue, from the Legendre expansion
    of the prolate differential equation (Bouwkamp; finufft src/common/pswf.cpp:180):
    the integral operator's top eigenvalues coincide to rounding at large `c`, the
    differential operator's do not."""
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


def _poly_fit(kernel: PSWFKernel, tap: int, num: int, dtype: np.dtype) -> np.ndarray:
    """`num` coefficients (highest power first) of tap `tap` in `z = 2s - 1`, in
    `dtype`, the kernel evaluated in float64 (finufft include/finufft/makeplan.hpp:237,
    `poly_fit` in include/finufft_common/kernel.h:19)."""
    w = kernel.width
    t = np.cos((2 * np.arange(num) + 1) * math.pi / (2 * num)).astype(dtype)
    x = (t + dtype.type(2 * tap + 1 - w)) / dtype.type(w)
    newton = evaluate_pswf(kernel, x.astype(np.float64)).astype(dtype)
    for j in range(1, num):
        newton[j:] = (newton[j:] - newton[j - 1 : -1]) / (t[j:] - t[: num - j])
    monomial = np.zeros(num, dtype)
    monomial[0] = newton[0]
    basis = np.ones(1, dtype)
    for j in range(1, num):
        shifted, scaled = np.zeros(j + 1, dtype), np.zeros(j + 1, dtype)
        shifted[1:], scaled[:-1] = basis, -t[j - 1] * basis
        basis = shifted + scaled
        monomial[: j + 1] += newton[j] * basis
    return monomial[::-1]


def _in_offset(coefficients: np.ndarray) -> np.ndarray:
    """Per-tap polynomials in `z = 2s - 1` re-expanded in `s` (highest power first)."""
    coefficients = np.asarray(coefficients, np.float64)
    out = np.zeros_like(coefficients)
    for k in range(coefficients.shape[1]):
        out = 2 * np.roll(out, -1, axis=1) - out
        out[:, -1] += coefficients[:, k]
    return out


def _horner(coefficients: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.zeros(coefficients.shape[:1] + s.shape)
    for k in range(coefficients.shape[1]):
        out = out * s + coefficients[:, k].reshape((-1,) + (1,) * s.ndim)
    return out


@functools.cache
def _pswf(width: int, upsample_factor: float) -> PSWFKernel:
    """The PSWF kernel without its tap coefficients."""
    # finufft src/common/kernel.cpp:126, 143
    c = math.pi * width * (1 - 1 / (2 * upsample_factor)) - 0.05
    eigenvalue, nodes, _, node_values = _pswf_at_nodes(c)
    nodes.setflags(write=False)
    node_values.setflags(write=False)
    return PSWFKernel(width, c, float(eigenvalue), nodes, node_values, np.empty((0, 0)))


def make_pswf_kernel(eps: float, upsample_factor: float, ndim: int, dtype) -> PSWFKernel:
    """Build the PSWF kernel and its per-tap Horner coefficients in the offset `s`
    (highest power first), fitted in `dtype`'s precision as finufft does."""
    return _make_pswf_kernel(eps, upsample_factor, ndim, jnp.finfo(dtype).dtype)


@functools.cache
def _make_pswf_kernel(
    eps: float, upsample_factor: float, ndim: int, dtype: np.dtype
) -> PSWFKernel:
    width = kernel_width(eps, upsample_factor, ndim, dtype)
    kernel = _pswf(width, upsample_factor)
    # finufft include/finufft/makeplan.hpp:201-271, finufft_common/kernel.h:117, 120
    num_fit = min(_MAX_NC, width + 3)
    fitted = np.array([_poly_fit(kernel, j, num_fit, dtype) for j in range(width)])
    is_large = np.abs(fitted) >= dtype.type(eps) * dtype.type(0.05)
    needed = np.where(is_large.any(axis=1), num_fit - np.argmax(is_large, axis=1), 0)
    num = max(_MIN_NC, width - 4, int(needed.max()))
    coefficients = _in_offset(fitted[:, num_fit - num :])
    coefficients.setflags(write=False)
    return kernel._replace(coefficients=coefficients)


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
    kernel: PSWFKernel, n: int, m: int, *, rfft: bool, dtype
) -> np.ndarray:
    """`phi_hat(k / m)` for `n` modes in `fftfreq` order, or `0..n // 2` if `rfft`, by
    quadrature of the fitted taps in `dtype`."""
    dtype = np.dtype(dtype)
    # finufft include/finufft/makeplan.hpp:40, 74-88; its first `q` nodes, on (-w/2, 0)
    half_width = kernel.width / 2.0
    q = int(2 + 3.0 * half_width)
    nodes, weights = np.polynomial.legendre.leggauss(2 * q)
    z = dtype.type(half_width) * nodes[:q].astype(dtype)
    x = z + dtype.type(half_width)
    tap = np.ceil(x).astype(int) - 1
    s = x - tap.astype(dtype)
    values = np.zeros_like(s)
    for coefficient in kernel.coefficients.astype(dtype)[tap].T:
        values = values * s + coefficient
    f = dtype.type(half_width) * weights[:q].astype(dtype) * values
    # Both integrate in grid units (finufft's 1/h); finufft's `(-1)^k` centres the
    # kernel at `m / 2`, native's sits at 0
    k = np.abs(np.arange(n // 2 + 1) if rfft else np.fft.fftfreq(n, 1 / n))
    phase = dtype.type(2 * math.pi) * z * (k.astype(dtype)[:, None] / dtype.type(m))
    return np.sum(dtype.type(2) * f * np.cos(phase), axis=1, dtype=dtype)


def fine_grid_size(n: int, upsample_factor: float, width: int) -> int:
    """Smallest even 2,3,5-smooth size >= `max(ceil(upsample_factor n), 2 width)`
    (finufft include/finufft/makeplan.hpp:19)."""
    candidate = max(math.ceil(upsample_factor * n), 2 * width)
    while candidate % 2 or not _is_smooth(candidate):
        candidate += 1
    return candidate


def _is_smooth(x: int) -> bool:
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def nufft_accuracy_limits(
    shape: tuple[int, ...], *, eps: float, upsample_factor: float, dtype
) -> NUFFTAccuracyLimits:
    """finufft 2.6's accuracy limits for [`cryojax.ndimage.dispatch_nufft1`][] and
    [`cryojax.ndimage.dispatch_nufft2`][] on the native backend.

    **Arguments:**

    - `shape`: The shape of the real-space grid.
    - `eps`: The requested relative accuracy.
    - `upsample_factor`: The ratio of the fine grid to `shape`, greater than 1.
    - `dtype`: The transform's dtype, real or complex, single or double precision.

    **Returns:**

    A `NUFFTAccuracyLimits` of `rounding_floor`, the smallest `eps` reachable on
    `shape` in `dtype`; `width`, the kernel width in fine-grid pixels at `eps` and
    `upsample_factor`; and `min_upsample_factor`, the smallest `upsample_factor`
    reaching `eps` at that width (`inf` if none).
    """
    name = "nufft_accuracy_limits(shape, eps, upsample_factor, dtype)"
    if not eps > 0:
        raise ValueError(
            f"Found invalid value for `{name}`. The `eps` must be positive, but got "
            f"`eps = {eps}`."
        )
    if isinstance(upsample_factor, str) or not upsample_factor > 1:
        raise ValueError(
            f"Found invalid value for `{name}`. The `upsample_factor` must be greater "
            f"than 1, but got `upsample_factor = {upsample_factor}`."
        )
    ndim, dtype = len(shape), jnp.finfo(dtype).dtype
    eps_mach = float(jnp.finfo(dtype).eps)
    width = kernel_width(eps, upsample_factor, ndim, dtype)
    # finufft include/finufft/setpts.hpp:22, 99-103; src/common/kernel.cpp:284
    floor = _ROUND_FAC * eps_mach * math.sqrt(sum(float(n) * n for n in shape))
    return NUFFTAccuracyLimits(
        floor, width, _min_upsample_factor(eps, ndim, width, eps_mach)
    )


def _min_upsample_factor(eps: float, ndim: int, width: int, eps_mach: float) -> float:
    """The smallest `upsample_factor` reaching `eps` at `width`, `inf` if none."""
    # finufft src/common/kernel.cpp:169 (aliasing)
    u = math.log(_tolfac(ndim) / eps) / ((width - _NSOFF) * math.pi)
    if not u < 1.0:
        aliasing = math.inf
    else:
        aliasing = 1.0 if u <= 0.0 else 1.0 / (1.0 - u * u)
    # finufft src/common/kernel.cpp:181 (deconvolution rounding)
    v = 2.0 * math.log(eps / (ndim * eps_mach)) / (math.pi * width)
    if not v > 0.0:
        return math.inf
    q = min(math.sqrt(v), 1.0)
    return max(aliasing, 1.0 / (q * (2.0 - q)))


def resolve_upsample_factor(
    upsample_factor: float | Literal["auto"],
    *,
    n_points: int,
    shape: tuple[int, ...],
    eps: float,
    dtype,
    pallas: bool,
) -> float:
    """`upsample_factor`, or for `"auto"` finufft 2.6's search: per width, the smallest
    factor reaching `eps` in [1.25, 2.5], plus 1e-3; the cheapest by `M w^d +
    c G log2(G)`, `G` the fine grid's size and `c` the Pallas (`pallas`) or pure-JAX
    backend's; 2.5 if none."""
    if isinstance(upsample_factor, str) and upsample_factor == "auto":
        return _auto_upsample_factor(
            n_points, tuple(shape), eps, jnp.finfo(dtype).dtype, bool(pallas)
        )
    if isinstance(upsample_factor, str) or not upsample_factor > 1:
        raise ValueError(
            "Found invalid value for `resolve_upsample_factor(upsample_factor, ...)`. "
            '`upsample_factor` must be `"auto"` or greater than 1, got '
            f"upsample_factor={upsample_factor}."
        )
    return upsample_factor


@functools.cache
def _auto_upsample_factor(n_points, shape, eps, dtype, pallas) -> float:
    # finufft src/heuristics.cpp:87 (best_sigma_for_tol) with our cost; 2.5 if none,
    # include/finufft/setpts.hpp:85
    ndim = len(shape)
    eps_mach = float(jnp.finfo(dtype).eps)
    fft_cost = _PALLAS_FFT_COST if pallas else _PURE_JAX_FFT_COST
    max_width = _MAX_WIDTH_DOUBLE if dtype == np.float64 else _MAX_WIDTH_SINGLE
    best_cost, best_sigma = math.inf, _MAX_AUTO_UPSAMPLE_FACTOR
    for ns in range(_MIN_WIDTH, max_width + 1):
        sigma = max(
            _min_upsample_factor(eps, ndim, ns, eps_mach), _MIN_AUTO_UPSAMPLE_FACTOR
        )
        if sigma > _MAX_AUTO_UPSAMPLE_FACTOR:
            continue
        sigma += _UPSAMPLE_FACTOR_EDGE_PAD
        width = kernel_width(eps, sigma, ndim, dtype)
        size = math.prod(fine_grid_size(n, sigma, width) for n in shape)
        cost = n_points * width**ndim + fft_cost * size * math.log2(size)
        if cost < best_cost:
            best_cost, best_sigma = cost, sigma
    return best_sigma
