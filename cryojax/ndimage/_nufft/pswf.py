"""The order-zero prolate spheroidal wave function `psi_0^c` on [-1, 1], `psi(0) = 1`: a
line-by-line port of finufft's `src/common/pswf.cpp` and `include/finufft_common/pswf.h`
(Apache-2.0, The Simons Foundation; master `d2c1ed2`), which repackage V. Rokhlin's
`legeexps.f` and `prolcrea.f` (DMK repository)."""

import math

import equinox as eqx
import numpy as np


# finufft src/common/pswf.cpp:168 (Legendre matrix size by c / 10)
_SIZES = (
    48,
    64,
    80,
    92,
    106,
    120,
    130,
    144,
    156,
    168,
    178,
    190,
    202,
    214,
    224,
    236,
    248,
    258,
    268,
    280,
)


def _prolcoef(rlam, k, c):  # pswf.cpp:21
    alpha0 = k * (k - 1.0) / ((2.0 * k + 1.0) * (2.0 * k - 1.0))
    beta0 = ((k + 1.0) ** 2 / (2.0 * k + 3.0) + k * k / (2.0 * k - 1.0)) / (2.0 * k + 1.0)
    gamma0 = (k + 1.0) * (k + 2.0) / ((2.0 * k + 1.0) * (2.0 * k + 3.0))
    return -c * c * alpha0, rlam - k * (k + 1.0) - c * c * beta0, -c * c * gamma0


def _prolmatr(a, b, g, n, c, rlam):  # pswf.cpp:35
    for k in range(n // 2 + 2):
        a[k], b[k], g[k] = _prolcoef(rlam, 2 * k, c)
        if k != 0:
            a[k] *= math.sqrt((2 * k + 0.5) / (2 * k - 1.5))
        g[k] *= math.sqrt((2 * k + 0.5) / (2 * k + 2.5))


def _prolql1(n, d, e):  # pswf.cpp:45, eigenvalues of a symmetric tridiagonal (QL)
    if n == 1:
        return
    e[: n - 1] = e[1:n]
    e[n - 1] = 0.0
    for l in range(n):
        j = 0
        while True:
            for m in range(l, n - 1):
                tst1 = abs(d[m]) + abs(d[m + 1])
                if tst1 + abs(e[m]) == tst1:
                    break
            else:
                m = n - 1
            if m == l:
                break
            if j == 30:
                raise RuntimeError("PSWF setup did not converge")
            j += 1
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.sqrt(g * g + 1.0)
            g = d[m] - d[l] + e[l] / (g + math.copysign(r, g))
            s = c = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f, b = s * e[i], c * e[i]
                r = math.sqrt(f * f + g * g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    break
                s, c = f / r, g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
            if r == 0.0:
                break
            d[l] -= p
            e[l] = g
            e[m] = 0.0
        i = l
        while i > 0 and d[i] < d[i - 1]:
            d[i], d[i - 1] = d[i - 1], d[i]
            i -= 1


def _prolfact(a, b, c, n, u, v, w):  # pswf.cpp:101, tridiagonal LU
    for i in range(n - 1):
        d = c[i + 1] / a[i]
        a[i + 1] -= b[i] * d
        u[i] = d
        v[i + 1] = b[i] / a[i + 1]
        w[i + 1] = 1.0 / a[i + 1]
    w[0] = 1.0 / a[0]


def _prolsolv(u, v, w, n, rhs):  # pswf.cpp:115
    for i in range(n - 1):
        rhs[i + 1] -= u[i] * rhs[i]
    for i in range(n - 1, 0, -1):
        rhs[i - 1] -= rhs[i] * v[i]
        rhs[i] *= w[i]
    rhs[0] *= w[0]


def _prolfun0(n, c, eps):  # pswf.cpp:128, top eigenvector by inverse iteration
    delta = 1.0e-8
    h = n // 2
    a, b, g, u, v, w = (np.zeros(h + 2) for _ in range(6))
    _prolmatr(a, b, g, n, c, 0.0)
    _prolql1(h, b, a)
    xk = np.ones(h + 3)
    rlam = -b[h - 1] + delta
    _prolmatr(a, b, g, n, c, rlam)
    _prolfact(b, g, a, h, u, v, w)
    for _ in range(4):
        _prolsolv(u, v, w, h, xk)
        xk[:h] /= math.sqrt(sum(xk[j] * xk[j] for j in range(h)))
    imax = 0
    for i in range(h):
        if abs(xk[i]) > eps:
            imax = i
        xk[i] *= math.sqrt(i * 2 + 0.5)
    return xk[: imax + 1]


def _eval_raw(legendre_coefficients, recurrence_coefficients, x):  # pswf.h:31
    # even Legendre recurrence in x^2
    xsq = x * x
    prev, cur = np.zeros_like(x), np.ones_like(x)
    val = legendre_coefficients[0] * cur
    for i in range(1, legendre_coefficients.size):
        a, b, g = recurrence_coefficients[i]
        prev, cur = cur, cur * (xsq * a - b) - prev * g
        val = val + legendre_coefficients[i] * cur
    return val


class PSWF0(eqx.Module):  # pswf.cpp:180, pswf.h:22
    legendre_coefficients: np.ndarray  # finufft's `workdata`
    recurrence_coefficients: np.ndarray  # `coef`, shape (K, 3)
    normalization: float  # `xv0`

    def __init__(self, c):
        i = int(c / 10)
        n = _SIZES[i] if i < len(_SIZES) else int(c * 3) // 2
        work = _prolfun0(n, c, 1e-16)
        coef = np.zeros((work.size, 3))
        for i in range(1, work.size):
            l = 2 * i - 1.0
            coef[i] = (
                (2 * l - 1) * (2 * l + 1) / (l * (l + 1)),
                ((2 * l + 1) * (l - 1) ** 2 + l * l * (2 * l - 3))
                / (l * (l + 1) * (2 * l - 3)),
                (2 * l + 1) * (l - 1) * (l - 2) / (l * (l + 1) * (2 * l - 3)),
            )
        work.setflags(write=False)
        coef.setflags(write=False)
        self.legendre_coefficients = work
        self.recurrence_coefficients = coef
        self.normalization = float(1.0 / _eval_raw(work, coef, np.zeros(())))

    def __call__(self, x):  # pswf.h:59
        x = np.asarray(x, np.float64)
        raw = _eval_raw(self.legendre_coefficients, self.recurrence_coefficients, x)
        return np.where(np.abs(x) > 1, 0.0, raw * self.normalization)
