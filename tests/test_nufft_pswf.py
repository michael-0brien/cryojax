import equinox as eqx
import numpy as np
import pytest
from cryojax.ndimage._nufft.pswf import PSWF0


# From a standalone build of finufft master d2c1ed2's src/common/pswf.cpp
# (g++ -O0 -ffp-contract=off)
_XS = (0.0, 0.3, -0.55, 0.8, 0.95, 0.999)
_FINUFFT = {
    3.72: (
        1.0,
        0.87640853168171728,
        0.62939051128030676,
        0.34451712435352316,
        0.19404889423631588,
        0.15159537189213187,
    ),
    15.03: (
        1.0,
        0.51871004928765496,
        0.096157414281318271,
        0.0035655385643075313,
        7.3616189683903304e-05,
        4.4444045059050546e-06,
    ),
    30.11: (
        1.0,
        0.25889018504941319,
        0.0079955249977655962,
        8.5208413960958134e-06,
        2.2707106595677278e-09,
        2.4075530579433822e-12,
    ),
    41.84: (
        1.0,
        0.15080996543626898,
        0.0011560407699966479,
        7.8030311216629213e-08,
        7.0960935185898601e-13,
        1.9322390564576006e-16,
    ),
}


@pytest.mark.parametrize("c", sorted(_FINUFFT))
def test_pswf_matches_finufft_build(c):
    np.testing.assert_allclose(PSWF0(c)(np.array(_XS)), _FINUFFT[c], rtol=1e-13, atol=0)


@pytest.mark.parametrize("c", sorted(_FINUFFT))
def test_pswf_support(c):
    pswf = PSWF0(c)
    assert pswf(0.0) == 1.0
    np.testing.assert_array_equal(pswf(np.array([-1.01, 1.01])), 0.0)


@pytest.mark.parametrize("c", sorted(_FINUFFT))
def test_pswf_integral_equation(c):
    """`int cos(c x t) psi(t) dt = lambda psi(x)`, with lambda the integral of psi."""
    pswf = PSWF0(c)
    t, v = np.polynomial.legendre.leggauss(256)
    values = pswf(t)
    eigenvalue = np.sum(v * values)
    x = np.linspace(-1, 1, 501)
    residual = np.cos(c * np.outer(x, t)) @ (v * values) - eigenvalue * pswf(x)
    assert np.max(np.abs(residual)) / eigenvalue < 1e-12


def test_pswf_is_module_and_read_only():
    pswf = PSWF0(15.03)
    assert isinstance(pswf, eqx.Module)
    assert not pswf.legendre_coefficients.flags.writeable
    assert not pswf.recurrence_coefficients.flags.writeable
