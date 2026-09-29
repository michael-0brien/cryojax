"""Tests for the contrast transfer functions (`AstigmaticCTF`, `AberratedCTF`) and
their use in the transfer theories."""

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.constants import wavelength_from_kilovolts
from cryojax.ndimage import make_frequency_grid


_WAVELENGTH = float(wavelength_from_kilovolts(300.0))


def _polar_to_xy(magnitude, angle_in_degrees, fold):
    theta = fold * np.deg2rad(angle_in_degrees)
    return np.array([magnitude * np.cos(theta), magnitude * np.sin(theta)])


def _astigmatic_ctf():
    return cxs.AstigmaticCTF(
        defocus_in_angstroms=12000.0,
        astigmatism_in_angstroms=400.0,
        astigmatism_angle=35.0,
        spherical_aberration_in_mm=2.7,
    )


def _aberrated_ctf(coma=None, trefoil=None):
    return cxs.AberratedCTF(
        defocus_in_um=1.2,
        astigmatism_xy_in_um=_polar_to_xy(0.04, 35.0, 2),
        spherical_aberration_in_mm=2.7,
        coma_xy_in_um=coma,
        trefoil_xy_in_um=trefoil,
    )


_COMA = _polar_to_xy(1.0, 20.0, 1)
_TREFOIL = _polar_to_xy(0.3, 50.0, 3)


# ── AstigmaticCTF ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("astigmatism_in_angstroms", [0.0, 300.0])
def test_ctf_pixel_size_derivatives_no_nan(astigmatism_in_angstroms):
    """The CTF phase must have finite first AND second derivatives w.r.t. the pixel
    size. `arctan2` on the frequency grid made the second derivative NaN at the origin;
    the astigmatic term is now a polynomial in the frequency components."""
    shape = (16, 16)
    transfer_theory = cxs.ContrastTransferTheory(
        cxs.AstigmaticCTF(
            defocus_in_angstroms=10000.0,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=30.0,
        )
    )
    spectrum = jax.random.normal(
        jax.random.key(0), (shape[0], shape[1] // 2 + 1), dtype=complex
    )

    def image_sum_sq(pixel_size):
        cfg = cxs.BasicImageConfig(
            shape, pixel_size=pixel_size, voltage_in_kilovolts=300.0
        )
        image = jnp.fft.irfftn(transfer_theory.propagate_object(spectrum, cfg), s=shape)
        return jnp.sum(image**2)

    ps = jnp.array(1.5)
    assert jnp.isfinite(jax.grad(image_sum_sq)(ps))
    assert jnp.isfinite(jax.hessian(image_sum_sq)(ps))


def test_ctf_phase_matches_polar_form():
    """The polynomial astigmatic term equals the `cos(2 (azimuth - angle))` form."""
    grid = make_frequency_grid((16, 12), 1.7)
    defocus, astigmatism, angle_in_degrees, wavelength, cs_in_mm = (
        12000.0,
        400.0,
        35.0,
        0.0197,
        2.7,
    )
    k_sqr = jnp.sum(grid**2, axis=-1)
    azimuth = jnp.arctan2(grid[..., 0], grid[..., 1])
    astigmatic_defocus = defocus + 0.5 * astigmatism * jnp.cos(
        2.0 * (azimuth - jnp.deg2rad(angle_in_degrees))
    )
    polar_form = (2 * jnp.pi) * (
        -0.5 * astigmatic_defocus * wavelength * k_sqr
        + 0.25 * (cs_in_mm * 1e7) * wavelength**3 * k_sqr**2
    )
    ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=defocus,
        astigmatism_in_angstroms=astigmatism,
        astigmatism_angle=angle_in_degrees,
        spherical_aberration_in_mm=cs_in_mm,
    )
    polynomial_form = ctf.compute_aberration_phase_shifts(
        grid, wavelength_in_angstroms=wavelength
    )
    np.testing.assert_allclose(polynomial_form, polar_form, rtol=1e-10, atol=1e-10)


def test_astigmatic_ctf_cartesian_properties():
    ctf = _astigmatic_ctf()
    np.testing.assert_allclose(ctf.defocus_in_um, 1.2)
    np.testing.assert_allclose(ctf.astigmatism_xy_in_um, _polar_to_xy(0.04, 35.0, 2))


# ── Parity ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ctf, has_odd",
    [
        (_astigmatic_ctf(), False),
        (_aberrated_ctf(), False),
        (_aberrated_ctf(coma=_COMA), True),
        (_aberrated_ctf(trefoil=_TREFOIL), True),
        (_aberrated_ctf(coma=_COMA, trefoil=_TREFOIL), True),
    ],
)
def test_aberration_phase_shifts_split_by_parity(ctf, has_odd):
    grid = make_frequency_grid((15, 15), 1.3, outputs_rfftfreqs=False)

    def chi(k, parity):
        return ctf.compute_aberration_phase_shifts(k, _WAVELENGTH, parity=parity)

    chi_all, chi_even, chi_odd = (chi(grid, p) for p in ("all", "even", "odd"))
    assert (chi_odd is not None) == has_odd
    np.testing.assert_allclose(chi(-grid, "even"), chi_even, atol=1e-10)
    if has_odd:
        np.testing.assert_allclose(chi(-grid, "odd"), -chi_odd, atol=1e-10)
        np.testing.assert_allclose(chi_all, chi_even + chi_odd, atol=1e-10)
        assert not np.allclose(chi_odd, 0.0)
    else:
        np.testing.assert_allclose(chi_all, chi_even, atol=1e-10)


# ── AberratedCTF ─────────────────────────────────────────────────────────────


def test_aberrated_ctf_from_astigmatic_ctf_properties():
    ctf = _astigmatic_ctf()
    aberrated = cxs.AberratedCTF(
        ctf.defocus_in_um, ctf.astigmatism_xy_in_um, ctf.spherical_aberration_in_mm
    )
    grid = make_frequency_grid((16, 12), 1.7)
    np.testing.assert_allclose(
        aberrated.compute_aberration_phase_shifts(grid, _WAVELENGTH),
        ctf.compute_aberration_phase_shifts(grid, _WAVELENGTH),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("outputs_exp", [False, True])
def test_aberrated_ctf_without_odd_terms_reduces_to_astigmatic_ctf(outputs_exp):
    grid = make_frequency_grid((16, 12), 1.7)
    kwargs = dict(
        amplitude_contrast_ratio=0.07, phase_shift=10.0, outputs_exp=outputs_exp
    )
    expected = _astigmatic_ctf()(grid, _WAVELENGTH, **kwargs)
    actual = _aberrated_ctf()(grid, _WAVELENGTH, **kwargs)
    assert jnp.iscomplexobj(actual) == outputs_exp
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_odd_aberrations_make_the_ctf_complex_and_hermitian():
    grid = make_frequency_grid((15, 15), 1.3, outputs_rfftfreqs=False)
    ctf = _aberrated_ctf(coma=_COMA, trefoil=_TREFOIL)
    values = ctf(grid, _WAVELENGTH, amplitude_contrast_ratio=0.07)
    assert jnp.iscomplexobj(values)
    np.testing.assert_allclose(
        ctf(-grid, _WAVELENGTH, amplitude_contrast_ratio=0.07),
        jnp.conj(values),
        atol=1e-10,
    )


@pytest.mark.parametrize("to_vector", [list, tuple, np.asarray, jnp.asarray])
def test_aberrated_ctf_accepts_sequences_and_arrays(to_vector):
    ctf = cxs.AberratedCTF(
        1.2,
        to_vector([0.03, -0.01]),
        2.7,
        coma_xy_in_um=to_vector([0.5, 0.2]),
        trefoil_xy_in_um=to_vector([0.1, -0.3]),
    )
    expected = cxs.AberratedCTF(
        1.2,
        np.array([0.03, -0.01]),
        2.7,
        coma_xy_in_um=np.array([0.5, 0.2]),
        trefoil_xy_in_um=np.array([0.1, -0.3]),
    )
    grid = make_frequency_grid((16, 12), 1.7)
    np.testing.assert_allclose(
        ctf(grid, _WAVELENGTH), expected(grid, _WAVELENGTH), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "name", ["astigmatism_xy_in_um", "coma_xy_in_um", "trefoil_xy_in_um"]
)
@pytest.mark.parametrize("value", [0.1, (0.1, 0.2, 0.3)])
def test_aberrated_ctf_rejects_non_2_vectors(name, value):
    with pytest.raises(ValueError, match=name):
        cxs.AberratedCTF(**{name: value})


def test_from_polar_coordinates():
    ctf = cxs.AberratedCTF.from_polar_coordinates(
        defocus_in_um=1.2,
        astigmatism_in_um=0.04,
        astigmatism_angle=35.0,
        spherical_aberration_in_mm=2.7,
        coma_in_um=1.0,
        coma_angle=20.0,
        trefoil_in_um=0.3,
        trefoil_angle=50.0,
    )
    np.testing.assert_allclose(ctf.astigmatism_xy_in_um, _polar_to_xy(0.04, 35.0, 2))
    np.testing.assert_allclose(ctf.coma_xy_in_um, _COMA)
    np.testing.assert_allclose(ctf.trefoil_xy_in_um, _TREFOIL)
    without_odd = cxs.AberratedCTF.from_polar_coordinates(
        defocus_in_um=1.2, astigmatism_in_um=0.04, astigmatism_angle=35.0
    )
    assert without_odd.coma_xy_in_um is None and without_odd.trefoil_xy_in_um is None
    grid = make_frequency_grid((16, 12), 1.7)
    np.testing.assert_allclose(
        without_odd.compute_aberration_phase_shifts(grid, _WAVELENGTH),
        _astigmatic_ctf().compute_aberration_phase_shifts(grid, _WAVELENGTH),
        rtol=1e-10,
        atol=1e-10,
    )


def test_aberrated_ctf_derivatives_are_finite():
    """First and second derivatives in every parameter and in the pixel size are
    finite, including at zero astigmatism, coma and trefoil."""
    shape = (16, 16)
    spectrum = jax.random.normal(
        jax.random.key(0), (shape[0], shape[1] // 2 + 1), dtype=complex
    )

    def image_sum_sq(params):
        defocus, astigmatism, cs, coma, trefoil, pixel_size = (
            params[0],
            params[1:3],
            params[3],
            params[4:6],
            params[6:8],
            params[8],
        )
        ctf = cxs.AberratedCTF(defocus, astigmatism, cs, coma, trefoil)
        cfg = cxs.BasicImageConfig(
            shape, pixel_size=pixel_size, voltage_in_kilovolts=300.0
        )
        contrast = cxs.ContrastTransferTheory(ctf).propagate_object(spectrum, cfg)
        return jnp.sum(jnp.fft.irfftn(contrast, s=shape) ** 2)

    params = jnp.array([1.0, 0.0, 0.0, 2.7, 0.0, 0.0, 0.0, 0.0, 1.5])
    assert jnp.all(jnp.isfinite(jax.grad(image_sum_sq)(params)))
    assert jnp.all(jnp.isfinite(jax.hessian(image_sum_sq)(params)))


# ── Transfer theories ────────────────────────────────────────────────────────


def _phase_object(shape):
    return jax.random.normal(jax.random.key(1), shape)


@pytest.mark.parametrize(
    "ctf",
    [
        _astigmatic_ctf(),
        _aberrated_ctf(coma=_COMA),
        _aberrated_ctf(coma=_COMA, trefoil=_TREFOIL),
    ],
)
def test_projection_contrast_is_the_linear_term_of_the_propagated_wave(ctf):
    """Ground truth for the sign of the odd aberrations: the linear term (in the
    phase) of the intensity of a propagated weak-phase exit wave is twice the
    contrast of `ContrastTransferTheory`, with no amplitude contrast."""
    shape = (33, 33)
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    phase = _phase_object(shape)

    def intensity(sigma):
        wave = jnp.fft.fftn(jnp.exp(1.0j * sigma * phase))
        wave = cxs.WaveTransferTheory(ctf).propagate_exit_wave(wave, config)
        return jnp.abs(jnp.fft.ifftn(wave)) ** 2

    _, linear_term = jax.jvp(intensity, (0.0,), (1.0,))
    transfer_theory = cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.0)
    contrast = jnp.fft.irfftn(
        transfer_theory.propagate_object(jnp.fft.rfftn(phase), config), s=shape
    )
    np.testing.assert_allclose(linear_term, 2 * contrast, atol=1e-10)


def test_wave_path_test_is_sensitive_to_the_odd_sign():
    """Flipping the sign of the odd factor must break the agreement above."""
    shape = (33, 33)
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    ctf = _aberrated_ctf(coma=_COMA, trefoil=_TREFOIL)
    phase = _phase_object(shape)

    def intensity(sigma):
        wave = jnp.fft.fftn(jnp.exp(1.0j * sigma * phase))
        wave = cxs.WaveTransferTheory(ctf).propagate_exit_wave(wave, config)
        return jnp.abs(jnp.fft.ifftn(wave)) ** 2

    _, linear_term = jax.jvp(intensity, (0.0,), (1.0,))
    grid = config.get_frequency_grid(padding=True)
    chi_even = ctf.compute_aberration_phase_shifts(grid, _WAVELENGTH, parity="even")
    chi_odd = ctf.compute_aberration_phase_shifts(grid, _WAVELENGTH, parity="odd")
    flipped = jnp.fft.irfftn(
        jnp.fft.rfftn(phase) * jnp.sin(chi_even) * jnp.exp(1.0j * chi_odd), s=shape
    )
    assert not np.allclose(linear_term, 2 * flipped, atol=1e-3)


def _ewald_spectrum(shape):
    key_re, key_im = jax.random.split(jax.random.key(2))
    return jax.random.normal(key_re, shape) + 1.0j * jax.random.normal(key_im, shape)


def test_ewald_sphere_without_odd_aberrations_matches_astigmatic_ctf():
    shape = (16, 16)
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    spectrum = _ewald_spectrum(shape)

    def propagate(ctf):
        return cxs.ContrastTransferTheory(ctf).propagate_object(
            spectrum, config, is_ewald_sphere=True
        )

    np.testing.assert_allclose(
        propagate(_aberrated_ctf()), propagate(_astigmatic_ctf()), atol=1e-10
    )


def test_ewald_sphere_with_odd_aberrations_raises():
    shape = (16, 16)
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    transfer_theory = cxs.ContrastTransferTheory(_aberrated_ctf(coma=_COMA))
    with pytest.raises(NotImplementedError):
        transfer_theory.propagate_object(
            _ewald_spectrum(shape), config, is_ewald_sphere=True
        )


def test_coma_matches_cistem_beam_tilt():
    """cisTEM applies a beam tilt `tau` at azimuth `alpha_tau` to its (real) CTF as
    `exp(+i phi)`, `phi = 2 pi Cs lambda^2 |k|^3 tau cos(alpha - alpha_tau)`
    (`CTF::PhaseShiftGivenBeamTiltAndShift`, without the particle-shift term). The
    real part agrees with `AstigmaticCTF` when cisTEM is given the azimuth
    `arctan2(k_x, k_y)` (see `generate_cistem_references.py`). `AberratedCTF` applies
    `exp(-i chi_odd)`, so cisTEM's tilt is reproduced by a coma of magnitude
    `Cs tau` pointing *against* the tilt."""
    grid = make_frequency_grid((32, 32), 1.2)
    cs_in_mm, tilt_in_mrad, tilt_angle = 2.7, 0.5, 40.0
    astigmatic_ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=15000.0,
        astigmatism_in_angstroms=200.0,
        astigmatism_angle=20.0,
        spherical_aberration_in_mm=cs_in_mm,
    )
    k = jnp.linalg.norm(grid, axis=-1)
    azimuth = jnp.arctan2(grid[..., 0], grid[..., 1])
    phi = (
        2
        * jnp.pi
        * (cs_in_mm * 1e7)
        * _WAVELENGTH**2
        * k**3
        * (tilt_in_mrad * 1e-3)
        * jnp.cos(azimuth - jnp.deg2rad(tilt_angle))
    )
    expected = astigmatic_ctf(grid, _WAVELENGTH, amplitude_contrast_ratio=0.07) * jnp.exp(
        1.0j * phi
    )
    ctf = cxs.AberratedCTF(
        astigmatic_ctf.defocus_in_um,
        astigmatic_ctf.astigmatism_xy_in_um,
        cs_in_mm,
        coma_xy_in_um=-cs_in_mm * tilt_in_mrad * _polar_to_xy(1.0, tilt_angle, 1),
    )
    np.testing.assert_allclose(
        ctf(grid, _WAVELENGTH, amplitude_contrast_ratio=0.07), expected, atol=1e-10
    )
