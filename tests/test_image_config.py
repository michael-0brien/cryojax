import math

import cryojax.simulator as cxs
import equinox as eqx
import numpy as np
import pytest


def _is_smooth(n: int) -> bool:
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@pytest.mark.parametrize("cls_name", ["BasicImageConfig", "DoseImageConfig"])
def test_pad_scale_one_equals_shape(cls_name):
    cls = getattr(cxs, cls_name)
    shape = (64, 64)
    extra = {"electron_dose": 20.0} if cls_name == "DoseImageConfig" else {}
    cfg = cls(shape, pixel_size=1.0, voltage_in_kilovolts=300.0, pad_scale=1.0, **extra)
    assert cfg.padded_shape == shape


@pytest.mark.parametrize("cls_name", ["BasicImageConfig", "DoseImageConfig"])
@pytest.mark.parametrize("pad_scale", [1.5, 2.0])
def test_pad_scale_greater_than_one(cls_name, pad_scale):
    cls = getattr(cxs, cls_name)
    shape = (64, 64)
    extra = {"electron_dose": 20.0} if cls_name == "DoseImageConfig" else {}
    cfg = cls(
        shape, pixel_size=1.0, voltage_in_kilovolts=300.0, pad_scale=pad_scale, **extra
    )
    for s, p in zip(shape, cfg.padded_shape):
        assert p >= math.ceil(pad_scale * s)
        assert _is_smooth(p)


@pytest.mark.parametrize("cls_name", ["BasicImageConfig", "DoseImageConfig"])
def test_explicit_padded_shape_overrides_pad_scale(cls_name):
    cls = getattr(cxs, cls_name)
    shape, padded_shape = (64, 64), (80, 80)
    extra = {"electron_dose": 20.0} if cls_name == "DoseImageConfig" else {}
    cfg = cls(
        shape,
        pixel_size=1.0,
        voltage_in_kilovolts=300.0,
        padded_shape=padded_shape,
        pad_scale=1.5,
        **extra,
    )
    assert cfg.padded_shape == padded_shape


@pytest.mark.parametrize("cls_name", ["BasicImageConfig", "DoseImageConfig"])
def test_pad_scale_less_than_one_raises(cls_name):
    cls = getattr(cxs, cls_name)
    extra = {"electron_dose": 20.0} if cls_name == "DoseImageConfig" else {}
    with pytest.raises(ValueError, match="pad_scale"):
        cls((64, 64), pixel_size=1.0, voltage_in_kilovolts=300.0, pad_scale=0.5, **extra)


@pytest.mark.parametrize(
    "padded_shape, precompute_mode",
    (
        ((5, 5), "rfft"),
        ((10, 10), "rfft"),
        ((5, 5), "fft"),
        ((10, 10), "fft"),
        ((5, 5), "all"),
        ((10, 10), "all"),
    ),
)
def test_precompute(padded_shape, precompute_mode):
    c = cxs.BasicImageConfig(
        (5, 5),
        pixel_size=1.0,
        voltage_in_kilovolts=300.0,
        padded_shape=padded_shape,
        precompute_mode=precompute_mode,
    )
    precomputed_grids = c.precomputed_grids
    assert precomputed_grids is not None
    # rfftfreq grids
    assert c.get_frequency_grid(padding=False, physical=False) is precomputed_grids.get(
        real_space=False, padding=False
    )
    assert c.get_frequency_grid(padding=True, physical=False) is precomputed_grids.get(
        real_space=False, padding=True
    )
    # coordinate grids
    if precompute_mode == "all":
        assert c.get_coordinate_grid(
            padding=False, physical=False
        ) is precomputed_grids.get(real_space=True, padding=False)
        assert c.get_coordinate_grid(
            padding=True, physical=False
        ) is precomputed_grids.get(real_space=True, padding=True)
    else:
        with pytest.raises(Exception):
            precomputed_grids.get(real_space=True)
        with pytest.raises(Exception):
            precomputed_grids.get(real_space=True, padding=True)
    # fftfreq grids
    if precompute_mode == "rfft":
        with pytest.raises(Exception):
            precomputed_grids.get(real_space=False, full=True)
        with pytest.raises(Exception):
            precomputed_grids.get(real_space=False, full=True, padding=True)
    else:
        assert c.get_frequency_grid(
            padding=True, physical=False, full=True
        ) is precomputed_grids.get(real_space=False, padding=True, full=True)
        assert c.get_frequency_grid(
            padding=False, physical=False, full=True
        ) is precomputed_grids.get(real_space=False, padding=False, full=True)


@pytest.mark.parametrize("shape", [(10, 10), (11, 11), (10, 11), (11, 10)])
def test_pixel_size_gradient_no_nan(shape):
    """Gradient of the radial norm of physical grids w.r.t. pixel_size must be finite."""
    import equinox as eqx
    import jax.numpy as jnp

    def radial_norm_sum(pixel_size):
        cfg = cxs.BasicImageConfig(
            shape, pixel_size=pixel_size, voltage_in_kilovolts=300.0
        )
        coord = cfg.get_coordinate_grid(physical=True)
        freq = cfg.get_frequency_grid(physical=True)
        return jnp.sum(jnp.linalg.norm(coord, axis=-1)) + jnp.sum(
            jnp.linalg.norm(freq, axis=-1)
        )

    ps = jnp.array(1.5)
    grad = eqx.filter_grad(radial_norm_sum)(ps)
    assert jnp.isfinite(grad), f"NaN/Inf gradient for shape {shape}: {grad}"


@pytest.mark.parametrize("astigmatism_in_angstroms", [0.0, 300.0])
def test_ctf_pixel_size_derivatives_no_nan(astigmatism_in_angstroms):
    """The CTF phase must have finite first AND second derivatives w.r.t. the pixel
    size. `arctan2` on the frequency grid made the second derivative NaN at the origin;
    the astigmatic term is now a polynomial in the frequency components."""
    import jax
    import jax.numpy as jnp

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
    import jax.numpy as jnp
    from cryojax.ndimage import make_frequency_grid

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


def test_compile_time_eval():
    c = cxs.BasicImageConfig(
        (5, 5),
        pixel_size=1.0,
        voltage_in_kilovolts=300.0,
        padded_shape=(10, 10),
        precompute_mode="compile_time_eval",
    )

    @eqx.filter_jit
    def _get_coords(_c):
        return _c.get_coordinate_grid()

    @eqx.filter_jit
    def _get_freqs(_c):
        return _c.get_coordinate_grid()

    _ = _get_coords(c)
    _ = _get_freqs(c)
