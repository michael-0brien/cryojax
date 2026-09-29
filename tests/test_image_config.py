import math

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage import make_coordinate_grid, make_frequency_grid


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


# ── Anisotropic magnification ────────────────────────────────────────────────


def _anisotropy_xy(magnitude, angle_in_degrees):
    theta = 2 * np.deg2rad(angle_in_degrees)
    return np.array([magnitude * np.cos(theta), magnitude * np.sin(theta)])


def _anisotropic_config(anisotropy_xy, pixel_size=1.3, shape=(12, 10)):
    return cxs.AnisotropicImageConfig(
        shape, pixel_size=pixel_size, anisotropy_xy=anisotropy_xy
    )


@pytest.mark.parametrize("cls_name", ["BasicImageConfig", "DoseImageConfig"])
def test_isotropic_configs_have_identity_magnification(cls_name):
    config = getattr(cxs, cls_name)((12, 10), pixel_size=1.3)
    assert not config.is_anisotropic
    np.testing.assert_array_equal(config.anisotropy_matrix, np.eye(2))


@pytest.mark.parametrize("magnitude, angle", [(0.02, 0.0), (0.05, 30.0), (0.1, 125.0)])
def test_anisotropy_matrix_stretches_along_the_anisotropy_angle(magnitude, angle):
    """`D` stretches by `1 + a` along `(sin(angle), cos(angle))`, the direction of
    `AstigmaticCTF.astigmatism_angle`, and compresses by `1 - a` perpendicular to it."""
    config = _anisotropic_config(_anisotropy_xy(magnitude, angle))
    D = np.asarray(config.anisotropy_matrix)
    theta = np.deg2rad(angle)
    u = np.array([np.sin(theta), np.cos(theta)])
    w = np.array([np.cos(theta), -np.sin(theta)])
    assert config.is_anisotropic
    np.testing.assert_allclose(D @ u, (1 + magnitude) * u, atol=1e-12)
    np.testing.assert_allclose(D @ w, (1 - magnitude) * w, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(D), 1 - magnitude**2, atol=1e-12)


def test_anisotropic_grids_are_the_specimen_frame_values():
    """Frequencies are `Dᵀk / p` and coordinates `D⁻¹x p`, which preserve `k·x`."""
    shape, pixel_size = (12, 10), 1.3
    config = _anisotropic_config(_anisotropy_xy(0.05, 30.0), pixel_size, shape)
    D = np.asarray(config.anisotropy_matrix)
    k = np.asarray(make_frequency_grid(shape, outputs_rfftfreqs=False))
    x = np.asarray(make_coordinate_grid(shape))
    frequencies = np.asarray(config.get_frequency_grid(full=True))
    coordinates = np.asarray(config.get_coordinate_grid())
    np.testing.assert_allclose(frequencies, (k / pixel_size) @ D, atol=1e-12)
    np.testing.assert_allclose(
        coordinates, (x * pixel_size) @ np.linalg.inv(D).T, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sum(frequencies * coordinates, axis=-1), np.sum(k * x, axis=-1), atol=1e-12
    )


def test_anisotropy_can_be_excluded_from_the_grids():
    shape, pixel_size = (12, 10), 1.3
    anisotropic = _anisotropic_config(_anisotropy_xy(0.05, 30.0), pixel_size, shape)
    isotropic = cxs.BasicImageConfig(shape, pixel_size=pixel_size)
    np.testing.assert_array_equal(
        anisotropic.get_frequency_grid(anisotropy=False), isotropic.get_frequency_grid()
    )
    np.testing.assert_array_equal(
        anisotropic.get_coordinate_grid(anisotropy=False),
        isotropic.get_coordinate_grid(),
    )


def test_zero_anisotropy_equals_the_isotropic_grids():
    shape, pixel_size = (12, 10), 1.3
    anisotropic = _anisotropic_config((0.0, 0.0), pixel_size, shape)
    isotropic = cxs.BasicImageConfig(shape, pixel_size=pixel_size)
    np.testing.assert_allclose(
        anisotropic.get_frequency_grid(), isotropic.get_frequency_grid(), atol=1e-12
    )
    np.testing.assert_allclose(
        anisotropic.get_coordinate_grid(), isotropic.get_coordinate_grid(), atol=1e-12
    )


@pytest.mark.parametrize("anisotropy_xy", [(0.0, 0.0), (0.03, -0.02)])
def test_anisotropic_grid_norm_gradients_are_finite(anisotropy_xy):
    """Gradients of `|grid|` stay finite at the origin, including at zero anisotropy."""

    def norm_sum(params):
        config = cxs.AnisotropicImageConfig(
            (12, 10), pixel_size=params[0], anisotropy_xy=params[1:]
        )
        return jnp.sum(jnp.linalg.norm(config.get_frequency_grid(), axis=-1)) + jnp.sum(
            jnp.linalg.norm(config.get_coordinate_grid(), axis=-1)
        )

    params = jnp.array([1.3, *anisotropy_xy])
    assert jnp.all(jnp.isfinite(jax.grad(norm_sum)(params)))


@pytest.mark.parametrize("anisotropy_xy", [0.1, (0.1, 0.2, 0.3)])
def test_anisotropic_config_rejects_non_2_vectors(anisotropy_xy):
    with pytest.raises(ValueError, match="anisotropy_xy"):
        _anisotropic_config(anisotropy_xy)


# ── Options ──────────────────────────────────────────────────────────────────


def test_anisotropic_config_accepts_nufft_options():
    options = {"nufft": {"eps": 1e-8, "upsampfac": 2.0}}
    config = _anisotropic_config((0.01, 0.0))
    configured = cxs.AnisotropicImageConfig((12, 10), 1.3, options=options)
    assert config.options == {}
    assert configured.options == options
    # Static, so a jitted function recompiles on new options rather than tracing them
    hash(configured.options)


@pytest.mark.parametrize(
    "cls_name, options",
    [
        ("AnisotropicImageConfig", {"nufft_eps": 1e-8}),
        ("AnisotropicImageConfig", {"nufft": {"tolerance": 1e-8}}),
        ("BasicImageConfig", {"nufft": {"eps": 1e-8}}),
        ("DoseImageConfig", {"nufft": {"eps": 1e-8}}),
    ],
)
def test_image_configs_reject_unsupported_options(cls_name, options):
    with pytest.raises(ValueError, match=f"{cls_name}\\(\\.\\.\\., options=\\.\\.\\.\\)"):
        getattr(cxs, cls_name)((12, 10), pixel_size=1.3, options=options)
