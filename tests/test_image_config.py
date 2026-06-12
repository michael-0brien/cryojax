import math

import cryojax.simulator as cxs
import equinox as eqx
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
