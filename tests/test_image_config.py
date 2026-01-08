import cryojax.simulator as cxs
import pytest


@pytest.mark.parametrize(
    "padded_shape, only_rfft",
    (((5, 5), True), ((10, 10), True), ((10, 10), False)),
)
def test_precompute(padded_shape, only_rfft):
    c = cxs.BasicImageConfig(
        (5, 5),
        pixel_size=1.0,
        voltage_in_kilovolts=300.0,
        pad_options=dict(shape=padded_shape),
        grid_options=dict(precompute=True, only_rfft=only_rfft),
    )
    precomputed_grids = c.precomputed_grids
    assert precomputed_grids is not None
    assert c.get_coordinate_grid(padding=False, physical=False) is precomputed_grids.get(
        real_space=True, padding=False
    )
    assert c.get_frequency_grid(padding=False, physical=False) is precomputed_grids.get(
        real_space=False, padding=False
    )
    assert c.get_coordinate_grid(padding=True, physical=False) is precomputed_grids.get(
        real_space=True, padding=True
    )
    assert c.get_frequency_grid(padding=True, physical=False) is precomputed_grids.get(
        real_space=False, padding=True
    )
    if only_rfft:
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
