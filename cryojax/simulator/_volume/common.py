"""Shared helpers for volume rendering backends."""

from jaxtyping import Array, Float

from ...ndimage import make_1d_frequency_grid


def make_frequencies_1d(
    shape_u: tuple[int, ...],
    pixel_size_u: Float[Array, ""],
    modeord: int = 0,
):
    return tuple(
        make_1d_frequency_grid(
            s,
            pixel_size_u,
            outputs_rfftfreqs=False,
            fftshifted=(True if modeord == 0 else False),
        )
        for s in shape_u[::-1]
    )
