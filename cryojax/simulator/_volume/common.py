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


def nspread_to_eps(n_spread: int) -> float:
    """Inverse of the FINUFFT-style `eps -> nspread` heuristic used by
    `nufftax`/`finufft` (`nspread = ceil(-log10(eps) + 1)`): the precision
    `eps` that a FINUFFT-compatible NUFFT would resolve using a kernel width
    of `n_spread`. Used to translate our own `n_spread` parameter into an
    `eps` for NUFFT backends that only accept a precision, not a kernel
    width, directly.
    """
    return 10.0 ** (1 - n_spread)
