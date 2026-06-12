"""Shared helpers for volume rendering backends."""

import jax.numpy as jnp
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


def make_fftshift_phase(
    shape: tuple[int, ...],
    axes: tuple[int, ...] | None = None,
) -> Array:
    """Build the `(-1)^(k1+k2+...)` sign pattern for the full (non-real) FFT.

    Multiplying `jnp.fft.fftn(x)` by this pattern gives the same result as
    `jnp.fft.fftn(jnp.fft.ifftshift(x))`, so the output has DC at corner
    (modeord=0 convention). Then `jnp.fft.fftshift` moves DC back to center
    if needed for storage.

    Only exact for even-sized dimensions.

    **Arguments:**

    - `shape`: shape of the FFT output array (same as input when s=None).
    - `axes`: axes to include; defaults to all axes.

    **Returns:**

    Broadcastable ±1 array with the same number of dimensions as `shape`.
    """
    ndim = len(shape)
    if axes is None:
        axes = tuple(range(ndim))
    else:
        axes = tuple(ax % ndim for ax in axes)
    phase = jnp.ones(())
    for ax in axes:
        n = shape[ax]
        p = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
        reshape = [1] * ndim
        reshape[ax] = n
        phase = phase * p.reshape(reshape)
    return phase
