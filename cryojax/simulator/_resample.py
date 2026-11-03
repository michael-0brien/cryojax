"""Resampling of an image through a 2x2 linear map, by cubic interpolation."""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ..ndimage import map_coordinates
from ._image_config import AbstractImageConfig, _inverse_2x2


def resample_with_anisotropy_matrix(
    image: Inexact[Array, "y_dim x_dim"],
    image_config: AbstractImageConfig,
    *,
    outputs_rfft: bool,
) -> Complex[Array, "y_dim x_dim//2+1"] | Complex[Array, "y_dim x_dim"]:
    """Given a real-space image `p(x)`, return the FFT of `p(D⁻¹x)` on the same pixel
    grid, for the `image_config.anisotropy_matrix` `D`. Coordinates `x` are in pixels,
    about the center `N//2`.

    1. `p` is interpolated at `D⁻¹x` with a cubic B-spline, without a prefilter
       (`cryojax.ndimage.map_coordinates(..., order=3)`). The image is treated as
       periodic, as it is by the FFT: coordinates wrap modulo the shape.
    2. The result is transformed with `rfftn` or `fftn`.
    3. The spectrum is divided by the response of the cubic B-spline sampled at the
       integers, `(2/3 + cos(2πq_x)/3)(2/3 + cos(2πq_y)/3)`, at `q = Dᵀk` for each
       FFT frequency `k` in cycles per pixel.

    Step 3 undoes the blur of step 1, a circular convolution, so for `D = I` the output
    is the FFT of `p` exactly. Otherwise, the error is aliasing that grows towards
    Nyquist.

    **Arguments:**

    - `image`:
        The image in real space, at `image_config.padded_shape`.
    - `image_config`:
        The image configuration.
    - `outputs_rfft`:
        If `True`, return the half plane of the FFT, as for `jax.numpy.fft.rfftn`.
        Otherwise, return the full plane. Must be `False` for a complex `image`.
    """
    y_dim, x_dim = image.shape
    matrix = image_config.anisotropy_matrix
    # Interpolate at `D⁻¹x`, in index units wrapped into `[0, N)`. The cubic taps at
    # `floor(c) + [-1, 0, 1, 2]` then read from the image wrapped by one pixel below
    # and two above
    coordinate_grid = image_config.get_coordinate_grid(padding=True, physical=False)
    coordinate_grid = coordinate_grid @ _inverse_2x2(matrix).T
    y_index = jnp.mod(coordinate_grid[..., 1] + y_dim // 2, y_dim)
    x_index = jnp.mod(coordinate_grid[..., 0] + x_dim // 2, x_dim)
    resampled = map_coordinates(
        jnp.pad(image, ((1, 2), (1, 2)), mode="wrap"),
        (y_index + 1, x_index + 1),
        order=3,
        mode="promise_in_bounds",
    )
    fourier_resampled = (
        jnp.fft.rfftn(resampled) if outputs_rfft else jnp.fft.fftn(resampled)
    )
    # Divide out the B-spline's response at `Dᵀk`
    frequency_grid = image_config.get_frequency_grid(
        padding=True, physical=False, full=not outputs_rfft
    )
    frequency_grid = frequency_grid @ matrix
    response = _bspline_response(frequency_grid[..., 0]) * _bspline_response(
        frequency_grid[..., 1]
    )
    return fourier_resampled / response


def _bspline_response(frequency: Float[Array, "..."]) -> Float[Array, "..."]:
    """The fourier response of the cubic B-spline sampled at the integers, at
    `frequency` in cycles per pixel."""
    return 2 / 3 + jnp.cos(2 * jnp.pi * frequency) / 3
