"""Resampling of images through linear maps."""

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ._coordinates import make_coordinate_grid
from ._nufft import dispatch_nufft1


def nufft_resample(
    image: Inexact[Array, "y_dim x_dim"],
    matrix: Float[Array, "2 2"],
    *,
    outputs_real_space: bool = False,
    outputs_rfft: bool = True,
    eps: float = 1e-6,
    upsampfac: float = 1.25,
) -> (
    Inexact[Array, "y_dim x_dim"]
    | Complex[Array, "y_dim x_dim//2+1"]
    | Complex[Array, "y_dim x_dim"]
):
    """Resample an image through an in-plane linear map `D` with a type-1 non-uniform
    FFT, so that the result is the image `p(D⁻¹x)`. In Fourier space, this is
    `|det D| P(Dᵀk)`.

    **Arguments:**

    - `image`:
        The image in real space, with its origin at the center `N//2`.
    - `matrix`:
        The dimensionless linear map `D`, which acts about the image center.
    - `outputs_real_space`:
        If `True`, return the resampled image in real space. Otherwise, return its FFT.
    - `outputs_rfft`:
        If `True`, return the half plane of the FFT, as for `jax.numpy.fft.rfftn`.
        Otherwise, return the full plane. Requires a real `image`, and is ignored if
        `outputs_real_space = True`.
    - `eps`:
        The tolerance of the non-uniform FFT.
    - `upsampfac`:
        The upsampling factor of the non-uniform FFT's internal grid. See
        [`cryojax.ndimage.dispatch_nufft1`][].

    **Returns:**

    The resampled image in real or Fourier space.
    """
    image = jnp.asarray(image)
    is_real = not jnp.iscomplexobj(image)
    if outputs_rfft and not outputs_real_space and not is_real:
        raise ValueError(
            "Found invalid value for `nufft_resample(..., outputs_rfft=...)`. A "
            "half plane of the FFT requires a real `image`, but got a complex "
            "`image`. Pass `outputs_rfft=False`."
        )
    shape = image.shape
    matrix = jnp.asarray(matrix)
    positions = make_coordinate_grid(shape).reshape(-1, 2) @ matrix.T
    strengths = image.reshape(-1).astype(jnp.result_type(image.dtype, jnp.complex64))
    fourier_image = jnp.abs(_determinant_2x2(matrix)) * dispatch_nufft1(
        positions, strengths, shape, eps=eps, upsampfac=upsampfac
    )
    half_plane = fourier_image[:, : shape[1] // 2 + 1]
    if outputs_real_space:
        return (
            jnp.fft.irfftn(half_plane, s=shape)
            if is_real
            else jnp.fft.ifftn(fourier_image)
        )
    return half_plane if outputs_rfft else fourier_image


def _determinant_2x2(matrix: Float[Array, "2 2"]) -> Float[Array, ""]:
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
