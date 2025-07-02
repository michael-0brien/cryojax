"""
Routines for rescaling image pixel size.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.image import scale_and_translate
from jaxtyping import Array, Complex, Float, Inexact

from ._fft import fftn, ifftn, irfftn, rfftn


def rescale_pixel_size(
    image: Float[Array, "y_dim x_dim"],
    current_pixel_size: Float[Array, ""] | float,
    new_pixel_size: Float[Array, ""] | float,
    method: str = "bicubic",
    antialias: bool = True,
) -> Float[Array, "y_dim x_dim"]:
    """
    Measure an image at a given pixel size using interpolation.

    For more detail, see ``cryojax.utils.interpolation.scale``.

    Parameters
    ----------
    image :
        The image to be magnified.
    current_pixel_size :
        The pixel size of the input image.
    new_pixel_size :
        The new pixel size after interpolation.
    method :
        Interpolation method. See ``jax.image.scale_and_translate``
        for documentation.
    antialias :
        Apply an anti-aliasing filter upon downsampling. See
        ``jax.image.scale_and_translate`` for documentation.


    Returns
    -------
    rescaled_image :
        An image with pixels whose size are rescaled by
        ``current_pixel_size / new_pixel_size``.
    """
    # Compute scale factor for pixel size rescaling
    scale_factor = current_pixel_size / new_pixel_size
    # Scaling in both dimensions is the same
    scaling = jnp.asarray([scale_factor, scale_factor])
    # Compute the translation in the jax.image convention that leaves
    # cryojax images untranslated
    N1, N2 = image.shape
    translation = (1 - scaling) * jnp.array([N1 // 2, N2 // 2], dtype=float)
    # Rescale pixel sizes
    rescaled_image = scale_and_translate(
        image,
        image.shape,
        (0, 1),
        scaling,
        translation,
        method,
        antialias=antialias,
        precision=jax.lax.Precision.HIGHEST,
    )

    return rescaled_image


def maybe_rescale_pixel_size(
    real_or_fourier_image: (
        Inexact[Array, "padded_y_dim padded_x_dim"]
        | Complex[Array, "padded_y_dim padded_x_dim//2+1"]
    ),
    current_pixel_size: Float[Array, ""] | float,
    new_pixel_size: Float[Array, ""] | float,
    input_is_real: bool = True,
    input_is_rfft: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
    method: str = "bicubic",
) -> (
    Inexact[Array, "padded_y_dim padded_x_dim"]
    | Complex[Array, "padded_y_dim padded_x_dim//2+1"]
):
    """Rescale the image pixel size using real-space interpolation. Only
    interpolate if the `pixel_size` is not the `current_pixel_size`."""
    if input_is_real:
        rescale_fn = lambda im: rescale_pixel_size(
            im, current_pixel_size, new_pixel_size, method=method
        )
    else:
        if input_is_rfft:
            if shape_in_real_space is None:
                rescale_fn = lambda im: rfftn(
                    rescale_pixel_size(
                        irfftn(im),
                        current_pixel_size,
                        new_pixel_size,
                        method=method,
                    )
                )
            else:
                rescale_fn = lambda im: rfftn(
                    rescale_pixel_size(
                        irfftn(im, s=shape_in_real_space),
                        current_pixel_size,
                        new_pixel_size,
                        method=method,
                    )
                )
        else:
            if shape_in_real_space is None:
                rescale_fn = lambda im: fftn(
                    rescale_pixel_size(
                        ifftn(im),
                        current_pixel_size,
                        new_pixel_size,
                        method=method,
                    )
                )
            else:
                rescale_fn = lambda im: fftn(
                    rescale_pixel_size(
                        ifftn(im, s=shape_in_real_space),
                        current_pixel_size,
                        new_pixel_size,
                        method=method,
                    )
                )
    null_fn = lambda im: im
    return jax.lax.cond(
        jnp.isclose(current_pixel_size, new_pixel_size),
        null_fn,
        rescale_fn,
        real_or_fourier_image,
    )
