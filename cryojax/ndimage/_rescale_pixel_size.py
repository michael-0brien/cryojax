"""
Routines for rescaling image pixel size.
"""

import jax
import jax.numpy as jnp
from jax.image import scale_and_translate
from jaxtyping import Array, Float


def rescale_pixel_size(
    image: Float[Array, "y_dim x_dim"],
    current_pixel_size: Float[Array, ""] | float,
    new_pixel_size: Float[Array, ""] | float,
    method: str = "bicubic",
    antialias: bool = True,
) -> Float[Array, "y_dim x_dim"]:
    """
    Measure an image at a given pixel size using interpolation.

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
