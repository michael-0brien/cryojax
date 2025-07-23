"""
Image normalization routines.
"""

import math
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Inexact


def normalize_image(
    image: Inexact[Array, "y_dim x_dim"],
    *,
    input_is_real_space: bool = True,
    where: Optional[Bool[Array, "y_dim x_dim"]] = None,
    input_is_rfft: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean 0 and standard deviation 1 in real space."""
    return rescale_image(
        image,
        1.0,
        0.0,
        input_is_real_space=input_is_real_space,
        where=where,
        input_is_rfft=input_is_rfft,
        shape_in_real_space=shape_in_real_space,
    )


def rescale_image(
    image: Inexact[Array, "y_dim x_dim"],
    std: float | Float[Array, ""],
    mean: float | Float[Array, ""],
    *,
    input_is_real_space: bool = True,
    where: Optional[Bool[Array, "y_dim x_dim"]] = None,
    input_is_rfft: bool = True,
    shape_in_real_space: Optional[tuple[int, int]] = None,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean `mean`
    and standard deviation `std` in real space.

    Parameters
    ----------
    image : `Array`, shape `(N1, N2)`
        The image in either real or Fourier space.
        If in Fourier space, the zero frequency
        component should be in the corner of the image.
        It should also be on the half space.
    std : `float`
        Intensity standard deviation.
    mean : `float`
        Intensity offset.
    input_is_real_space : `bool`
        If ``True``, the given ``image`` is in real
        space. If ``False``, it is in Fourier space.
    where :
        As in `where` argument in `jax.numpy.std` and
        `jax.numpy.mean`. This argument is ignored if
        `input_is_real_space = False`.

    Returns
    -------
    rescaled_image : `Array`, shape `(N1, N2)`
        Image rescaled to have mean ``mu`` and standard
        deviation ``N``.
    """
    image = jnp.asarray(image)
    # First normalize image to zero mean and unit standard deviation
    if input_is_real_space:
        normalized_image = (image - jnp.mean(image, where=where)) / jnp.std(
            image, where=where
        )
        rescaled_image = std * normalized_image + mean
    else:
        N1, N2 = image.shape
        n_pixels = (
            (
                N1 * (2 * N2 - 1)
                if shape_in_real_space is None
                else math.prod(shape_in_real_space)
            )
            if input_is_rfft
            else N1 * N2
        )
        image_with_zero_mean = image.at[0, 0].set(0.0)
        image_std = (
            jnp.sqrt(
                jnp.sum(jnp.abs(image_with_zero_mean[:, 0]) ** 2)
                + 2 * jnp.sum(jnp.abs(image_with_zero_mean[:, 1:]) ** 2)
            )
            if input_is_rfft
            else jnp.linalg.norm(image_with_zero_mean)
        ) / n_pixels
        normalized_image = image_with_zero_mean / image_std
        rescaled_image = (normalized_image * std).at[0, 0].set(mean * n_pixels)
    return rescaled_image
