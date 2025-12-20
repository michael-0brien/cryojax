"""
Image normalization routines.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Inexact

from ..jax_util import FloatLike, NDArrayLike


def rescale_image(
    image: Inexact[NDArrayLike, "y_dim x_dim"],
    mean: FloatLike = 0.0,
    std: FloatLike = 1.0,
    *,
    input_is_real_space: bool = True,
    where: Bool[Array, "y_dim x_dim"] | None = None,
    input_is_rfft: bool = True,
    shape_in_real_space: tuple[int, int] | None = None,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean `mean`
    and standard deviation `std` in real space.

    **Parameters:**

    - `image`:
        The image in either the real or fourier domain.
        If in fourier, pass `input_is_real_space = True`
        and ensure the zero frequency
        component is in the corner.
    - `std`:
        Rescale to this standard deviation.
    - `mean`:
        Rescale to this mean.
    - `input_is_real_space`:
        If `True`, the given `image` is in real
        space. If `False`, it is in Fourier space.
    - `where`:
        As `where` in `jax.numpy.std` and
        `jax.numpy.mean`. This argument is ignored if
        `input_is_real_space = False`.


    **Returns:**

    Image rescaled to the given mean and standard deviation.
    """
    if where is not None and not input_is_real_space:
        raise ValueError(
            "`cryojax.ndimage.standardize_image` does "
            "not support argument `where` if `input_is_real_space` "
            "is `False`."
        )
    image, mean, std = jnp.asarray(image), jnp.asarray(mean), jnp.asarray(std)
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


def standardize_image(
    image: Inexact[NDArrayLike, "y_dim x_dim"],
    *,
    input_is_real_space: bool = True,
    where: Bool[Array, "y_dim x_dim"] | None = None,
    input_is_rfft: bool = True,
    shape_in_real_space: tuple[int, int] | None = None,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean 0
    and standard deviation 1 in real space.

    **Parameters:**

    - `image`:
        The image in either the real or fourier domain.
        If in fourier, pass `input_is_real_space = True`
        and ensure the zero frequency
        component is in the corner.
    - `input_is_real_space`:
        If `True`, the given `image` is in real
        space. If `False`, it is in Fourier space.
    - `where`:
        As `where` in `jax.numpy.std` and
        `jax.numpy.mean`. This argument is ignored if
        `input_is_real_space = False`.


    **Returns:**

    The standardized image
    """
    return rescale_image(
        image,
        mean=0.0,
        std=1.0,
        input_is_real_space=input_is_real_space,
        where=where,
        input_is_rfft=input_is_rfft,
        shape_in_real_space=shape_in_real_space,
    )


def background_subtract_image(image: Float[NDArrayLike, "y_dim x_dim"]):
    """Ensure an image is on a background with mode equal to zero
    by subtracting the mean value on its outer edge.

    Assumes the signal in the image has sufficiently decayed out
    toward the edges.

    **Arguments:**

    - `image`:
        The image to be background subtracted.

    **Returns:**

    The background subtracted image.
    """
    return jnp.asarray(image) - compute_edge_value(image)


def compute_edge_value(image: Float[NDArrayLike, "y_dim x_dim"]):
    """Compute the median of the values at the image edges.
    Useful for background subtraction.

    **Arguments:**

    - `image`:
        The image to retrieve the edge value of.

    **Returns:**

    The median edge value.
    """
    image = jnp.asarray(image)
    edge_values = jnp.concatenate(
        (image[0, :], image[-1, :], image[1:-1, 0], image[1:-1, -1]), axis=0
    )
    return jnp.median(edge_values)
