"""
Image normalization routines.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, Inexact

from ..jax_util import FloatLike, NDArrayLike


def rescale_image(
    image: Inexact[NDArrayLike, "y_dim x_dim"],
    mean: FloatLike = 0.0,
    std: FloatLike = 1.0,
    *,
    where: Bool[Array, "y_dim x_dim"] | None = None,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean `mean`
    and standard deviation `std` in real space.

    **Parameters:**

    - `image`:
        The image in real-space.
    - `mean`:
        Rescale to this mean.
    - `std`:
        Rescale to this standard deviation.
    - `where`:
        As `where` in `jax.numpy.std` and
        `jax.numpy.mean`. This argument is ignored if
        `input_is_real_space = False`.


    **Returns:**

    Image rescaled to the given mean and standard deviation.
    """
    image = jnp.asarray(image)
    mean, std = jnp.asarray(mean), jnp.asarray(std)
    # First normalize image to zero mean and unit standard deviation
    normalized_image = (image - jnp.mean(image, where=where)) / jnp.std(
        image, where=where
    )
    # Then rescale
    rescaled_image = std * normalized_image + mean

    return rescaled_image


def standardize_image(
    image: Inexact[NDArrayLike, "y_dim x_dim"],
    *,
    where: Bool[Array, "y_dim x_dim"] | None = None,
) -> Inexact[Array, "y_dim x_dim"]:
    """Normalize so that the image is mean 0
    and standard deviation 1 in real space.

    **Parameters:**

    - `image`:
        The image in real-space.
    - `where`:
        As `where` in `jax.numpy.std` and
        `jax.numpy.mean`. This argument is ignored if
        `input_is_real_space = False`.


    **Returns:**

    The standardized image in real-space.
    """
    image = jnp.asarray(image)
    return (image - jnp.mean(image, where=where)) / jnp.std(image, where=where)


def rescale_fft(
    fourier_image: Complex[NDArrayLike, "y_dim x_dim"],
    mean: FloatLike = 0.0,
    std: FloatLike = 1.0,
    *,
    input_is_rfft: bool = True,
    shape_in_real_space: tuple[int, int] | None = None,
) -> Complex[Array, "y_dim x_dim"]:
    """Rescale so that the image is mean `mean`
    and standard deviation `std` in real-space.

    **Parameters:**

    - `fourier_image`:
        The image in the Fourier domain.
        Ensure the zero frequency component is in the corner.
    - `std`:
        The real-space image is rescaled to this standard deviation.
    - `mean`:
        The real-space image is rescaled to this mean.


    **Returns:**

    Image in the Fourier domain rescaled to
    the given mean and standard deviation.
    """
    fourier_image = jnp.asarray(fourier_image)
    n1, n2 = fourier_image.shape
    n_pixels = (
        (
            n1 * (2 * n2 - 1)
            if shape_in_real_space is None
            else math.prod(shape_in_real_space)
        )
        if input_is_rfft
        else n1 * n2
    )
    fourier_image_zero_mean = fourier_image.at[0, 0].set(0.0)
    image_std = (
        jnp.sqrt(
            jnp.sum(jnp.abs(fourier_image_zero_mean[:, 0]) ** 2)
            + 2 * jnp.sum(jnp.abs(fourier_image_zero_mean[:, 1:]) ** 2)
        )
        if input_is_rfft
        else jnp.linalg.norm(fourier_image_zero_mean)
    ) / n_pixels
    normalized_image = fourier_image_zero_mean / image_std
    rescaled_image = (normalized_image * std).at[0, 0].set(mean * n_pixels)

    return rescaled_image


def standardize_fft(
    fourier_image: Complex[NDArrayLike, "y_dim x_dim"],
    *,
    input_is_rfft: bool = True,
    shape_in_real_space: tuple[int, int] | None = None,
) -> Complex[Array, "y_dim x_dim"]:
    """Standardize so that the image is mean 0
    and standard deviation 1 in real-space.

    **Parameters:**

    - `fourier_image`:
        The image in the Fourier domain.
        Ensure the zero frequency component is in the corner.


    **Returns:**

    Standardized image in the Fourier domain.
    """
    return rescale_fft(
        fourier_image,
        mean=0.0,
        std=1.0,
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
