"""
Routines to compute radial averages of images.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


def compute_binned_radial_average(
    image: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    radial_coordinate_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    bins: Float[Array, " n_bins"],
) -> Inexact[Array, " n_bins"]:
    """Average vectors $\\mathbf{r}$ of constant radius $|\\mathbf{r}|$ into
    discrete bins.

    **Arguments:**

    - `image`:
        Two-dimensional image or three-dimensional volume.
    - `radial_coordinate_grid`:
        Radial coordinate system of image or volume.
    - `bins`:
        Radial bins for averaging.

    **Returns:**

    The binned radial averaged of `image` in bins `bins`.
    """
    # Discretize the radial grid
    digitized_radial_grid = jnp.digitize(radial_coordinate_grid, bins, right=True)
    # Compute the radial profile as the average value of the image in each bin
    binned_radial_average = jnp.bincount(
        digitized_radial_grid.ravel(),
        weights=image.ravel(),
        length=bins.size,
    ) / jnp.bincount(digitized_radial_grid.ravel(), length=bins.size)

    return binned_radial_average


def radial_average_to_grid(
    binned_radial_average: Inexact[Array, " n_bins"],
    bins: Float[Array, " n_bins"],
    radial_coordinate_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    interpolation_mode: str = "linear",
) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
    """Interpolate a binned radially averaged profile onto a grid.

    **Arguments:**

    - `binned_radial_average`:
        The binned, radially averaged profile.
    - `bins`:
        Radial bins over which `binned_radial_average` is computed.
    - `radial_coordinate_grid`:
        Radial coordinate system of image or volume.
    - `interpolation_mode`:
        If `"linear"`, evaluate the grid using linear
        interpolation. If `"nearest"`, use nearest-neighbor
        interpolation.

    **Returns:**

    The `binned_radial_average` evaluated on the `radial_coordinate_grid`.
    """
    if interpolation_mode == "nearest":
        radial_average_on_grid = jnp.take(
            binned_radial_average,
            jnp.digitize(radial_coordinate_grid, bins, right=True),
            mode="clip",
        )
    elif interpolation_mode == "linear":
        radial_average_on_grid = jnp.interp(
            radial_coordinate_grid.ravel(),
            bins,
            binned_radial_average,
        ).reshape(radial_coordinate_grid.shape)
    else:
        raise ValueError(
            f"`interpolation_mode` = {interpolation_mode} not supported. Supported "
            "interpolation modes are 'nearest' or 'linear'."
        )
    return radial_average_on_grid
