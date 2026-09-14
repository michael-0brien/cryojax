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
    *,
    weights: (
        Float[Array, "y_dim x_dim"]
        | Float[Array, "z_dim y_dim x_dim"]
        | Float[Array, " x_dim"]
        | None
    ) = None,
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
    - `weights`:
        Optional weight for each element of `image` in the average. Must
        broadcast against `image`. Used, for example, to weight the modes of
        an `rfftn` array by their Hermitian multiplicity (see
        `cryojax.ndimage.make_rfftn_multiplicity`). If `None`, every element
        is weighted equally.

    **Returns:**

    The binned radial averaged of `image` in bins `bins`.
    """
    # Discretize the radial grid
    digitized_radial_grid = jnp.digitize(radial_coordinate_grid, bins, right=True)
    # Compute the radial profile as the (weighted) average value of the image
    # in each bin
    if weights is None:
        numerator = jnp.bincount(
            digitized_radial_grid.ravel(), weights=image.ravel(), length=bins.size
        )
        denominator = jnp.bincount(digitized_radial_grid.ravel(), length=bins.size)
    else:
        weights = jnp.broadcast_to(weights, image.shape)
        numerator = jnp.bincount(
            digitized_radial_grid.ravel(),
            weights=(image * weights).ravel(),
            length=bins.size,
        )
        denominator = jnp.bincount(
            digitized_radial_grid.ravel(), weights=weights.ravel(), length=bins.size
        )

    return numerator / denominator


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
