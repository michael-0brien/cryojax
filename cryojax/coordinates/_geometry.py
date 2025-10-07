import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


def cartesian_to_polar(
    coordinate_or_frequency_grid: Float[Array, "y_dim x_dim 2"], square: bool = False
) -> tuple[Inexact[Array, "y_dim x_dim"], Inexact[Array, "y_dim x_dim"]]:
    """Convert from cartesian to polar coordinates.

    **Arguments:**

    - `coordinate_or_frequency_grid`:
        The cartesian coordinate system in real or fourier space.
    - `square`:
        If `True`, return the square of the radial coordinate
        $|r|^2$. Otherwise, return $|r|$.

    **Returns:**

    A tuple `(r, theta)`, where `r` is the radial coordinate system and
    `theta` is the angular coordinate system. If `square=True`, return a
    tuple `(r_squared, theta)`.
    """
    theta = jnp.arctan2(
        coordinate_or_frequency_grid[..., 0], coordinate_or_frequency_grid[..., 1]
    )
    k_sqr = jnp.sum(jnp.square(coordinate_or_frequency_grid), axis=-1)
    if square:
        return k_sqr, theta
    else:
        kr = jnp.sqrt(k_sqr)
        return kr, theta
