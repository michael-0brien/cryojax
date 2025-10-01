import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def b_factor_to_variance(
    b_factor: Float[Array | np.ndarray, "..."] | float,
) -> Float[Array, "..."]:
    """From the B-factor of a gaussian, return the corresponding
    value of the variance of the gaussian. This simply does the
    conversion

    $$\\sigma^2 = B / (8 \\pi^2)$$

    The purpose of this function is to make it easy to convert
    between conventions when defining gaussians.

    **Arguments:**

    - `b_factor`:
        The B-factor of the gaussian.

    **Returns:**

    The variance.
    """
    return jnp.asarray(b_factor / (8 * np.pi**2))


def variance_to_b_factor(
    variance: Float[Array | np.ndarray, "..."] | float,
) -> Float[Array, "..."]:
    """From the variance of a gaussian, return the corresponding
    value of the B-factor of the gaussian. This simply does the
    conversion

    $$B = 8 \\pi^2 * \\sigma^2$$

    The purpose of this function is to make it easy to convert
    between conventions when defining gaussians.

    **Arguments:**

    - `variance`:
        The variance of the gaussian.

    **Returns:**

    The B-factor.
    """
    return jnp.asarray((8 * np.pi**2) * variance)
