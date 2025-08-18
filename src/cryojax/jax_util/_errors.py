"""
Utilities for runtime errors, wrapping `equinox.error_if`.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def error_if_negative(x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    return eqx.error_if(x, x < 0, "A non-negative quantity was found to be negative!")


def error_if_not_positive(x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    return eqx.error_if(
        x, x <= 0, "A positive quantity was found to be negative or zero!"
    )


def error_if_zero(x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    return eqx.error_if(
        x, jnp.isclose(x, 0.0), "A non-zero quantity was found to be zero!"
    )


def error_if_not_fractional(x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    return eqx.error_if(
        x,
        ~jnp.logical_and(x >= 0.0, x <= 1.0),
        "A fractional quantity was found to not be between 0 and 1!",
    )
