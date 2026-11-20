"""Private helpers shared by the simulator's submodules."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def inverse_2x2(matrix: Float[Array, "2 2"]) -> Float[Array, "2 2"]:
    """The inverse of a 2x2 matrix, by its adjugate."""
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    adjugate = jnp.stack(
        [jnp.stack([d, -b], axis=-1), jnp.stack([-c, a], axis=-1)], axis=-2
    )
    return adjugate / (a * d - b * c)[..., None, None]
