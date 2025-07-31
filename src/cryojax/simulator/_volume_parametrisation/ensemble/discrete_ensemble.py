from typing import Optional
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from ....internal import error_if_negative
from ..base_parametrisation import (
    AbstractEnsembleParametrisation,
    AbstractVolumeRepresentation,
)


class DiscreteStructuralEnsemble(AbstractEnsembleParametrisation, strict=True):
    """Abstraction of an ensemble with discrete conformational
    heterogeneity.
    """

    conformational_space: tuple[AbstractVolumeRepresentation, ...]
    conformation: Int[Array, ""]

    def __init__(
        self,
        conformational_space: tuple[AbstractVolumeRepresentation, ...],
        conformation: int | Int[Array, ""],
    ):
        """**Arguments:**

        - `conformational_space`: A tuple of `AbstractStructureRepresentation`s.
        - `conformation`: A conformation with a discrete index at which to evaluate
                          the tuple.
        """
        self.conformational_space = conformational_space
        self.conformation = jnp.asarray(error_if_negative(conformation))

    @override
    def to_volume_representation(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> AbstractVolumeRepresentation:
        """Map to the volume at `conformation`.

        **Arguments:**

        - `rng_key`:
            Not used in this implementation, but optionally
            included for other implementations.
        """
        funcs = [
            lambda i=i: self.conformational_space[i]
            for i in range(len(self.conformational_space))
        ]
        volume = jax.lax.switch(self.conformation, funcs)

        return volume
