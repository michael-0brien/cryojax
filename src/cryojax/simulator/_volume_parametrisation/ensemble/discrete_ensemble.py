from typing import Optional, Self
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from ....jax_util import error_if_negative
from ..base_parametrisation import (
    AbstractVolumeRepresentation,
)
from .base_ensemble import AbstractConformationalEnsemble


class DiscreteConformationalEnsemble(
    AbstractConformationalEnsemble[
        tuple[AbstractVolumeRepresentation, ...], Int[Array, ""]
    ],
    strict=True,
):
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
        self.conformation = jnp.asarray(error_if_negative(conformation), dtype=int)

    @property
    def n_conformations(self) -> int:
        return len(self.conformational_space)

    @override
    def compute_volume_representation(
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

    @override
    @classmethod
    def sample_conformation(
        cls,
        rng_key: PRNGKeyArray,
        conformational_space: tuple[AbstractVolumeRepresentation, ...],
    ) -> Self:
        n_conformations = len(conformational_space)
        conformation = jax.random.randint(
            rng_key, (), minval=0, maxval=n_conformations - 1
        )
        return cls(conformational_space=conformational_space, conformation=conformation)
