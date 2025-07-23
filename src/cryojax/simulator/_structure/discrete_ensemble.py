"""
Abstractions of ensembles on discrete conformational variables.
"""

from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from ...internal import error_if_negative
from .._pose import AbstractPose, EulerAnglePose
from .._potential_representation import AbstractPotentialRepresentation
from .base_ensemble import AbstractStructuralEnsemble


class DiscreteStructuralEnsemble(AbstractStructuralEnsemble, strict=True):
    """Abstraction of an ensemble with discrete conformational
    heterogeneity.
    """

    conformational_space: tuple[AbstractPotentialRepresentation, ...]
    pose: AbstractPose
    conformation: Int[Array, ""]

    def __init__(
        self,
        conformational_space: tuple[AbstractPotentialRepresentation, ...],
        pose: AbstractPose,
        conformation: int | Int[Array, ""],
    ):
        """**Arguments:**

        - `conformational_space`: A tuple of `AbstractPotential` representations.
        - `pose`: The pose of the specimen.
        - `conformation`: A conformation with a discrete index at which to evaluate
                          the scattering potential tuple.
        """
        self.conformational_space = conformational_space
        self.pose = pose or EulerAnglePose()
        self.conformation = jnp.asarray(error_if_negative(conformation))

    @override
    def get_potential_in_body_frame(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential at configured conformation."""
        funcs = [
            lambda i=i: self.conformational_space[i]
            for i in range(len(self.conformational_space))
        ]
        potential = jax.lax.switch(self.conformation, funcs)

        return potential
