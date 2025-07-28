"""
Abstractions of ensembles on discrete conformational variables.
"""

from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from ...internal import error_if_negative
from .base_structure import AbstractStructuralEnsemble, AbstractStructureRepresentation


class DiscreteStructuralEnsemble(AbstractStructuralEnsemble, strict=True):
    """Abstraction of an ensemble with discrete conformational
    heterogeneity.
    """

    conformational_space: tuple[AbstractStructureRepresentation, ...]
    conformation: Int[Array, ""]

    def __init__(
        self,
        conformational_space: tuple[AbstractStructureRepresentation, ...],
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
    def map_to_structure(self) -> AbstractStructureRepresentation:
        """Get the scattering potential at configured conformation."""
        funcs = [
            lambda i=i: self.conformational_space[i]
            for i in range(len(self.conformational_space))
        ]
        potential = jax.lax.switch(self.conformation, funcs)

        return potential
