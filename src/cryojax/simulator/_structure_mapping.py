import abc
from typing import Any
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from ..internal import error_if_negative
from ._structure_representation import AbstractStructureRepresentation


class AbstractStructureMapping(eqx.Module, strict=True):
    """Abstract interface for a data representation of a protein
    structure.
    """

    @abc.abstractmethod
    def map_to_structure(self) -> AbstractStructureRepresentation:
        raise NotImplementedError


class AbstractStructuralEnsemble(AbstractStructureMapping, strict=True):
    """Abstract interface for a structure with conformational
    heterogeneity.
    """

    conformation: eqx.AbstractVar[Any]


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
        """Map to the structure at `conformation`."""
        funcs = [
            lambda i=i: self.conformational_space[i]
            for i in range(len(self.conformational_space))
        ]
        structure = jax.lax.switch(self.conformation, funcs)

        return structure
