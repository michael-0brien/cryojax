"""
Methods for integrating the structure directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from equinox import AbstractClassVar
from jaxtyping import Array, Complex, Float

from .._config import AbstractConfig
from .._structure_modeling import AbstractVoxelStructure


StructureT = TypeVar("StructureT")
VoxelStructureT = TypeVar("VoxelStructureT", bound="AbstractVoxelStructure")


class AbstractDirectIntegrator(eqx.Module, Generic[StructureT], strict=True):
    """Base class for a method of integrating a structure onto
    the exit plane.
    """

    is_projection_approximation: AbstractClassVar[bool]

    @abstractmethod
    def integrate(
        self,
        structure: StructureT,
        config: AbstractConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ]
        | Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
        | Float[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        raise NotImplementedError


class AbstractDirectVoxelIntegrator(
    AbstractDirectIntegrator[StructureT], Generic[StructureT], strict=True
):
    outputs_integral: eqx.AbstractVar[bool]
