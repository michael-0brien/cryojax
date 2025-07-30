"""
Methods for integrating the volume directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from equinox import AbstractClassVar
from jaxtyping import Array, Complex, Float

from .._image_config import AbstractImageConfig


VolumeT = TypeVar("VolumeT")
VoxelVolumeT = TypeVar("VoxelVolumeT")


class AbstractDirectIntegrator(eqx.Module, Generic[VolumeT], strict=True):
    """Base class for a method of integrating a volume onto
    the exit plane.
    """

    is_projection_approximation: AbstractClassVar[bool]

    @abstractmethod
    def integrate(
        self,
        volume: VolumeT,
        config: AbstractImageConfig,
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
    AbstractDirectIntegrator[VoxelVolumeT], Generic[VoxelVolumeT], strict=True
):
    outputs_integral: eqx.AbstractVar[bool]
