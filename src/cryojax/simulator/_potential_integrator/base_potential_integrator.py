"""
Methods for integrating the scattering potential directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import jax.numpy as jnp
from equinox import AbstractClassVar, AbstractVar, Module, error_if
from jaxtyping import Array, Complex, Float

from ...ndimage import maybe_rescale_pixel_size
from .._config import AbstractConfig
from .._potential_representation import AbstractVoxelPotential


PotentialT = TypeVar("PotentialT")
VoxelPotentialT = TypeVar("VoxelPotentialT", bound="AbstractVoxelPotential")


class AbstractPotentialIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a method of integrating a potential onto
    the exit plane.
    """

    is_projection_approximation: AbstractClassVar[bool]

    @abstractmethod
    def integrate(
        self,
        potential: PotentialT,
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


class AbstractVoxelPotentialIntegrator(
    AbstractPotentialIntegrator[VoxelPotentialT], strict=True
):
    """Base class for a method of integrating a voxel-based potential."""

    pixel_rescaling_mode: AbstractVar[Optional[str]]

    def _postprocess_in_plane_potential(
        self,
        fourier_in_plane_potential: Array,
        potential: AbstractVoxelPotential,
        config: AbstractConfig,
        input_is_rfft: bool,
    ) -> Array:
        """Return the integrated potential in fourier space at the
        `config.pixel_size` and the `config.padded_shape.`
        """
        if self.pixel_rescaling_mode is None:
            fourier_in_plane_potential = error_if(
                potential.voxel_size * fourier_in_plane_potential,
                ~jnp.isclose(potential.voxel_size, config.pixel_size),
                f"Tried to use {type(self).__name__} with `{type(potential).__name__}."
                f"voxel_size != {type(potential).__name__}.pixel_size`. If this is true, "
                f"`{type(self).__name__}.pixel_size_rescaling_method` must not be set to "
                f"`None`. Try setting `{type(self).__name__}.pixel_size_rescaling_method "
                "= 'bicubic'`.",
            )
            return fourier_in_plane_potential
        else:
            fourier_in_plane_potential = maybe_rescale_pixel_size(
                potential.voxel_size * fourier_in_plane_potential,
                potential.voxel_size,
                config.pixel_size,
                input_is_real=False,
                input_is_rfft=input_is_rfft,
                shape_in_real_space=config.padded_shape,
            )
            return fourier_in_plane_potential
