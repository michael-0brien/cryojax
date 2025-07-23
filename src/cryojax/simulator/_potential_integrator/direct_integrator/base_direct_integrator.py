"""
Methods for integrating the scattering potential directly onto the exit plane.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp
from equinox import AbstractClassVar, error_if
from jaxtyping import Array, Complex, Float

from ..._config import AbstractConfig
from ..._potential_representation import AbstractVoxelPotential
from ..base_potential_integrator import AbstractPotentialIntegrator


PotentialT = TypeVar("PotentialT")
VoxelPotentialT = TypeVar("VoxelPotentialT", bound="AbstractVoxelPotential")


class AbstractDirectIntegrator(
    AbstractPotentialIntegrator, Generic[PotentialT], strict=True
):
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


class AbstractDirectVoxelIntegrator(
    AbstractDirectIntegrator[VoxelPotentialT], strict=True
):
    """Base class for a method of integrating a voxel-based potential."""

    checks_pixel_size: bool

    def _check_pixel_size(
        self,
        fourier_in_plane_potential: Array,
        potential: AbstractVoxelPotential,
        config: AbstractConfig,
    ) -> Array:
        """Check to make sure the voxel and pixel sizes are the same."""
        fourier_in_plane_potential = error_if(
            fourier_in_plane_potential,
            ~jnp.isclose(potential.voxel_size, config.pixel_size),
            f"Tried to use {type(self).__name__} with `{type(potential).__name__}."
            f"voxel_size != {type(config).__name__}.pixel_size`.",
        )
        return fourier_in_plane_potential
