"""
Base representations of structures.
"""

import abc
from typing import TypeVar
from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ....internal import NDArrayLike
from ..._pose import AbstractPose
from ..base_parametrisation import AbstractVolumeParametrisation


T = TypeVar("T")


class AbstractPointCloudVolume(AbstractVolumeParametrisation, strict=True):
    """Abstract interface for a volume represented as a point-cloud."""

    @abc.abstractmethod
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        raise NotImplementedError


class AbstractVoxelVolume(AbstractVolumeParametrisation, strict=True):
    """Abstract interface for a volume represented with voxels."""

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the voxel array."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_real_voxel_grid(
        cls, real_voxel_grid: Float[NDArrayLike, "dim dim dim"]
    ) -> Self:
        """Load an `AbstractVoxelStructure` from a 3D grid in
        real-space.
        """
        raise NotImplementedError


class AbstractIndependentAtomVolume(AbstractPointCloudVolume, strict=True):
    """A molecular structure representation as independent atoms."""

    atom_positions: eqx.AbstractVar[Float[Array, "n_atoms 3"]]

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new potential with rotated `atom_positions`."""
        return eqx.tree_at(
            lambda d: d.atom_positions,
            self,
            pose.rotate_coordinates(self.atom_positions, inverse=inverse),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `atom_positions`."""
        offset_in_angstroms = pose.offset_in_angstroms
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        return eqx.tree_at(
            lambda d: d.atom_positions, self, self.atom_positions + offset_in_angstroms
        )
