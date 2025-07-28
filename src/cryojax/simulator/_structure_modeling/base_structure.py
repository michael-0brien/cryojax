"""
Base representations of structures.
"""

import abc
from typing import Any, TypeVar
from typing_extensions import Self, override

import equinox as eqx
from jaxtyping import Array, Float

from ...internal import NDArrayLike
from .._pose import AbstractPose


T = TypeVar("T")


#
# Structures and maps to structures
#
class AbstractStructureMapping(eqx.Module, strict=True):
    """Abstract interface for a data representation of a protein
    structure.
    """

    @abc.abstractmethod
    def map_to_structure(self) -> "AbstractStructureRepresentation":
        raise NotImplementedError


class AbstractStructureRepresentation(AbstractStructureMapping, strict=True):
    """Abstract interface for a structure with a coordinate system."""

    @abc.abstractmethod
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        raise NotImplementedError


#
# With and without conformational heterogeneity
#
class AbstractFixedStructure(AbstractStructureRepresentation, strict=True):
    """Abstract interface for a structure with no conformational
    heterogeneity.
    """

    @override
    def map_to_structure(self) -> "AbstractStructureRepresentation":
        return self


class AbstractStructuralEnsemble(AbstractStructureMapping, strict=True):
    """Abstract interface for a structure with conformational
    heterogeneity.
    """

    conformation: eqx.AbstractVar[Any]


#
# Common representations
#
class AbstractPointCloudStructure(AbstractFixedStructure, strict=True):
    """Abstract interface for a structure represented as a point-cloud."""

    @abc.abstractmethod
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        raise NotImplementedError


class AbstractVoxelStructure(AbstractFixedStructure, strict=True):
    """Abstract interface for a voxel-based structure."""

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the voxel array."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
    ) -> Self:
        """Load an `AbstractVoxelStructure` from a 3D grid in
        real-space.
        """
        raise NotImplementedError


#
# Data converting interfaces
#
class AbstractRealVoxelRendering(eqx.Module, strict=True):
    """Abstract interface for real-space voxel rendering."""

    @abc.abstractmethod
    def as_real_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        options: dict = {},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        raise NotImplementedError
