"""
Base representations of structures.
"""

import abc
from typing import TypeVar
from typing_extensions import Self, override

from jaxtyping import Float

from ...internal import NDArrayLike
from .._pose import AbstractPose
from .._structure_mapping import AbstractStructureMapping


T = TypeVar("T")


#
# Base structure
#
class AbstractStructureRepresentation(AbstractStructureMapping, strict=True):
    """Abstract interface for a structure with a coordinate system."""

    @abc.abstractmethod
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        raise NotImplementedError

    @override
    def map_to_structure(self) -> "AbstractStructureRepresentation":
        """Return the structure."""
        return self


#
# Core library representations
#
class AbstractPointCloudStructure(AbstractStructureRepresentation, strict=True):
    """Abstract interface for a structure represented as a point-cloud."""

    @abc.abstractmethod
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        raise NotImplementedError


class AbstractVoxelStructure(AbstractStructureRepresentation, strict=True):
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
