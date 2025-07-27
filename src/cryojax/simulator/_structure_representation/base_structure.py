"""
Base representations of structures.
"""

import abc
from typing import Any, TypeVar
from typing_extensions import Self, override

import equinox as eqx

from .._pose import AbstractPose


T = TypeVar("T")


class AbstractStructureRepresentation(eqx.Module, strict=True):
    """Abstract interface for a data representation of a protein
    structure.
    """

    @abc.abstractmethod
    def map_to_structure(self) -> "AbstractStructureRepresentation":
        raise NotImplementedError

    @abc.abstractmethod
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new `AbstractStructureRepresentation` at the given pose.

        **Arguments:**

        - `pose`: The pose at which to view the `AbstractStructureRepresentation`.
        """
        raise NotImplementedError


class AbstractHomogeneousStructure(AbstractStructureRepresentation, strict=True):
    """Abstract interface for a structure with no conformational
    heterogeneity.
    """

    @override
    def map_to_structure(self) -> "AbstractHomogeneousStructure":
        return self


class AbstractHeterogeneousStructure(AbstractStructureRepresentation, strict=True):
    """Abstract interface for a structure with conformational
    heterogeneity.
    """

    conformation: eqx.AbstractVar[Any]
