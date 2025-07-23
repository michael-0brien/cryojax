"""
Base scattering potential representation.
"""

from abc import abstractmethod
from typing_extensions import Self

from equinox import Module

from .._pose import AbstractPose


class AbstractPotentialRepresentation(Module, strict=True):
    """Abstract interface for the spatial potential energy distribution of a
    scatterer.
    """

    @abstractmethod
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new `AbstractPotentialRepresentation` at the given pose.

        **Arguments:**

        - `pose`: The pose at which to view the `AbstractPotentialRepresentation`.
        """
        raise NotImplementedError
