"""
Abstractions of ensembles of biological specimen.
"""

from abc import abstractmethod
from typing_extensions import override

import equinox as eqx

from .._pose import AbstractPose, EulerAnglePose
from .._potential_representation import (
    AbstractAtomicPotential,
    AbstractPotentialRepresentation,
)


class AbstractStructure(eqx.Module, strict=True):
    """A map from a pose to an `AbstractPotentialRepresentation`."""

    pose: eqx.AbstractVar[AbstractPose]

    @abstractmethod
    def get_potential_in_body_frame(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential representation."""
        raise NotImplementedError

    def get_potential_in_transformed_frame(
        self, *, apply_translation: bool = False
    ) -> AbstractPotentialRepresentation:
        """Get the scattering potential, transformed by the pose."""
        potential = self.get_potential_in_body_frame()
        transformed_potential = potential.rotate_to_pose(self.pose)
        if isinstance(transformed_potential, AbstractAtomicPotential):
            if apply_translation:
                transformed_potential = transformed_potential.translate_to_pose(self.pose)
        return transformed_potential


class BasicStructure(AbstractStructure, strict=True):
    """An "ensemble" with one conformation."""

    potential: AbstractPotentialRepresentation
    pose: AbstractPose

    def __init__(self, potential: AbstractPotentialRepresentation, pose: AbstractPose):
        """**Arguments:**

        - `potential`:
            The scattering potential representation of the specimen.
        - `pose`:
            The pose of the specimen.
        """
        self.potential = potential
        self.pose = pose or EulerAnglePose()

    @override
    def get_potential_in_body_frame(self) -> AbstractPotentialRepresentation:
        """Get the scattering potential in the center of mass
        frame.
        """
        return self.potential
