from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..._pose import AbstractPose
from ..base_structure import AbstractPointCloudStructure


class AbstractIndependentAtomStructure(AbstractPointCloudStructure, strict=True):
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
