"""
Real voxel-based representations of a volume.
"""

from typing import ClassVar, Self, cast
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import NDArrayLike
from ...ndimage import crop_to_shape, make_coordinate_grid
from .._pose import AbstractPose
from .base_volume import AbstractVoxelVolume


class RealVoxelGridVolume(AbstractVoxelVolume, strict=True):
    """A 3D voxel grid in real-space."""

    values: Float[Array, "dim dim dim"]
    coordinate_grid: Float[Array, "dim dim dim 3"]

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        values: Float[NDArrayLike, "dim dim dim"],
        coordinate_grid: Float[NDArrayLike, "dim dim dim 3"],
    ):
        """**Arguments:**

        - `values`: The voxel grid in real space.
        - `coordinate_grid`: A coordinate grid, in pixel units.
        """
        self.values = jnp.asarray(values, dtype=float)
        self.coordinate_grid = jnp.asarray(coordinate_grid, dtype=float)

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new volume with a rotated `coordinate_grid`."""
        return eqx.tree_at(
            lambda d: d.coordinate_grid,
            self,
            pose.rotate_coordinates(self.coordinate_grid, inverse=self.is_frame_rotation),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the voxel grid."""
        return cast(tuple[int, int, int], self.values.shape)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        /,
        *,
        coordinate_grid: Float[Array, "dim dim dim 3"] | None = None,
        crop_scale: float | None = None,
    ) -> Self:
        """Load a `RealVoxelGridVolume` from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A voxel grid in real space.
        - `coordinate_grid`: A coordinate grid, in pixel units. Built from
                             `real_voxel_grid`'s shape if not given.
        - `crop_scale`: Scale factor at which to crop `real_voxel_grid`.
                        Must be a value greater than `1`.
        """
        # Cast to jax array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Make coordinates if not given
        if coordinate_grid is None:
            # Option for cropping template
            if crop_scale is not None:
                if crop_scale < 1.0:
                    raise ValueError("`crop_scale` must be greater than 1.0")
                cropped_shape = cast(
                    tuple[int, int, int],
                    tuple([int(s / crop_scale) for s in real_voxel_grid.shape[-3:]]),
                )
                real_voxel_grid = crop_to_shape(real_voxel_grid, cropped_shape)
            coordinate_grid = make_coordinate_grid(real_voxel_grid.shape[-3:])

        return cls(real_voxel_grid, coordinate_grid)
