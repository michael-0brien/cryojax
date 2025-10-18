"""
Base representations of volumes.
"""

import abc
from typing import TypeVar
from typing_extensions import Self

from jaxtyping import Float

from ....jax_util import NDArrayLike
from ..._pose import AbstractPose
from ..base_volume import AbstractVolumeRepresentation


T = TypeVar("T")


class AbstractPointCloudVolume(AbstractVolumeRepresentation, strict=True):
    """Abstract interface for a volume represented as a point-cloud."""

    @abc.abstractmethod
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        raise NotImplementedError


class AbstractVoxelVolume(AbstractVolumeRepresentation, strict=True):
    """Abstract interface for a volume represented with voxels.

    !!! info

        If you are using a `volume` in a voxel representation
        pass, the voxel size *must* be passed as the
        `pixel_size` argument, e.g.

        ```python
        import cryojax.simulator as cxs
        from cryojax.io import read_array_from_mrc

        real_voxel_grid, voxel_size = read_array_from_mrc("example.mrc")
        volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
        ...
        config = cxs.BasicImageConfig(shape, pixel_size=voxel_size, ...)
        ```

        If this is not done, the resulting
        image will be incorrect and *not* rescaled to the specified
        to the different pixel size.
    """

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
