"""Data conversion mix-ins classes."""

import abc

import equinox as eqx
from jaxtyping import Array, Float

from ..internal import NDArrayLike


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
