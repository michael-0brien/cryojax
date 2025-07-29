import abc

import equinox as eqx
from jaxtyping import Array, Float

from ...internal import NDArrayLike


class AbstractDiscretizeRealVoxels(eqx.Module, strict=True):
    """Abstract interface that can discretize its data to a real voxel grid."""

    @abc.abstractmethod
    def to_real_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        options: dict = {},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        raise NotImplementedError


class AbstractDiscretizeFourierVoxels(eqx.Module, strict=True):
    """Abstract interface that can discretize its data to a fourier voxel grid."""

    @abc.abstractmethod
    def to_fourier_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        options: dict = {},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        raise NotImplementedError
