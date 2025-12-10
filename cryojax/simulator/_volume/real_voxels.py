"""
Real voxel-based representations of a volume.
"""

import math
from typing import Any, ClassVar, cast
from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import NDArrayLike
from ...ndimage import convert_fftn_to_rfftn, crop_to_shape, irfftn, make_coordinate_grid
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import AbstractVolumeIntegrator, AbstractVoxelVolume, ProjectionArray


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


class AbstractRealVoxelVolume(AbstractVoxelVolume, strict=True):
    """Abstract interface for a voxel-based volume."""

    coordinate_grid_in_pixels: eqx.AbstractVar[Float[Array, "dim dim dim 3"]]

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new volume with a rotated
        `coordinate_grid_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_grid_in_pixels,
            self,
            pose.rotate_coordinates(self.coordinate_grid_in_pixels, inverse=inverse),
        )


class RealVoxelGridVolume(AbstractRealVoxelVolume, strict=True):
    """A 3D voxel grid in real-space."""

    real_voxel_grid: Float[Array, "dim dim dim"]
    coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"]

    def __init__(
        self,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        coordinate_grid_in_pixels: Float[NDArrayLike, "dim dim dim 3"],
    ):
        """**Arguments:**

        - `real_voxel_grid`: The voxel grid in fourier space.
        - `coordinate_grid_in_pixels`: A coordinate grid.
        """
        self.real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        self.coordinate_grid_in_pixels = jnp.asarray(
            coordinate_grid_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `real_voxel_grid`."""
        return cast(tuple[int, int, int], self.real_voxel_grid.shape)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"] | None = None,
        crop_scale: float | None = None,
    ) -> Self:
        """Load a `RealVoxelGridVolume` from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A voxel grid in real space.
        - `crop_scale`: Scale factor at which to crop `real_voxel_grid`.
                        Must be a value greater than `1`.
        """
        # Cast to jax array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Make coordinates if not given
        if coordinate_grid_in_pixels is None:
            # Option for cropping template
            if crop_scale is not None:
                if crop_scale < 1.0:
                    raise ValueError("`crop_scale` must be greater than 1.0")
                cropped_shape = cast(
                    tuple[int, int, int],
                    tuple([int(s / crop_scale) for s in real_voxel_grid.shape[-3:]]),
                )
                real_voxel_grid = crop_to_shape(real_voxel_grid, cropped_shape)
            coordinate_grid_in_pixels = make_coordinate_grid(real_voxel_grid.shape[-3:])

        return cls(real_voxel_grid, coordinate_grid_in_pixels)


class RealVoxelProjection(
    AbstractVolumeIntegrator[RealVoxelGridVolume],
    strict=True,
):
    """Integrate points onto the exit plane using non-uniform FFTs."""

    eps: float
    opts: Any

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(self, *, eps: float = 1e-6, opts: Any = None):
        """**Arguments:**

        - `eps`:
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        - `opts`:
            A `jax_finufft.options.Opts` or `jax_finufft.options.NestedOpts`
            dataclass.
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        """
        if jnufft is None:
            raise RuntimeError(
                "Tried to use the `RealVoxelProjection` "
                "class, but `jax-finufft` is not installed. "
                "See https://github.com/flatironinstitute/jax-finufft "
                "for installation instructions."
            ) from JAX_FINUFFT_IMPORT_ERROR
        self.eps = eps
        self.opts = opts

    @override
    def integrate(
        self,
        volume_representation: RealVoxelGridVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionArray:
        """Integrate the volume at the `AbstractImageConfig` settings
        of a voxel-based representation in real-space, using non-uniform FFTs.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The image configuration.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in Fourier.

        **Returns:**

        The volume projection in real or Fourier space, at the
        `image_config.padded_shape`.
        """
        n_voxels = math.prod(volume_representation.shape)
        fourier_projection = _project_with_nufft(
            volume_representation.real_voxel_grid.ravel(),
            volume_representation.coordinate_grid_in_pixels.reshape((n_voxels, 3)),
            image_config.padded_shape,
            eps=self.eps,
            opts=self.opts,
        )
        # Scale by voxel size for units
        fourier_projection *= image_config.pixel_size
        return (
            irfftn(fourier_projection, s=image_config.padded_shape)
            if outputs_real_space
            else fourier_projection
        )


def _project_with_nufft(weights, coordinate_list, shape, eps=1e-6, opts=None):
    assert jnufft is not None
    weights, coordinate_list = (
        jnp.asarray(weights, dtype=complex),
        jnp.asarray(coordinate_list, dtype=float),
    )
    # Get x and y coordinates
    coordinates_xy = coordinate_list[:, :2]
    # Normalize coordinates betweeen -pi and pi
    ny, nx = shape
    box_xy = jnp.asarray((nx, ny))
    coordinates_periodic = 2 * jnp.pi * coordinates_xy / box_xy
    # Unpack and compute
    x, y = coordinates_periodic[:, 0], coordinates_periodic[:, 1]
    fourier_projection = jnufft.nufft1(shape, weights, y, x, eps=eps, opts=opts, iflag=-1)
    # Shift zero frequency component to corner
    fourier_projection = jnp.fft.ifftshift(fourier_projection)
    # Convert to rfftn output
    return convert_fftn_to_rfftn(fourier_projection, mode="real")
