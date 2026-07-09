"""
Real voxel-based representations of a volume.
"""

import math
from typing import ClassVar, Literal, Self, cast
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import nufftax
from jaxtyping import Array, Float

from ...jax_util import NDArrayLike
from ...ndimage import convert_fftn_to_rfftn, crop_to_shape, irfftn, make_coordinate_grid
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import AbstractVolumeIntegrator, AbstractVoxelVolume, ProjectionArray
from .common import nspread_to_eps


try:
    import jax_finufft
    from jax_finufft.options import NestedOpts, Opts

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jax_finufft, Opts, NestedOpts = None, None, None
    JAX_FINUFFT_IMPORT_ERROR = err


class AbstractRealVoxelVolume(AbstractVoxelVolume, strict=True):
    """Abstract interface for a voxel-based volume."""


class RealVoxelGridVolume(AbstractRealVoxelVolume, strict=True):
    """A 3D voxel grid in real-space."""

    real_voxel_grid: Float[Array, "dim dim dim"]
    coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"]

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        coordinate_grid_in_pixels: Float[NDArrayLike, "dim dim dim 3"],
    ):
        """**Arguments:**

        - `real_voxel_grid`: The voxel grid in real space.
        - `coordinate_grid_in_pixels`: A coordinate grid.
        """
        self.real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        self.coordinate_grid_in_pixels = jnp.asarray(
            coordinate_grid_in_pixels, dtype=float
        )

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new volume with a rotated
        `coordinate_grid_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_grid_in_pixels,
            self,
            pose.rotate_coordinates(
                self.coordinate_grid_in_pixels, inverse=self.is_frame_rotation
            ),
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


class RealVoxelCloudVolume(AbstractRealVoxelVolume, strict=True):
    """A 3D cloud of voxels in real-space."""

    weights: Float[Array, " N"]
    coordinate_list_in_pixels: Float[Array, "N 3"]
    box_dim: int

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        weights: Float[NDArrayLike, " N"],
        coordinate_list_in_pixels: Float[NDArrayLike, "N 3"],
        box_dim: int,
    ):
        """**Arguments:**

        - `weights`:
            The intensity of each voxel.
        - `coordinate_list_in_pixels`:
            The coordinate for each voxel in pixel units, e.g.

            ```python
            import math
            import cryojax.ndimage as im

            box_dim = 128
            shape = (box_dim, box_dim, box_dim)
            coordinate_grid = im.make_coordinate_grid(shape)
            n_voxels =
            coordinate_list_in_pixels = coordinate_grid.reshape((math.prod(shape), 3))
            ```
        - `box_dim`:
            The box dimension of the original unflattened voxel array.
        """  # noqa: E501
        self.weights = jnp.asarray(weights, dtype=float)
        self.coordinate_list_in_pixels = jnp.asarray(
            coordinate_list_in_pixels, dtype=float
        )
        self.box_dim = box_dim

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new volume with a rotated
        `coordinate_grid_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_list_in_pixels,
            self,
            pose.rotate_coordinates(
                self.coordinate_list_in_pixels, inverse=self.is_frame_rotation
            ),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the unflattened voxel grid."""
        dim = self.box_dim
        return (dim, dim, dim)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"] | None = None,
        crop_scale: float | None = None,
    ) -> Self:
        """Load a `RealVoxelCloudVolume` from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A voxel grid in real space.
        - `crop_scale`: Scale factor at which to crop `real_voxel_grid`.
                        Must be a value greater than `1`.
        """
        # Get voxel grid representation
        voxel_volume = RealVoxelGridVolume.from_real_voxel_grid(
            real_voxel_grid,
            coordinate_grid_in_pixels=coordinate_grid_in_pixels,
            crop_scale=crop_scale,
        )
        shape = voxel_volume.real_voxel_grid.shape
        box_dim = shape[0]
        if not all(box_dim == dim for dim in shape):
            raise ValueError(
                "`RealVoxelCloudVolume.from_real_voxel_grid` only supports "
                "grids with cubic box dimensions, but passed a grid with "
                f"shape {real_voxel_grid.shape}."
            )
        # Convert to cloud of voxels
        n_voxels = math.prod(voxel_volume.real_voxel_grid.shape)
        weights = voxel_volume.real_voxel_grid.ravel()
        coordinate_list_in_pixels = voxel_volume.coordinate_grid_in_pixels.reshape(
            (n_voxels, 3)
        )

        return cls(weights, coordinate_list_in_pixels, box_dim)


class RealVoxelProjection(
    AbstractVolumeIntegrator[RealVoxelCloudVolume],
    strict=True,
):
    """Integrate points onto the exit plane using non-uniform FFTs."""

    n_spread: int
    backend: Literal["jax-finufft", "nufftax"]

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self, *, backend: Literal["jax-finufft", "nufftax"] = "nufftax", n_spread: int = 7
    ):
        """**Arguments:**

        - `backend`:
            The backend for non-uniform FFT computation. This is either
            [`nufftax`](https://github.com/GragasLab/nufftax/tree/custom-kernel-spread)
            for a pure-JAX implementation of the
            [`finufft`](https://finufft.readthedocs.io) algorithm,
            or [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft) for
            calling `finufft` directly via `jax.ffi`.
        - `n_spread`:
            The width (number of grid points, per dimension) of the kernel
            used to spread/interpolate. Controls speed / accuracy tradeoff:
            larger `n_spread` is more accurate but slower. Translated into an
            `eps` precision for the underlying non-uniform FFT implementation
            (see [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)).
        """  # noqa: E501
        if backend not in ["jax-finufft", "nufftax"]:
            raise ValueError(
                "`backend` in `IndependentAtomRenderFn` "
                "must be either 'jax-finufft' or 'nufftax'. Got "
                f"`backend = {backend}`."
            )
        self.backend = backend
        self.n_spread = n_spread

    @override
    def integrate(
        self,
        volume_representation: RealVoxelCloudVolume,
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
        if not isinstance(volume_representation, RealVoxelCloudVolume):
            raise ValueError(
                "Got unsupported type for `volume_representation` in "
                "`RealVoxelProjection.integrate`. Expected `RealVoxelCloudVolume`, "
                f"but got `{volume_representation.__class__.__name__}`."
            )
        fourier_projection = _project_with_nufft(
            volume_representation.weights,
            volume_representation.coordinate_list_in_pixels,
            image_config.padded_shape,
            backend=self.backend,
            eps=nspread_to_eps(self.n_spread),
        )
        # Scale by voxel size for units
        fourier_projection *= image_config.pixel_size
        return (
            irfftn(fourier_projection, s=image_config.padded_shape)
            if outputs_real_space
            else fourier_projection
        )


def _project_with_nufft(
    weights,
    coordinate_list,
    shape,
    backend: Literal["jax-finufft", "nufftax"],
    eps: float,
):
    weights, coordinate_list = (
        jnp.asarray(weights, dtype=complex),
        jnp.asarray(coordinate_list, dtype=float),
    )
    # Get x and y coordinates
    coordinates_xy = coordinate_list[:, :2]
    # Normalize coordinates to [-pi, pi) such that the center voxel (index N//2)
    # maps to theta = 2*pi*(N//2)/N.  For even N this equals pi (same as before);
    # for odd N it equals pi*(1 - 1/N).
    ny, nx = shape
    box_xy = jnp.asarray((nx, ny), dtype=float)
    center_xy = jnp.asarray([nx // 2, ny // 2], dtype=float)
    coordinates_periodic = 2 * jnp.pi * (coordinates_xy + center_xy) / box_xy
    # Unpack and compute
    x, y = coordinates_periodic[:, 0], coordinates_periodic[:, 1]
    if backend == "jax-finufft":
        if jax_finufft is None:
            raise RuntimeError(
                "Tried to use "
                "`RealVoxelProjection(..., backend='jax-finufft')`, "
                "but `jax-finufft` is not installed. "
                "See https://github.com/flatironinstitute/jax-finufft "
                "for installation instructions."
            ) from JAX_FINUFFT_IMPORT_ERROR
        projection_fft = jax_finufft.nufft1(
            shape, weights, y, x, eps=eps, iflag=-1, opts=_make_opts()
        )
    else:
        projection_fft = nufftax.nufft2d1(
            n_modes=shape[::-1], c=weights, x=x, y=y, eps=eps, isign=-1
        )
    # Convert to rfftn output
    return convert_fftn_to_rfftn(jnp.fft.ifftshift(projection_fft), mode="real")


def _make_opts():
    assert NestedOpts is not None
    assert Opts is not None
    return NestedOpts(
        forward=Opts(upsampfac=1.25, gpu_upsampfac=1.25),
        backward=Opts(upsampfac=1.25, gpu_upsampfac=1.25),
    )
