"""
Using non-uniform FFTs for computing volume projections.
"""

import math
from typing import Any, ClassVar
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...ndimage import convert_fftn_to_rfftn, irfftn
from .._image_config import AbstractImageConfig
from .._volume import RealVoxelGridVolume
from .base_integrator import AbstractVolumeIntegrator


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


class RealVoxelProjection(
    AbstractVolumeIntegrator[RealVoxelGridVolume],
    strict=True,
):
    """Integrate points onto the exit plane using non-uniform FFTs."""

    eps: float
    opts: Any

    is_projection_approximation: ClassVar[bool] = True

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
    ) -> (
        Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ]
        | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    ):
        """Integrate the volume at the `AbstractImageConfig` settings
        of a voxel-based representation in real-space, using non-uniform FFTs.

        **Arguments:**

        - `volume_representation`: The volume representation.
        - `image_config`: The configuration of the resulting image.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in fourier.

        **Returns:**

        The projection integral of the `volume_representation` in fourier space, at the
        `image_config.padded_shape` and the `image_config.pixel_size`.
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
