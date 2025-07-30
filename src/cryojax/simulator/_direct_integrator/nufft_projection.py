"""
Using non-uniform FFTs for computing volume projections.
"""

import math
from typing import ClassVar
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...ndimage import convert_fftn_to_rfftn, irfftn
from .._image_config import AbstractImageConfig
from .._structure_parametrisation import RealVoxelGridVolume
from .base_direct_integrator import AbstractDirectVoxelIntegrator


class NufftProjection(
    AbstractDirectVoxelIntegrator[RealVoxelGridVolume],
    strict=True,
):
    """Integrate points onto the exit plane using non-uniform FFTs."""

    eps: float
    outputs_integral: bool

    is_projection_approximation: ClassVar[bool] = True

    def __init__(self, *, outputs_integral: bool = True, eps: float = 1e-6):
        """**Arguments:**

        - `outputs_integral`:
            If `False`, returns a projection. If `True`, return the
            projection multiplied by the voxel size. This is necessary
            for simulating in physical units.
        - `eps`:
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        """
        self.outputs_integral = outputs_integral
        self.eps = eps

    def project_voxel_cloud_with_nufft(
        self,
        weights: Float[Array, " size"],
        coordinate_list_in_angstroms: Float[Array, "size 2"] | Float[Array, "size 3"],
        shape: tuple[int, int],
    ) -> Complex[Array, "{shape[0]} {shape[1]//2+1}"]:
        """Project and interpolate 3D volume point cloud
        onto imaging plane using a non-uniform FFT.

        **Arguments:**

        - `weights`:
            Density point cloud.
        - `coordinate_list_in_angstroms`:
            Coordinate system of point cloud.
        - `shape`:
            Shape of the real-space imaging plane in pixels.

        **Returns:**

        The fourier-space projection of the density point cloud defined by `weights` and
        `coordinate_list_in_angstroms`.
        """
        return _project_with_nufft(weights, coordinate_list_in_angstroms, shape, self.eps)

    @override
    def integrate(
        self,
        volume: RealVoxelGridVolume,
        config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ]
        | Float[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        """Integrate the volume at the `AbstractImageConfig` settings
        of a voxel-based representation in real-space, using non-uniform FFTs.

        **Arguments:**

        - `volume`: The volume representation.
        - `config`: The configuration of the resulting image.

        **Returns:**

        The projection integral of the `volume` in fourier space, at the
        `config.padded_shape` and the `config.pixel_size`.
        """
        if isinstance(volume, RealVoxelGridVolume):
            shape = volume.shape
            fourier_projection = self.project_voxel_cloud_with_nufft(
                volume.real_voxel_grid.ravel(),
                volume.coordinate_grid_in_pixels.reshape((math.prod(shape), 3)),
                config.padded_shape,
            )
        else:
            raise ValueError("Supported type for `volume` is `RealVoxelGridVolume`")
        if self.outputs_integral:
            # Scale by voxel size to convert from projection to integral
            fourier_projection *= config.pixel_size
        return (
            irfftn(fourier_projection, s=config.padded_shape)
            if outputs_real_space
            else fourier_projection
        )


def _project_with_nufft(weights, coordinate_list, shape, eps=1e-6):
    from jax_finufft import nufft1

    weights, coordinate_list = (
        jnp.asarray(weights).astype(complex),
        jnp.asarray(coordinate_list),
    )
    # Get x and y coordinates
    coordinates_xy = coordinate_list[:, :2]
    # Normalize coordinates betweeen -pi and pi
    ny, nx = shape
    box_xy = jnp.asarray((nx, ny), dtype=float)
    coordinates_periodic = 2 * jnp.pi * coordinates_xy / box_xy
    # Unpack and compute
    x, y = coordinates_periodic[:, 0], coordinates_periodic[:, 1]
    fourier_projection = nufft1(shape, weights, y, x, eps=eps, iflag=-1)
    # Shift zero frequency component to corner
    fourier_projection = jnp.fft.ifftshift(fourier_projection)
    # Convert to rfftn output
    return convert_fftn_to_rfftn(fourier_projection, mode="real")
