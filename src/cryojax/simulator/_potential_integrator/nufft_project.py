"""
Using non-uniform FFTs for computing volume projections.
"""

import math
from typing import ClassVar, Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...ndimage import convert_fftn_to_rfftn, irfftn
from .._config import AbstractConfig
from .._potential_representation import RealVoxelCloudPotential, RealVoxelGridPotential
from .base_potential_integrator import AbstractVoxelPotentialIntegrator


class NufftProjection(
    AbstractVoxelPotentialIntegrator[RealVoxelGridPotential | RealVoxelCloudPotential],
    strict=True,
):
    """Integrate points onto the exit plane using non-uniform FFTs."""

    pixel_rescaling_mode: Optional[str]
    eps: float

    is_projection_approximation: ClassVar[bool] = True

    def __init__(self, *, pixel_rescaling_mode: Optional[str] = None, eps: float = 1e-6):
        """**Arguments:**

        - `pixel_rescaling_mode`: Method for interpolating the final image to
                                    the `AbstractConfig` pixel size. See
                                    `cryojax.image.rescale_pixel_size` for documentation.
        - `eps`: See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
                 for documentation.
        """
        self.pixel_rescaling_mode = pixel_rescaling_mode
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
        potential: RealVoxelGridPotential | RealVoxelCloudPotential,
        config: AbstractConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ]
        | Float[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        """Compute the integrated scattering potential at the `AbstractConfig` settings
        of a voxel-based representation in real-space, using non-uniform FFTs.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `config`: The configuration of the resulting image.

        **Returns:**

        The projection integral of the `potential` in fourier space, at the
        `config.padded_shape` and the `config.pixel_size`.
        """
        if isinstance(potential, RealVoxelGridPotential):
            shape = potential.shape
            fourier_in_plane_potential = self.project_voxel_cloud_with_nufft(
                potential.real_voxel_grid.ravel(),
                potential.coordinate_grid_in_pixels.reshape((math.prod(shape), 3)),
                config.padded_shape,
            )
        elif isinstance(potential, RealVoxelCloudPotential):
            fourier_in_plane_potential = self.project_voxel_cloud_with_nufft(
                potential.voxel_weights,
                potential.coordinate_list_in_pixels,
                config.padded_shape,
            )
        else:
            raise ValueError(
                "Supported types for `potential` are `RealVoxelGridPotential` and "
                "`RealVoxelCloudPotential`."
            )
        fourier_in_plane_potential = self._postprocess_in_plane_potential(
            fourier_in_plane_potential, potential, config, input_is_rfft=True
        )
        return (
            irfftn(fourier_in_plane_potential, s=config.padded_shape)
            if outputs_real_space
            else fourier_in_plane_potential
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
