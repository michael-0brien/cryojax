from abc import abstractmethod
from typing import Generic, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex

from ...ndimage import fftn, ifftn, map_coordinates, resize_with_crop_or_pad
from .._image_config import AbstractImageConfig
from .._volume import AbstractVolumeRepresentation, RealVoxelGridVolume


VolRep = TypeVar("VolRep", bound="AbstractVolumeRepresentation")


class AbstractMultisliceIntegrator(eqx.Module, Generic[VolRep], strict=True):
    """Base class for a multislice integration scheme."""

    @abstractmethod
    def integrate(
        self, volume_representation: VolRep, image_config: AbstractImageConfig
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        raise NotImplementedError


class FFTMultisliceIntegrator(
    AbstractMultisliceIntegrator[RealVoxelGridVolume], strict=True
):
    """Multislice integrator that steps using successive FFT-based convolutions."""

    slice_thickness_in_voxels: int

    def __init__(
        self,
        slice_thickness_in_voxels: int = 1,
    ):
        """**Arguments:**

        - `slice_thickness_in_voxels`:
            The number of slices to step through per iteration of the
            rasterized voxel grid.
        - `options_for_interpolation`:
            See `cryojax.image.map_coordinates` for documentation.
        """
        if slice_thickness_in_voxels < 1:
            raise AttributeError(
                "FFTMultisliceIntegrator.slice_thickness_in_voxels must be an "
                "integer greater than or equal to 1."
            )
        self.slice_thickness_in_voxels = slice_thickness_in_voxels

    @override
    def integrate(
        self,
        volume_representation: RealVoxelGridVolume,
        image_config: AbstractImageConfig,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        """Compute the exit wave from an atomic potential using the multislice
        method.

        **Arguments:**

        - `volume_representation`:
            The volume to integrate to the exit plane. This is
            a real-valued voxel grid, which must be in physical units
            of a scattering potential. See rendering method
            `PengAtomPotential.to_real_voxel_grid` for an example.
        - `image_config`:
            The configuration of the imaging instrument.

        **Returns:**

        The wavefunction in the exit plane of the specimen.
        """  # noqa: E501
        # Interpolate volume to new pose at given coordinate system
        z_dim, y_dim, x_dim = volume_representation.shape
        voxel_size = image_config.pixel_size
        real_voxel_grid = _sample_voxel_grid(volume_representation)
        # Initialize multislice geometry
        n_slices = z_dim // self.slice_thickness_in_voxels
        slice_thickness = voxel_size * self.slice_thickness_in_voxels
        # Locally average the potential to be at the given slice thickness.
        # Thow away some slices equal to the remainder
        # `dim % self.slice_thickness_in_voxels`
        if self.slice_thickness_in_voxels > 1:
            potential_per_slice = jnp.sum(
                real_voxel_grid[
                    : z_dim - z_dim % self.slice_thickness_in_voxels, ...
                ].reshape((self.slice_thickness_in_voxels, n_slices, y_dim, x_dim)),
                axis=0,
            )
            # ... take care of remainder
            if z_dim % self.slice_thickness_in_voxels != 0:
                potential_per_slice = jnp.concatenate(
                    (
                        potential_per_slice,
                        real_voxel_grid[
                            z_dim - z_dim % self.slice_thickness_in_voxels :, ...
                        ],
                    )
                )
        else:
            potential_per_slice = real_voxel_grid
        # Compute the integrated potential in a given slice interval, multiplying by
        # the slice thickness (TODO: interpolate for different slice thicknesses?)
        const = image_config.interaction_constant * voxel_size
        compute_object_fn = lambda pot: const * pot
        object_per_slice = jax.vmap(compute_object_fn)(potential_per_slice)
        # Compute the transmission function
        transmission = jnp.exp(1.0j * object_per_slice)
        # Compute the fresnel propagator (TODO: check numerical factors)
        q = image_config.get_frequency_grid(padding=True, physical=True, full=True)
        q_sqr = jnp.sum(q**2, axis=-1)
        fresnel_propagator = jnp.exp(
            1.0j * jnp.pi * image_config.wavelength_in_angstroms * q_sqr * slice_thickness
        )
        # Prepare for iteration. First, initialize plane wave
        plane_wave = jnp.ones((y_dim, x_dim), dtype=complex)
        # ... stepping function
        make_step = lambda n, last_exit_wave: ifftn(
            fftn(transmission[n, :, :] * last_exit_wave) * fresnel_propagator
        )
        # Compute exit wave
        exit_wave = jax.lax.fori_loop(0, n_slices, make_step, plane_wave)
        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != exit_wave.shape:
            exit_wave = resize_with_crop_or_pad(
                exit_wave, image_config.padded_shape, constant_values=1.0 + 0.0j
            )

        return exit_wave


def _sample_voxel_grid(volume: RealVoxelGridVolume):
    # Convert to logical coordinates
    z_dim, y_dim, x_dim = volume.real_voxel_grid.shape
    logical_coordinate_grid = (
        volume.coordinate_grid_in_pixels
        + jnp.asarray((x_dim // 2, y_dim // 2, z_dim // 2))[None, None, None, :]
    )
    # Convert arguments to map_coordinates convention and compute
    x, y, z = jnp.transpose(logical_coordinate_grid, axes=[3, 0, 1, 2])
    return map_coordinates(volume.real_voxel_grid, (z, y, x), order=1, mode="fill")
