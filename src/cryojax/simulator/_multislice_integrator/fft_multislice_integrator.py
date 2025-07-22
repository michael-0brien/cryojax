from typing import Any
from typing_extensions import override

import jax
import jax.numpy as jnp
from equinox import error_if
from jaxtyping import Array, Complex, Float

from ...coordinates import make_frequency_grid
from ...ndimage import fftn, ifftn, map_coordinates
from .._config import AbstractConfig
from .._potential_representation import AbstractAtomicPotential, RealVoxelGridPotential
from .._scattering_theory import (
    apply_amplitude_contrast_ratio,
    apply_interaction_constant,
)
from .base_multislice_integrator import AbstractMultisliceIntegrator


class FFTMultisliceIntegrator(
    AbstractMultisliceIntegrator[AbstractAtomicPotential | RealVoxelGridPotential],
    strict=True,
):
    """Multislice integrator that steps using successive FFT-based convolutions."""

    slice_thickness_in_voxels: int
    options_for_rasterization: dict[str, Any]
    options_for_interpolation: dict[str, Any]

    def __init__(
        self,
        slice_thickness_in_voxels: int = 1,
        *,
        options_for_rasterization: dict[str, Any] = {},
        options_for_interpolation: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `slice_thickness_in_voxels`:
            The number of slices to step through per iteration of the
            rasterized voxel grid.
        - `options_for_rasterization`:
            See `cryojax.simulator.AbstractAtomicPotential.as_real_voxel_grid`
            for documentation. Ignored if a `RealVoxelGridPotential` is passed.
        - `options_for_interpolation`:
            See `cryojax.image.map_coordinates` for documentation.
            Ignored if an `AbstractAtomicPotential` is passed.
        """
        if slice_thickness_in_voxels < 1:
            raise AttributeError(
                "FFTMultisliceIntegrator.slice_thickness_in_voxels must be an "
                "integer greater than or equal to 1."
            )
        self.slice_thickness_in_voxels = slice_thickness_in_voxels
        self.options_for_interpolation = options_for_interpolation
        self.options_for_rasterization = options_for_rasterization

    @override
    def integrate(
        self,
        potential: AbstractAtomicPotential | RealVoxelGridPotential,
        config: AbstractConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        """Compute the exit wave from an atomic potential using the multislice
        method.

        **Arguments:**

        - `potential`: The atomic potential to project.
        - `config`: The configuration of the imaging instrument.

        **Returns:**

        The wavefunction in the exit plane of the specimen.
        """  # noqa: E501
        # Rasterize a voxel grid at the given settings
        if isinstance(potential, AbstractAtomicPotential):
            z_dim, y_dim, x_dim = (
                min(config.padded_shape),
                *config.padded_shape,
            )
            voxel_size = config.pixel_size
            potential_voxel_grid = potential.as_real_voxel_grid(
                (z_dim, y_dim, x_dim), voxel_size, **self.options_for_rasterization
            )
        else:
            # Interpolate volume to new pose at given coordinate system
            z_dim, y_dim, x_dim = potential.real_voxel_grid.shape
            voxel_size = potential.voxel_size
            potential_voxel_grid = _interpolate_voxel_grid_to_rotated_coordinates(
                potential.real_voxel_grid,
                potential.coordinate_grid_in_pixels,
                **self.options_for_interpolation,
            )
        # Initialize multislice geometry
        n_slices = z_dim // self.slice_thickness_in_voxels
        slice_thickness = voxel_size * self.slice_thickness_in_voxels
        # Locally average the potential to be at the given slice thickness.
        # Thow away some slices equal to the remainder
        # `dim % self.slice_thickness_in_voxels`
        if self.slice_thickness_in_voxels > 1:
            potential_per_slice = jnp.sum(
                potential_voxel_grid[
                    : z_dim - z_dim % self.slice_thickness_in_voxels, ...
                ].reshape((self.slice_thickness_in_voxels, n_slices, y_dim, x_dim)),
                axis=0,
            )
            # ... take care of remainder
            if z_dim % self.slice_thickness_in_voxels != 0:
                potential_per_slice = jnp.concatenate(
                    (
                        potential_per_slice,
                        potential_voxel_grid[
                            z_dim - z_dim % self.slice_thickness_in_voxels :, ...
                        ],
                    )
                )
        else:
            potential_per_slice = potential_voxel_grid
        # Compute the integrated potential in a given slice interval, multiplying by
        # the slice thickness (TODO: interpolate for different slice thicknesses?)
        compute_object_fn = lambda pot: apply_interaction_constant(
            apply_amplitude_contrast_ratio(voxel_size * pot, amplitude_contrast_ratio),
            config.wavelength_in_angstroms,
        )
        object_per_slice = jax.vmap(compute_object_fn)(potential_per_slice)
        # Compute the transmission function
        transmission = jnp.exp(1.0j * object_per_slice)
        # Compute the fresnel propagator (TODO: check numerical factors)
        if isinstance(potential, AbstractAtomicPotential):
            radial_frequency_grid = jnp.sum(
                config.padded_full_frequency_grid_in_angstroms**2,
                axis=-1,
            )
        else:
            radial_frequency_grid = jnp.sum(
                make_frequency_grid((y_dim, x_dim), voxel_size, outputs_rfftfreqs=False)
                ** 2,
                axis=-1,
            )
        fresnel_propagator = jnp.exp(
            1.0j
            * jnp.pi
            * config.wavelength_in_angstroms
            * radial_frequency_grid
            * slice_thickness
        )
        # Prepare for iteration. First, initialize plane wave
        plane_wave = jnp.ones((y_dim, x_dim), dtype=complex)
        # ... stepping function
        make_step = lambda n, last_exit_wave: ifftn(
            fftn(transmission[n, :, :] * last_exit_wave) * fresnel_propagator
        )
        # Compute exit wave
        exit_wave = jax.lax.fori_loop(0, n_slices, make_step, plane_wave)

        return (
            exit_wave
            if isinstance(potential, AbstractAtomicPotential)
            else self._postprocess_exit_wave_for_voxel_potential(
                exit_wave, potential, config
            )
        )

    def _postprocess_exit_wave_for_voxel_potential(
        self,
        exit_wave: Complex[Array, "_ _"],
        potential,
        config: AbstractConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        # Check exit wave is at correct pixel size
        exit_wave = error_if(
            exit_wave,
            ~jnp.isclose(potential.voxel_size, config.pixel_size),
            f"Tried to use {type(self).__name__} with `{type(potential).__name__}."
            "voxel_size != AbstractConfig.pixel_size`. If this is true, then "
            f"`{type(self).__name__}.pixel_rescaling_method` must not be set to "
            f"`None`. Try setting `{type(self).__name__}.pixel_rescaling_method = "
            "'bicubic'`.",
        )
        # Resize the image to match the AbstractConfig.padded_shape
        if config.padded_shape != exit_wave.shape:
            exit_wave = config.crop_or_pad_to_padded_shape(
                exit_wave, constant_values=1.0 + 0.0j
            )

        return exit_wave


def _interpolate_voxel_grid_to_rotated_coordinates(
    real_voxel_grid,
    coordinate_grid_in_pixels,
    **options,
):
    # Convert to logical coordinates
    z_dim, y_dim, x_dim = real_voxel_grid.shape
    logical_coordinate_grid = (
        coordinate_grid_in_pixels
        + jnp.asarray((x_dim // 2, y_dim // 2, z_dim // 2))[None, None, None, :]
    )
    # Convert arguments to map_coordinates convention and compute
    x, y, z = jnp.transpose(logical_coordinate_grid, axes=[3, 0, 1, 2])
    return map_coordinates(real_voxel_grid, (z, y, x), **options)
