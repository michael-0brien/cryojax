"""
Using the fourier slice theorem for computing volume projections.
"""

from typing import ClassVar, Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...ndimage import (
    convert_fftn_to_rfftn,
    fftn,
    ifftn,
    irfftn,
    map_coordinates,
    map_coordinates_with_cubic_spline,
    rfftn,
)
from ...ndimage.transforms import InverseSincMask
from .._config import AbstractConfig
from .._potential_representation import (
    FourierVoxelGridPotential,
    FourierVoxelSplinePotential,
)
from .base_potential_integrator import AbstractVoxelPotentialIntegrator


class FourierSliceExtraction(AbstractVoxelPotentialIntegrator, strict=True):
    """Integrate points to the exit plane using the Fourier projection-slice theorem.

    This extracts slices using interpolation methods housed in
    `cryojax.image.map_coordinates` and `cryojax.image.map_coordinates_with_cubic_spline`.
    """

    pixel_rescaling_mode: Optional[str]
    correction_mask: Optional[InverseSincMask]
    out_of_bounds_mode: str
    fill_value: complex

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        pixel_rescaling_mode: Optional[str] = None,
        correction_mask: Optional[InverseSincMask] = None,
        out_of_bounds_mode: str = "fill",
        fill_value: complex = 0.0 + 0.0j,
    ):
        """**Arguments:**

        - `pixel_rescaling_mode`:
            Method for rescaling the final image to the `AbstractConfig`
            pixel size. See `cryojax.image.rescale_pixel_size` for documentation.
        - `correction_mask`:
            A `cryojax.image.operators.SincCorrectionMask` for performing
            sinc-correction on the linear-interpolated projections. This
            should be computed on a coordinate grid with shape matching
            the `FourierVoxelGridPotential.shape`.
        - `out_of_bounds_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.image.map_coordinates` for documentation.
        - `fill_value`:
            Value for filling out-of-bounds indices. Used only when
            `out_of_bounds_mode = "fill"`.
        """
        self.pixel_rescaling_mode = pixel_rescaling_mode
        self.correction_mask = correction_mask
        self.out_of_bounds_mode = out_of_bounds_mode
        self.fill_value = fill_value

    @override
    def integrate(
        self,
        potential: FourierVoxelGridPotential | FourierVoxelSplinePotential,
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
        of a voxel-based representation in fourier-space, using fourier slice extraction.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `config`: The configuration of the resulting image.

        **Returns:**

        The extracted fourier voxels of the `potential`, at the
        `config.padded_shape` and the `config.pixel_size`.
        """
        frequency_slice = potential.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if potential.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(potential, FourierVoxelSplinePotential):
            fourier_in_plane_potential = (
                self.extract_fourier_slice_from_spline_coefficients(
                    potential.spline_coefficients,
                    frequency_slice,
                )
            )
        elif isinstance(potential, FourierVoxelGridPotential):
            fourier_in_plane_potential = self.extract_fourier_slice_from_grid_points(
                potential.fourier_voxel_grid,
                frequency_slice,
            )
        else:
            raise ValueError(
                "Supported types for `potential` are `FourierVoxelGridPotential` and "
                "`FourierVoxelSplinePotential`."
            )

        # Resize the image to match the AbstractConfig.padded_shape
        if config.padded_shape != (N, N):
            fourier_in_plane_potential = rfftn(
                config.crop_or_pad_to_padded_shape(
                    irfftn(fourier_in_plane_potential, s=(N, N))
                )
            )
        fourier_in_plane_potential = self._postprocess_in_plane_potential(
            fourier_in_plane_potential, potential, config, input_is_rfft=True
        )
        return (
            irfftn(fourier_in_plane_potential, s=config.padded_shape)
            if outputs_real_space
            else fourier_in_plane_potential
        )

    def extract_fourier_slice_from_spline_coefficients(
        self,
        spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
    ) -> Complex[Array, "dim dim//2+1"]:
        """Extract a fourier slice using the interpolation defined by
        `spline_coefficients` at coordinates `frequency_slice_in_pixels`.

        **Arguments:**

        - `spline_coefficients`:
            Spline coefficients of the density grid in fourier space.
            The coefficients should be computed from a `fourier_voxel_grid`
            with the zero frequency component in the center. These are
            typically computed with the function
            `cryojax.image.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`. This argument is
            not used in the `FourierSliceExtraction` class.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam. This argument is
            not used in the `FourierSliceExtraction` class.

        **Returns:**

        The interpolated fourier slice at coordinates `frequency_slice_in_pixels`.
        """
        return _extract_slice_with_cubic_spline(
            spline_coefficients,
            frequency_slice_in_pixels,
            mode=self.out_of_bounds_mode,
            cval=self.fill_value,
        )

    def extract_fourier_slice_from_grid_points(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
    ) -> Complex[Array, "dim dim//2+1"]:
        """Extract a fourier slice of the `fourier_voxel_grid` at coordinates
        `frequency_slice_in_pixels`.

        **Arguments:**

        - `fourier_voxel_grid`:
            Density grid in fourier space. The zero frequency component
            should be in the center.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`. This argument is
            not used in the `FourierSliceExtraction` class.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam. This argument is
            not used in the `FourierSliceExtraction` class.

        **Returns:**

        The interpolated fourier slice at coordinates `frequency_slice_in_pixels`.
        """
        fourier_slice = _extract_slice(
            fourier_voxel_grid,
            frequency_slice_in_pixels,
            interpolation_order=1,
            mode=self.out_of_bounds_mode,
            cval=self.fill_value,
        )
        if self.correction_mask is not None:
            fourier_slice = fftn(self.correction_mask(ifftn(fourier_slice)))

        return fourier_slice


class EwaldSphereExtraction(AbstractVoxelPotentialIntegrator, strict=True):
    """Integrate points to the exit plane by extracting a surface of the ewald
    sphere in fourier space.

    This extracts surfaces using interpolation methods housed in
    `cryojax.image.map_coordinates` and `cryojax.image.map_coordinates_with_cubic_spline`.
    """

    pixel_rescaling_mode: Optional[str]
    correction_mask: Optional[InverseSincMask]
    out_of_bounds_mode: str
    fill_value: complex

    is_projection_approximation: ClassVar[bool] = False

    def __init__(
        self,
        *,
        pixel_rescaling_mode: Optional[str] = None,
        correction_mask: Optional[InverseSincMask] = None,
        out_of_bounds_mode: str = "fill",
        fill_value: complex = 0.0 + 0.0j,
    ):
        """**Arguments:**

        - `pixel_rescaling_mode`:
            Method for rescaling the final image to the `AbstractConfig`
            pixel size. See `cryojax.image.rescale_pixel_size` for documentation.
        - `correction_mask`:
            A `cryojax.image.operators.SincCorrectionMask` for performing
            sinc-correction on the linear-interpolated projections. This
            should be computed on a coordinate grid with shape matching
            the `FourierVoxelGridPotential.shape`.
        - `out_of_bounds_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.image.map_coordinates` for documentation.
        - `fill_value`:
            Value for filling out-of-bounds indices. Used only when
            `out_of_bounds_mode = "fill"`.
        """
        self.pixel_rescaling_mode = pixel_rescaling_mode
        self.correction_mask = correction_mask
        self.out_of_bounds_mode = out_of_bounds_mode
        self.fill_value = fill_value

    @override
    def integrate(
        self,
        potential: FourierVoxelGridPotential | FourierVoxelSplinePotential,
        config: AbstractConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
        | Float[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        """Compute the integrated scattering potential at the `AbstractConfig` settings
        of a voxel-based representation in fourier-space, using fourier slice extraction.

        **Arguments:**

        - `potential`: The scattering potential representation.
        - `config`: The configuration of the resulting image.

        **Returns:**

        The extracted fourier voxels of the `potential`, at the
        `config.padded_shape` and the `config.pixel_size`.
        """
        frequency_slice = potential.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if potential.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(potential, FourierVoxelSplinePotential):
            ewald_sphere_surface = self.extract_ewald_sphere_from_spline_coefficients(
                potential.spline_coefficients,
                frequency_slice,
                potential.voxel_size,
                config.wavelength_in_angstroms,
            )
        elif isinstance(potential, FourierVoxelGridPotential):
            ewald_sphere_surface = self.extract_ewald_sphere_from_grid_points(
                potential.fourier_voxel_grid,
                frequency_slice,
                potential.voxel_size,
                config.wavelength_in_angstroms,
            )
        else:
            raise ValueError(
                "Supported types for `potential` are `FourierVoxelGridPotential` and "
                "`FourierVoxelSplinePotential`."
            )

        # Resize the image to match the AbstractConfig.padded_shape
        if config.padded_shape != (N, N):
            ewald_sphere_surface = fftn(
                config.crop_or_pad_to_padded_shape(ifftn(ewald_sphere_surface, s=(N, N)))
            )
        ewald_sphere_surface = self._postprocess_in_plane_potential(
            ewald_sphere_surface,
            potential,
            config,
            input_is_rfft=False,
        )
        return (
            irfftn(ewald_sphere_surface, s=config.padded_shape)
            if outputs_real_space
            else ewald_sphere_surface
        )

    def extract_ewald_sphere_from_spline_coefficients(
        self,
        spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""],
        wavelength_in_angstroms: Float[Array, ""],
    ) -> Complex[Array, "dim dim"]:
        """Extract an ewald sphere surface of the `fourier_voxel_grid` at
        coordinates normal to `frequency_slice_in_pixels` at wavelength
        `wavelength_in_angstroms`.

        **Arguments:**

        - `spline_coefficients`:
            Spline coefficients of the density grid in fourier space.
            The coefficients should be computed from a `fourier_voxel_grid`
            with the zero frequency component in the center. These are
            typically computed with the function
            `cryojax.image.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam.

        **Returns:**

        The interpolated ewald sphere surface at coordinates normal to
        `frequency_slice_in_pixels`.
        """
        return _extract_ewald_sphere_surface_with_cubic_spline(
            spline_coefficients,
            frequency_slice_in_pixels,
            voxel_size,
            wavelength_in_angstroms,
            mode=self.out_of_bounds_mode,
            cval=self.fill_value,
        )

    def extract_ewald_sphere_from_grid_points(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""],
        wavelength_in_angstroms: Float[Array, ""],
    ) -> Complex[Array, "dim dim"]:
        """Extract an ewald sphere surface of the `fourier_voxel_grid` at
        coordinates normal to `frequency_slice_in_pixels` at wavelength
        `wavelength_in_angstroms`.

        **Arguments:**

        - `fourier_voxel_grid`:
            Density grid in fourier space. The zero frequency component
            should be in the center.
        - `frequency_slice_in_pixels`:
            Frequency central slice coordinate system. The zero
            frequency component should be in the center.
        - `voxel_size`:
            The voxel size of the `fourier_voxel_grid`.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electron beam.

        **Returns:**

        The interpolated ewald sphere surface at coordinates normal to
        `frequency_slice_in_pixels`.
        """
        ewald_sphere_surface = _extract_ewald_sphere_surface(
            fourier_voxel_grid,
            frequency_slice_in_pixels,
            voxel_size,
            wavelength_in_angstroms,
            interpolation_order=1,
            mode=self.out_of_bounds_mode,
            cval=self.fill_value,
        )
        if self.correction_mask is not None:
            ewald_sphere_surface = fftn(self.correction_mask(ifftn(ewald_sphere_surface)))

        return ewald_sphere_surface


def _extract_slice(
    fourier_voxel_grid,
    frequency_slice,
    interpolation_order,
    **kwargs,
) -> Complex[Array, "dim dim//2+1"]:
    return convert_fftn_to_rfftn(
        _extract_surface_from_voxel_grid(
            fourier_voxel_grid,
            frequency_slice,
            is_spline_coefficients=False,
            interpolation_order=interpolation_order,
            **kwargs,
        ),
        mode="real",
    )


def _extract_slice_with_cubic_spline(
    spline_coefficients, frequency_slice, **kwargs
) -> Complex[Array, "dim dim//2+1"]:
    return convert_fftn_to_rfftn(
        _extract_surface_from_voxel_grid(
            spline_coefficients, frequency_slice, is_spline_coefficients=True, **kwargs
        ),
        mode="real",
    )


def _extract_ewald_sphere_surface(
    fourier_voxel_grid,
    frequency_slice,
    voxel_size,
    wavelength,
    interpolation_order,
    **kwargs,
) -> Complex[Array, "dim dim"]:
    ewald_sphere_frequencies = _get_ewald_sphere_surface_from_slice(
        frequency_slice, voxel_size, wavelength
    )
    return _extract_surface_from_voxel_grid(
        fourier_voxel_grid,
        ewald_sphere_frequencies,
        is_spline_coefficients=False,
        interpolation_order=interpolation_order,
        **kwargs,
    )


def _extract_ewald_sphere_surface_with_cubic_spline(
    spline_coefficients, frequency_slice, voxel_size, wavelength, **kwargs
) -> Complex[Array, "dim dim"]:
    ewald_sphere_frequencies = _get_ewald_sphere_surface_from_slice(
        frequency_slice, voxel_size, wavelength
    )
    return _extract_surface_from_voxel_grid(
        spline_coefficients,
        ewald_sphere_frequencies,
        is_spline_coefficients=True,
        **kwargs,
    )


def _get_ewald_sphere_surface_from_slice(
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
    voxel_size: Float[Array, ""],
    wavelength: Float[Array, ""],
) -> Float[Array, "1 dim dim 3"]:
    frequency_slice_with_zero_in_corner = jnp.fft.ifftshift(
        frequency_slice_in_pixels, axes=(0, 1, 2)
    )
    # Get zhat unit vector of the frequency slice
    xhat, yhat = (
        frequency_slice_with_zero_in_corner[0, 0, 1, :],
        frequency_slice_with_zero_in_corner[0, 1, 0, :],
    )
    xhat, yhat = xhat / jnp.linalg.norm(xhat), yhat / jnp.linalg.norm(yhat)
    zhat = jnp.cross(xhat, yhat)
    # Compute the ewald sphere surface, assuming the frequency slice is
    # in a rotated frame
    q_at_slice = frequency_slice_in_pixels
    q_squared = jnp.sum(q_at_slice**2, axis=-1)
    q_at_surface = (
        q_at_slice
        + (wavelength / voxel_size)
        * (q_squared[..., None] * zhat[None, None, None, :])
        / 2
    )
    return q_at_surface


def _extract_surface_from_voxel_grid(
    voxel_grid,
    frequency_coordinates,
    is_spline_coefficients=False,
    interpolation_order=1,
    **kwargs,
):
    # Convert to logical coordinates
    N = frequency_coordinates.shape[1]
    logical_frequency_coordinates = (frequency_coordinates * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_z, k_y, k_x = jnp.transpose(logical_frequency_coordinates, axes=[3, 0, 1, 2])
    if is_spline_coefficients:
        spline_coefficients = voxel_grid
        surface = map_coordinates_with_cubic_spline(
            spline_coefficients, (k_x, k_y, k_z), **kwargs
        )[0, :, :]
    else:
        fourier_voxel_grid = voxel_grid
        surface = map_coordinates(
            fourier_voxel_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
        )[0, :, :]
    # Shift zero frequency component to corner
    surface = jnp.fft.ifftshift(surface)

    return surface
