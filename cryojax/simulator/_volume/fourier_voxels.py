"""
Fourier voxel-based representations of a volume.
"""

from typing import ClassVar, cast
from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import NDArrayLike
from ...ndimage import (
    AbstractFilter,
    InverseSincMask,
    compute_spline_coefficients,
    convert_fftn_to_rfftn,
    fftn,
    ifftn,
    irfftn,
    make_frequency_slice,
    map_coordinates,
    map_coordinates_spline,
    pad_to_shape,
    rfftn,
)
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import AbstractVolumeIntegrator, AbstractVoxelVolume


class AbstractFourierVoxelVolume(AbstractVoxelVolume, strict=True):
    """Abstract interface for a voxel-based volume."""

    frequency_slice_in_pixels: eqx.AbstractVar[Float[Array, "1 dim dim 3"]]

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new volume with a rotated `frequency_slice_in_pixels`."""
        return eqx.tree_at(
            lambda d: d.frequency_slice_in_pixels,
            self,
            pose.rotate_coordinates(self.frequency_slice_in_pixels, inverse=inverse),
        )


class FourierVoxelGridVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space."""

    fourier_voxel_grid: Complex[Array, "dim dim dim"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]

    def __init__(
        self,
        fourier_voxel_grid: Complex[NDArrayLike, "dim dim dim"],
        frequency_slice_in_pixels: Float[NDArrayLike, "1 dim dim 3"],
    ):
        """**Arguments:**

        - `fourier_voxel_grid`:
            The cubic voxel grid in fourier space.
        - `frequency_slice_in_pixels`:
            The frequency slice coordinate system.
        """
        self.fourier_voxel_grid = jnp.asarray(fourier_voxel_grid, dtype=complex)
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `fourier_voxel_grid`."""
        return cast(tuple[int, int, int], self.fourier_voxel_grid.shape)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
        filter: AbstractFilter | None = None,
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A voxel grid in real space.
        - `pad_scale`: Scale factor at which to pad `real_voxel_grid` before fourier
                     transform. Must be a value greater than `1.0`.
        - `pad_mode`: Padding method. See `jax.numpy.pad` for documentation.
        - `filter`: A filter to apply to the result of the fourier transform of
                  `real_voxel_grid`, i.e. `fftn(real_voxel_grid)`. Note that the zero
                  frequency component is assumed to be in the corner.
        """
        # Cast to jax array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Pad template
        if pad_scale < 1.0:
            raise ValueError("`pad_scale` must be greater than 1.0")
        # ... always pad to even size to avoid interpolation issues in
        # fourier slice extraction.
        padded_shape = cast(
            tuple[int, int, int],
            tuple([int(s * pad_scale) for s in real_voxel_grid.shape]),
        )
        padded_real_voxel_grid = pad_to_shape(
            real_voxel_grid, padded_shape, mode=pad_mode
        )
        # Load grid and coordinates. For now, do not store the
        # fourier grid only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used.
        fourier_voxel_grid_with_zero_in_corner = (
            fftn(padded_real_voxel_grid)
            if filter is None
            else filter(fftn(padded_real_voxel_grid))
        )
        # ... store the grid with the zero frequency component in the center
        fourier_voxel_grid = jnp.fft.fftshift(fourier_voxel_grid_with_zero_in_corner)
        # ... create in-plane frequency slice on the half space
        frequency_slice = make_frequency_slice(
            cast(tuple[int, int], padded_real_voxel_grid.shape[:-1]),
            outputs_rfftfreqs=False,
        )

        return cls(fourier_voxel_grid, frequency_slice)


class FourierVoxelSplineVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space, represented
    by spline coefficients.
    """

    spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]

    def __init__(
        self,
        spline_coefficients: Complex[NDArrayLike, "coeff_dim coeff_dim coeff_dim"],
        frequency_slice_in_pixels: Float[NDArrayLike, "1 dim dim 3"],
    ):
        """**Arguments:**

        - `spline_coefficients`:
            The spline coefficents computed from the cubic voxel grid
            in fourier space. See `cryojax.ndimage.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency slice coordinate system.
            See `cryojax.coordinates.make_frequency_slice`.
        """
        self.spline_coefficients = jnp.asarray(spline_coefficients, dtype=complex)
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the original `fourier_voxel_grid` from which
        `coefficients` were computed.
        """
        return cast(
            tuple[int, int, int], tuple([s - 2 for s in self.spline_coefficients.shape])
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
        filter: AbstractFilter | None = None,
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A voxel grid in real space.
        - `pad_scale`: Scale factor at which to pad `real_voxel_grid` before fourier
                     transform. Must be a value greater than `1.0`.
        - `pad_mode`: Padding method. See `jax.numpy.pad` for documentation.
        - `filter`: A filter to apply to the result of the fourier transform of
                  `real_voxel_grid`, i.e. `fftn(real_voxel_grid)`. Note that the zero
                  frequency component is assumed to be in the corner.
        """
        # Cast to jax array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Pad template
        if pad_scale < 1.0:
            raise ValueError("`pad_scale` must be greater than 1.0")
        # ... always pad to even size to avoid interpolation issues in
        # fourier slice extraction.
        padded_shape = cast(
            tuple[int, int, int],
            tuple([int(s * pad_scale) for s in real_voxel_grid.shape]),
        )
        padded_real_voxel_grid = pad_to_shape(
            real_voxel_grid, padded_shape, mode=pad_mode
        )
        # Load grid and coordinates. For now, do not store the
        # fourier grid only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used.
        fourier_voxel_grid_with_zero_in_corner = (
            fftn(padded_real_voxel_grid)
            if filter is None
            else filter(fftn(padded_real_voxel_grid))
        )
        # ... store the grid with the zero frequency component in the center
        fourier_voxel_grid = jnp.fft.fftshift(fourier_voxel_grid_with_zero_in_corner)
        # ... compute spline coefficients
        spline_coefficients = compute_spline_coefficients(fourier_voxel_grid)
        # ... create in-plane frequency slice on the half space
        frequency_slice = make_frequency_slice(
            cast(tuple[int, int], padded_real_voxel_grid.shape[:-1]),
            outputs_rfftfreqs=False,
        )

        return cls(spline_coefficients, frequency_slice)


class FourierSliceExtraction(
    AbstractVolumeIntegrator[FourierVoxelGridVolume | FourierVoxelSplineVolume],
    strict=True,
):
    """Integrate points to the exit plane using the Fourier
    projection-slice theorem.

    This extracts slices using interpolation methods housed in
    `cryojax.ndimage.map_coordinates` and
    `cryojax.ndimage.map_coordinates_spline`.
    """

    outputs_integral: bool
    correction_mask: InverseSincMask | None
    out_of_bounds_mode: str
    fill_value: complex

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        outputs_integral: bool = True,
        correction_mask: InverseSincMask | None = None,
        out_of_bounds_mode: str = "fill",
        fill_value: complex = 0.0 + 0.0j,
    ):
        """**Arguments:**

        - `outputs_integral`:
            If `True`, return the fourier slice
            *multiplied by the voxel size*. Including the voxel size
            numerical approximates the projection integral and is
            necessary for simulating images in physical units.
        - `correction_mask`:
            A `cryojax.ndimage.transforms.InverseSincMask` for performing
            sinc-correction on the linear-interpolated projections. This
            should be computed on a coordinate grid with shape matching
            the `FourierVoxelGridVolume.shape`.
        - `out_of_bounds_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.ndimage.map_coordinates` for documentation.
        - `fill_value`:
            Value for filling out-of-bounds indices. Used only when
            `out_of_bounds_mode = "fill"`.
        """
        self.outputs_integral = outputs_integral
        self.correction_mask = correction_mask
        self.out_of_bounds_mode = out_of_bounds_mode
        self.fill_value = fill_value

    @override
    def integrate(
        self,
        volume_representation: FourierVoxelGridVolume | FourierVoxelSplineVolume,
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
        of a voxel-based representation in fourier-space,
        using fourier slice extraction.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The configuration of the resulting image.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in fourier.

        **Returns:**

        The extracted fourier voxels of the `volume_representation`,
        at the `image_config.padded_shape` and the `image_config.pixel_size`.
        """
        frequency_slice = volume_representation.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if volume_representation.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(volume_representation, FourierVoxelSplineVolume):
            fourier_projection = self.extract_fourier_slice_from_spline(
                volume_representation.spline_coefficients,
                frequency_slice,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            fourier_projection = self.extract_fourier_slice_from_grid(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
            )
        else:
            raise ValueError(
                "Supported types for `volume_representation` are "
                "`FourierVoxelGridVolume` and FourierVoxelSplineVolume`."
            )

        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != (N, N):
            fourier_projection = rfftn(
                image_config.crop_or_pad_to_padded_shape(
                    irfftn(fourier_projection, s=(N, N))
                )
            )
        # Scale by voxel size to convert from projection to integral
        if self.outputs_integral:
            fourier_projection *= image_config.pixel_size
        return (
            irfftn(fourier_projection, s=image_config.padded_shape)
            if outputs_real_space
            else fourier_projection
        )

    def extract_fourier_slice_from_spline(
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

    def extract_fourier_slice_from_grid(
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


class EwaldSphereExtraction(
    AbstractVolumeIntegrator[FourierVoxelGridVolume | FourierVoxelSplineVolume],
    strict=True,
):
    """Integrate points to the exit plane by extracting a surface of
    the ewald sphere in fourier space.

    This extracts surfaces using interpolation methods housed in
    `cryojax.image.map_coordinates`
    and `cryojax.image.map_coordinates_spline`.
    """

    outputs_integral: bool
    correction_mask: InverseSincMask | None
    out_of_bounds_mode: str
    fill_value: complex

    is_projection_approximation: ClassVar[bool] = False

    def __init__(
        self,
        *,
        outputs_integral: bool = True,
        correction_mask: InverseSincMask | None = None,
        out_of_bounds_mode: str = "fill",
        fill_value: complex = 0.0 + 0.0j,
    ):
        """**Arguments:**

        - `outputs_integral`:
            If `True`, return the ewald sphere surface
            *multiplied by the voxel size*. Including the voxel size
            numerical approximates the projection integral and is
            necessary for simulating images in physical units.
        - `correction_mask`:
            A `cryojax.ndimage.transforms.SincCorrectionMask` for performing
            sinc-correction on the linear-interpolated projections. This
            should be computed on a coordinate grid with shape matching
            the `FourierVoxelGridVolume.shape`.
        - `out_of_bounds_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.image.map_coordinates` for documentation.
        - `fill_value`:
            Value for filling out-of-bounds indices. Used only when
            `out_of_bounds_mode = "fill"`.
        """
        self.outputs_integral = outputs_integral
        self.correction_mask = correction_mask
        self.out_of_bounds_mode = out_of_bounds_mode
        self.fill_value = fill_value

    @override
    def integrate(
        self,
        volume_representation: FourierVoxelGridVolume | FourierVoxelSplineVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
        | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    ):
        """Integrate the volume at the `AbstractImageConfig` settings
        of a voxel-based representation in fourier-space, using fourier
        slice extraction.

        **Arguments:**

        - `volume_representation`: The volume representation.
        - `image_config`: The configuration of the resulting image.

        **Returns:**

        The extracted fourier voxels of the `volume_representation`, at the
        `image_config.padded_shape` and the `image_config.pixel_size`.
        """
        frequency_slice = volume_representation.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if volume_representation.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(volume_representation, FourierVoxelSplineVolume):
            ewald_sphere_surface = self.extract_ewald_sphere_from_spline_coefficients(
                volume_representation.spline_coefficients,
                frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            ewald_sphere_surface = self.extract_ewald_sphere_from_grid_points(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
            )
        else:
            raise ValueError(
                "Supported types for `volume_representation` are "
                "`FourierVoxelGridVolume` and `FourierVoxelSplineVolume`."
            )

        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != (N, N):
            ewald_sphere_surface = fftn(
                image_config.crop_or_pad_to_padded_shape(
                    ifftn(ewald_sphere_surface, s=(N, N))
                )
            )
        # Scale by voxel size to convert from projection to integral
        if self.outputs_integral:
            ewald_sphere_surface *= image_config.pixel_size
        return (
            irfftn(ewald_sphere_surface, s=image_config.padded_shape)
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
        surface = map_coordinates_spline(spline_coefficients, (k_x, k_y, k_z), **kwargs)[
            0, :, :
        ]
    else:
        fourier_voxel_grid = voxel_grid
        surface = map_coordinates(
            fourier_voxel_grid, (k_x, k_y, k_z), interpolation_order, **kwargs
        )[0, :, :]
    # Shift zero frequency component to corner
    surface = jnp.fft.ifftshift(surface)

    return surface
