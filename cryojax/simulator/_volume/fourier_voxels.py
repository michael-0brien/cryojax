"""
Fourier voxel-based representations of a volume.
"""

import abc
from typing import Any, ClassVar, Self, cast
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import NDArrayLike
from ...ndimage import (
    compute_spline_coefficients,
    convert_fftn_to_rfftn,
    fftn,
    ifftn,
    irfftn,
    make_1d_coordinate_grid,
    make_fftshift_phase,
    make_frequency_slice,
    map_coordinates,
    map_coordinates_spline,
    pad_to_shape,
    query_efficient_grid_size,
    resize_with_crop_or_pad,
    rfftn,
)
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import (
    AbstractVolumeIntegrator,
    AbstractVoxelVolume,
    EwaldSphereArray,
    ProjectionArray,
)


class AbstractFourierVoxelVolume(AbstractVoxelVolume, strict=True):
    """Abstract interface for a voxel-based volume."""

    frequency_slice_in_pixels: eqx.AbstractVar[Float[Array, "1 dim dim 3"]]

    def __check_init__(self):
        if any(s % 2 == 1 for s in self.shape):
            raise ValueError(
                f"`{type(self).__name__}` does not support odd voxel map dimensions, "
                f"but got a voxel map with shape `{self.shape}`. Please pass "
                "a voxel map with even dimensions."
            )
        dim = self.shape[0]
        if self.shape != (dim, dim, dim):
            raise AttributeError(
                f"Only cubic boxes are supported for `{type(self).__name__}.shape`, "
                f"but got `shape = {self.shape}`."
            )

    @classmethod
    @abc.abstractmethod
    def from_fourier_voxel_grid(
        cls,
        fourier_voxel_grid: Float[NDArrayLike, "dim dim dim"],
    ) -> Self:
        raise NotImplementedError

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new volume with a rotated `frequency_slice_in_pixels`."""
        return eqx.tree_at(
            lambda d: d.frequency_slice_in_pixels,
            self,
            pose.rotate_coordinates(
                self.frequency_slice_in_pixels, inverse=self.is_frame_rotation
            ),
        )


class FourierVoxelGridVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space.

    !!! note
        Prefer the class-method constructors over direct instantiation
        via ` volume = FourierVoxelGridVolume(...)`:

        - `from_real_voxel_grid`:
            Instantiate from a real-space map.
        - `from_fourier_voxel_grid`:
            Instantiate from the output of `cryojax.ndimage.fftn`.

        Using `__init__` directly requires `fourier_voxel_grid` and
        `frequency_grid_in_pixels` to have the correct conventions for
        interpolation. This is:

        ```python
        import jax.numpy as jnp
        import cryojax.ndimage as im

        # Load real voxel grid
        real_voxel_grid = ...
        # Prepare arguments
        # ... verify cubic
        dim = real_voxel_grid.shape[0]
        assert all(d == dim for d in real_voxel_grid.shape)
        # ... compute grid and coordinates in correct convention
        fourier_voxel_grid = jnp.fft.fftshift(im.fftn(jnp.fft.ifftshift(real_voxel_grid))))
        frequency_slice = jnp.fft.fftshift(im.make_frequency_slice((dim, dim)))
        ```
    """  # noqa: E501

    fourier_voxel_grid: Complex[Array, "dim dim dim"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]

    is_frame_rotation: ClassVar[bool] = True

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
        # Multiply by phase correction for interpolation logic
        self.fourier_voxel_grid = jnp.asarray(fourier_voxel_grid, dtype=complex)
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `fourier_voxel_grid`."""
        return cast(tuple[int, int, int], self.fourier_voxel_grid.shape)

    @classmethod
    def from_fourier_voxel_grid(cls, fourier_voxel_grid: NDArrayLike) -> Self:
        """Load from a fourier-domain 3D voxel grid.

        This should be the output of

        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage import im

        fourier_voxel_grid = im.fftn(real_voxel_grid)
        volume = cxs.FourierVoxelSplineVolume(fourier_voxel_grid)
        ```

        **Arguments:**

        - `fourier_voxel_grid`:
            A voxel grid in fourier space.
        """
        fourier_voxel_grid, frequency_slice = _prepare_fourier_voxel_arguments(
            jnp.asarray(fourier_voxel_grid)
        )

        return cls(jnp.asarray(fourier_voxel_grid), frequency_slice)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        apply_deconvolve: bool = False,
        pad_scale: float = 1.0,
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`:
            A voxel grid in real space.
        - `apply_deconvolve`:
            If `True`, deconvolve the effect of the linear interpolation
            kernel for more accurate Fourier slice extraction.
        - `pad_scale`:
            Scale factor at which to pad `real_voxel_grid` before fourier
            transform. Must be a value greater than `1.0`.
        """
        # Cast to JAX array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Preprocess to fourier grid, deconvolving after any padding so that
        # the sinc² correction uses the actual Fourier grid size.
        fourier_voxel_grid, frequency_slice = _real_to_fourier_voxels(
            cls, real_voxel_grid, pad_scale, apply_deconvolve
        )

        return cls(fourier_voxel_grid, frequency_slice)


class FourierVoxelSplineVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space, represented
    by spline coefficients.
    """

    spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]

    is_frame_rotation: ClassVar[bool] = True

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
    def from_fourier_voxel_grid(cls, fourier_voxel_grid: NDArrayLike) -> Self:
        """Load from a fourier-domain 3D voxel grid.

        This should be the output of

        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage import im

        fourier_voxel_grid = im.fftn(real_voxel_grid)
        volume = cxs.FourierVoxelSplineVolume(fourier_voxel_grid)
        ```

        **Arguments:**

        - `fourier_voxel_grid`:
            A voxel grid in fourier space.
        """
        fourier_voxel_grid, frequency_slice = _prepare_fourier_voxel_arguments(
            jnp.asarray(fourier_voxel_grid)
        )
        # Compute spline coefficients
        spline_coefficients = compute_spline_coefficients(fourier_voxel_grid)

        return cls(spline_coefficients, frequency_slice)

    @classmethod
    def from_real_voxel_grid(
        cls, real_voxel_grid: Float[NDArrayLike, "dim dim dim"], *, pad_scale: float = 1.0
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`:
            A voxel grid in real space.
        - `pad_scale`:
            Scale factor at which to pad `real_voxel_grid` before fourier
            transform. Must be a value greater than `1.0`.
        """
        # Cast to JAX array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Preprocess to fourier grid
        fourier_voxel_grid, frequency_slice = _real_to_fourier_voxels(
            cls, real_voxel_grid, pad_scale
        )
        # Compute spline coefficients
        spline_coefficients = compute_spline_coefficients(fourier_voxel_grid)

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
    out_of_bounds_mode: str

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        outputs_integral: bool = True,
        out_of_bounds_mode: str = "fill",
    ):
        """**Arguments:**

        - `outputs_integral`:
            If `True`, return the fourier slice
            *multiplied by the voxel size*. Including the voxel size
            numerically approximates the projection integral and is
            necessary for simulating images in physical units.
        - `out_of_bounds_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.ndimage.map_coordinates` for documentation.
        """
        self.outputs_integral = outputs_integral
        self.out_of_bounds_mode = out_of_bounds_mode

    @override
    def integrate(
        self,
        volume_representation: FourierVoxelGridVolume | FourierVoxelSplineVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionArray:
        """Integrate the volume at the `AbstractImageConfig` settings
        of a voxel-based representation in fourier-space,
        using fourier slice extraction.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The image configuration.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in Fourier.

        **Returns:**

        The volume projection in real or Fourier space at the
        `AbstractImageConfig.padded_shape` and the `image_config.pixel_size`.
        """
        frequency_slice = volume_representation.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        # Compute the fourier projection
        if isinstance(volume_representation, FourierVoxelSplineVolume):
            fourier_projection = _extract_slice_spline(
                volume_representation.spline_coefficients,
                frequency_slice,
                mode=self.out_of_bounds_mode,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            fourier_projection = _extract_slice(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
                mode=self.out_of_bounds_mode,
            )
        else:
            raise ValueError(
                "Got unsupported type for `volume_representation` in "
                "`FourierSliceExtraction.integrate`. Expected `FourierVoxelGridVolume` "
                "or `FourierVoxelSplineVolume`, "
                f"but got `{volume_representation.__class__.__name__}`."
            )

        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != (N, N):
            fourier_projection = rfftn(
                resize_with_crop_or_pad(
                    irfftn(fourier_projection, s=(N, N)), image_config.padded_shape
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


class EwaldSphereExtraction(
    AbstractVolumeIntegrator[FourierVoxelGridVolume | FourierVoxelSplineVolume],
    strict=True,
):
    """Integrate points to the exit plane by extracting a surface of
    the ewald sphere in fourier space.

    This extracts surfaces using interpolation methods housed in
    `cryojax.ndimage.map_coordinates`
    and `cryojax.ndimage.map_coordinates_spline`.
    """

    outputs_integral: bool
    out_of_bounds_mode: str

    outputs_ewald_sphere: ClassVar[bool] = True

    def __init__(
        self,
        *,
        outputs_integral: bool = True,
        out_of_bounds_mode: str = "fill",
    ):
        """**Arguments:**

        - `outputs_integral`:
            If `True`, return the ewald sphere surface
            *multiplied by the voxel size*. Including the voxel size
            numerically approximates the projection integral and is
            necessary for simulating images in physical units.
        - `out_of_bounds_mode`:
            Specify how to handle out of bounds indexing. See
            `cryojax.ndimage.map_coordinates` for documentation.
        """
        self.outputs_integral = outputs_integral
        self.out_of_bounds_mode = out_of_bounds_mode

    @override
    def integrate(
        self,
        volume_representation: FourierVoxelGridVolume | FourierVoxelSplineVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> EwaldSphereArray:
        """Extract the ewald sphere surface.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The image configuration.
        - `outputs_real_space`:
            If `True`, return the Ewald sphere surface in
            real space. Otherwise, return in Fourier.

        **Returns:**

        The Ewald sphere surface in the real-space or fourier-space at the
        `image_config.padded_shape`, `image_config.pixel_size`,
        and `image_config.voltage_in_kilovolts`.
        """
        frequency_slice = volume_representation.frequency_slice_in_pixels
        N = frequency_slice.shape[1]
        if volume_representation.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # Compute the fourier projection
        if isinstance(volume_representation, FourierVoxelSplineVolume):
            ewald_sphere_surface = _extract_ewald_sphere_spline(
                volume_representation.spline_coefficients,
                frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
                mode=self.out_of_bounds_mode,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            ewald_sphere_surface = _extract_ewald_sphere(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
                mode=self.out_of_bounds_mode,
            )
        else:
            raise ValueError(
                "Got unsupported type for `volume_representation` in "
                "`EwaldSphereExtraction.integrate`. Expected `FourierVoxelGridVolume` "
                "or `FourierVoxelSplineVolume`, "
                f"but got `{volume_representation.__class__.__name__}`."
            )

        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != (N, N):
            ewald_sphere_surface = fftn(
                resize_with_crop_or_pad(
                    ifftn(ewald_sphere_surface, s=(N, N)), image_config.padded_shape
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


def _extract_slice(
    fourier_voxel_grid: Array,
    frequency_slice: Array,
    **kwargs: Any,
) -> Complex[Array, "dim dim//2+1"]:
    return convert_fftn_to_rfftn(
        _extract_surface_from_voxel_grid(
            fourier_voxel_grid,
            frequency_slice,
            is_spline_coefficients=False,
            **kwargs,
        ),
        mode="zero",
    )


def _extract_slice_spline(
    spline_coefficients: Array, frequency_slice: Array, **kwargs: Any
) -> Complex[Array, "dim dim//2+1"]:
    return convert_fftn_to_rfftn(
        _extract_surface_from_voxel_grid(
            spline_coefficients, frequency_slice, is_spline_coefficients=True, **kwargs
        ),
        mode="zero",
    )


def _extract_ewald_sphere(
    fourier_voxel_grid: Array,
    frequency_slice: Array,
    voxel_size: Array,
    wavelength: Array,
    **kwargs: Any,
) -> Complex[Array, "dim dim"]:
    ewald_sphere_frequencies = _get_ewald_sphere_surface_from_slice(
        frequency_slice, voxel_size, wavelength
    )
    return _extract_surface_from_voxel_grid(
        fourier_voxel_grid,
        ewald_sphere_frequencies,
        is_spline_coefficients=False,
        **kwargs,
    )


def _extract_ewald_sphere_spline(
    spline_coefficients: Array,
    frequency_slice: Array,
    voxel_size: Array,
    wavelength: Array,
    **kwargs: Any,
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
    frequency_slice_in_pixels: Array, voxel_size: Array, wavelength: Array
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
    voxel_grid: Array,
    frequency_coordinates: Array,
    is_spline_coefficients: bool = False,
    **kwargs: Any,
):
    # Convert to logical coordinates
    N = frequency_coordinates.shape[1]
    logical_coordinates = (frequency_coordinates * N) + N // 2
    # Convert arguments to map_coordinates convention and compute
    k_x, k_y, k_z = jnp.transpose(logical_coordinates, axes=[3, 0, 1, 2])
    if is_spline_coefficients:
        spline_coefficients = voxel_grid
        surface = map_coordinates_spline(spline_coefficients, (k_z, k_y, k_x), **kwargs)[
            0, :, :
        ]
    else:
        fourier_voxel_grid = voxel_grid
        surface = map_coordinates(fourier_voxel_grid, (k_z, k_y, k_x), **kwargs)[0, :, :]
    # FFT shift and multiply by (-1)^k phase factors
    surface = jnp.fft.ifftshift(make_fftshift_phase(surface.shape) * surface)

    return surface


def _deconvolve_linear(real_voxel_grid: Array) -> Array:
    """Deconvolves the effect of the triangular interpolation kernel"""
    dim = real_voxel_grid.shape[0]
    assert all(dim == d for d in real_voxel_grid.shape)
    x = make_1d_coordinate_grid(dim)
    sinc_array = jnp.sinc(x / dim)
    deconvolve_factor = (
        sinc_array[:, None, None] * sinc_array[None, :, None] * sinc_array[None, None, :]
    ) ** 2
    return real_voxel_grid / deconvolve_factor


def _real_to_fourier_voxels(
    cls, real_voxel_grid: Array, pad_scale: float, apply_deconvolve: bool = False
) -> tuple[Array, Array]:
    if pad_scale == 1.0:
        shape_p = real_voxel_grid.shape
        real_voxel_grid_p = real_voxel_grid
    elif pad_scale > 1.0:
        shape_p = query_efficient_grid_size(
            real_voxel_grid.shape, pad_scale=pad_scale, only_even=True
        )
        real_voxel_grid_p = pad_to_shape(real_voxel_grid, shape_p)
    else:
        raise ValueError(
            "Invalid value for "
            f"`{cls.__name__}.from_real_voxel_grid(..., pad_scale=...)`. "
            f"This must be greater than `1.0`, but got value `{pad_scale}`."
        )
    # Deconvolve after padding so the sinc² correction uses the actual
    # Fourier grid size (N_pad), not the original unpadded size.
    if apply_deconvolve:
        real_voxel_grid_p = _deconvolve_linear(real_voxel_grid_p)

    return _prepare_fourier_voxel_arguments(fftn(real_voxel_grid_p))


def _prepare_fourier_voxel_arguments(fourier_voxel_grid: Array) -> tuple[Array, Array]:
    dim = fourier_voxel_grid.shape[0]
    return (
        jnp.fft.fftshift(make_fftshift_phase((dim, dim, dim)) * fourier_voxel_grid),
        make_frequency_slice((dim, dim), fftshifted=True),
    )
