"""
Fourier voxel-based representations of a volume.
"""

import abc
from typing import Any, ClassVar, Self
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
    is_rfft: eqx.AbstractVar[bool]

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


def _check_voxel_array_shape(
    shape: tuple[int, ...],
    is_rfft: bool,
    padding: int,
    cls_name: str,
    array_name: str,
) -> None:
    """Validate that `shape` (the raw stored array shape, e.g.
    `fourier_voxel_grid.shape` or `spline_coefficients.shape`) is consistent
    with a cubic, even-dimension volume, given `is_rfft` (whether the last
    axis is RFFT-truncated) and `padding` (extra samples added per axis,
    e.g. `2` for cubic-spline coefficients).
    """
    d0, d1, d2 = (s - padding for s in shape)
    if d0 % 2 == 1:
        raise ValueError(
            f"`{cls_name}` does not support odd voxel map dimensions, but got "
            f"a voxel map with `{array_name}.shape = {shape}`. Please pass a "
            "voxel map with even dimensions."
        )
    expected_d2 = d0 // 2 + 1 if is_rfft else d0
    if d1 != d0 or d2 != expected_d2:
        expected_shape = tuple(s + padding for s in (d0, d0, expected_d2))
        # Common misuse: the array is a valid voxel grid, just for the
        # opposite `is_rfft` convention (e.g. passed the output of `rfftn`
        # but left `is_rfft` at its default of `False`, or vice versa).
        other_d2 = d0 if is_rfft else d0 // 2 + 1
        if d1 == d0 and d2 == other_d2:
            raise AttributeError(
                f"`{array_name}` passed to `{cls_name}` has shape `{shape}`, "
                f"which does not match `is_rfft={is_rfft}` (expected shape "
                f"`{expected_shape}`). This shape matches `is_rfft={not is_rfft}` "
                f"instead -- did you mean to set `is_rfft={not is_rfft}`?"
            )
        raise AttributeError(
            f"`{array_name}` passed to `{cls_name}` has an invalid shape "
            f"`{shape}`. Expected shape `{expected_shape}` for "
            f"`is_rfft={is_rfft}`."
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

    fourier_voxel_grid: Complex[Array, "dim dim dim"] | Complex[Array, "dim dim dim//2+1"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]
    is_rfft: bool = eqx.field(static=True)

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        fourier_voxel_grid: (
            Complex[NDArrayLike, "dim dim dim"] | Complex[NDArrayLike, "dim dim dim//2+1"]
        ),
        frequency_slice_in_pixels: Float[NDArrayLike, "1 dim dim 3"],
        is_rfft: bool = False,
    ):
        """**Arguments:**

        - `fourier_voxel_grid`:
            The cubic voxel grid in fourier space. If `is_rfft = True`,
            this is truncated to the half-space `(dim, dim, dim // 2 + 1)`,
            as returned by `cryojax.ndimage.rfftn`.
        - `frequency_slice_in_pixels`:
            The frequency slice coordinate system.
        - `is_rfft`:
            Whether `fourier_voxel_grid` is a half-space RFFT grid (see
            above) rather than the full FFT grid.
        """
        # Multiply by phase correction for interpolation logic
        self.fourier_voxel_grid = jnp.asarray(fourier_voxel_grid, dtype=complex)
        _check_voxel_array_shape(
            self.fourier_voxel_grid.shape,
            is_rfft,
            padding=0,
            cls_name=type(self).__name__,
            array_name="fourier_voxel_grid",
        )
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )
        self.is_rfft = is_rfft

    @property
    def shape(self) -> tuple[int, int, int]:
        """The logical cubic shape of the volume, regardless of whether
        `fourier_voxel_grid` is stored as a full or half-space (RFFT) grid.
        """
        dim = self.fourier_voxel_grid.shape[0]
        return (dim, dim, dim)

    @classmethod
    def from_fourier_voxel_grid(
        cls, fourier_voxel_grid: NDArrayLike, *, is_rfft: bool = True
    ) -> Self:
        """Load from a fourier-domain 3D voxel grid.

        This should be the output of

        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage import im

        fourier_voxel_grid = im.rfftn(real_voxel_grid) if is_rfft else im.fftn(real_voxel_grid)
        volume = cxs.FourierVoxelSplineVolume(fourier_voxel_grid)
        ```

        **Arguments:**

        - `fourier_voxel_grid`:
            A voxel grid in fourier space. If `is_rfft = True`, this
            should be the output of `cryojax.ndimage.rfftn`; otherwise,
            the output of `cryojax.ndimage.fftn`.
        - `is_rfft`:
            Whether `fourier_voxel_grid` is the output of `cryojax.ndimage.rfftn`
            (`True`, default) rather than `cryojax.ndimage.fftn` (`False`). If
            `True`, the volume is stored as a half-space RFFT grid, halving
            memory usage. This relies on the fact that `fourier_voxel_grid`
            is the transform of a real-valued signal.
        """  # noqa: E501
        fourier_voxel_grid, frequency_slice = _prepare_fourier_voxel_arguments(
            jnp.asarray(fourier_voxel_grid), use_rfft=is_rfft
        )

        return cls(jnp.asarray(fourier_voxel_grid), frequency_slice, is_rfft=is_rfft)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        apply_deconvolve: bool = False,
        pad_scale: float = 1.0,
        use_rfft: bool = True,
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
        - `use_rfft`:
            If `True` (default), store the volume as a half-space RFFT
            grid, halving memory usage and speeding up construction (via
            `cryojax.ndimage.rfftn` instead of `cryojax.ndimage.fftn`).
        """
        # Cast to JAX array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Preprocess to fourier grid, deconvolving after any padding so that
        # the sinc² correction uses the actual Fourier grid size.
        fourier_voxel_grid, frequency_slice = _real_to_fourier_voxels(
            cls, real_voxel_grid, pad_scale, apply_deconvolve, use_rfft=use_rfft
        )

        return cls(fourier_voxel_grid, frequency_slice, is_rfft=use_rfft)


class FourierVoxelSplineVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space, represented
    by spline coefficients.
    """

    spline_coefficients: (
        Complex[Array, "coeff_dim coeff_dim coeff_dim"]
        | Complex[Array, "coeff_dim coeff_dim coeff_dim//2+1"]
    )
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]
    is_rfft: bool = eqx.field(static=True)

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        spline_coefficients: (
            Complex[NDArrayLike, "coeff_dim coeff_dim coeff_dim"]
            | Complex[NDArrayLike, "coeff_dim coeff_dim coeff_dim//2+1"]
        ),
        frequency_slice_in_pixels: Float[NDArrayLike, "1 dim dim 3"],
        is_rfft: bool = False,
    ):
        """**Arguments:**

        - `spline_coefficients`:
            The spline coefficents computed from the cubic voxel grid
            in fourier space. See `cryojax.ndimage.compute_spline_coefficients`.
            If `is_rfft = True`, these are computed from the half-space
            RFFT grid (i.e. last axis of size `dim // 2 + 1`, before the
            `+ 2` spline padding), as returned by `cryojax.ndimage.rfftn`.
        - `frequency_slice_in_pixels`:
            Frequency slice coordinate system.
            See `cryojax.coordinates.make_frequency_slice`.
        - `is_rfft`:
            Whether `spline_coefficients` were computed from a half-space
            RFFT grid (see above) rather than the full FFT grid.
        """
        self.spline_coefficients = jnp.asarray(spline_coefficients, dtype=complex)
        _check_voxel_array_shape(
            self.spline_coefficients.shape,
            is_rfft,
            padding=2,
            cls_name=type(self).__name__,
            array_name="spline_coefficients",
        )
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )
        self.is_rfft = is_rfft

    @property
    def shape(self) -> tuple[int, int, int]:
        """The logical cubic shape of the original `fourier_voxel_grid`
        from which `coefficients` were computed, regardless of whether it
        was a full or half-space (RFFT) grid.
        """
        dim = self.spline_coefficients.shape[0] - 2
        return (dim, dim, dim)

    @classmethod
    def from_fourier_voxel_grid(
        cls, fourier_voxel_grid: NDArrayLike, *, is_rfft: bool = True
    ) -> Self:
        """Load from a fourier-domain 3D voxel grid.

        This should be the output of

        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage import im

        fourier_voxel_grid = im.rfftn(real_voxel_grid) if is_rfft else im.fftn(real_voxel_grid)
        volume = cxs.FourierVoxelSplineVolume(fourier_voxel_grid)
        ```

        **Arguments:**

        - `fourier_voxel_grid`:
            A voxel grid in fourier space. If `is_rfft = True`, this
            should be the output of `cryojax.ndimage.rfftn`; otherwise,
            the output of `cryojax.ndimage.fftn`.
        - `is_rfft`:
            Whether `fourier_voxel_grid` is the output of `cryojax.ndimage.rfftn`
            (`True`, default) rather than `cryojax.ndimage.fftn` (`False`). If
            `True`, spline coefficients are computed from a half-space RFFT
            grid, halving memory usage.
        """  # noqa: E501
        fourier_voxel_grid, frequency_slice = _prepare_fourier_voxel_arguments(
            jnp.asarray(fourier_voxel_grid), use_rfft=is_rfft
        )
        # Compute spline coefficients
        spline_coefficients = compute_spline_coefficients(fourier_voxel_grid)

        return cls(spline_coefficients, frequency_slice, is_rfft=is_rfft)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        pad_scale: float = 1.0,
        use_rfft: bool = True,
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`:
            A voxel grid in real space.
        - `pad_scale`:
            Scale factor at which to pad `real_voxel_grid` before fourier
            transform. Must be a value greater than `1.0`.
        - `use_rfft`:
            If `True` (default), store the volume as a half-space RFFT
            grid, halving memory usage and speeding up construction (via
            `cryojax.ndimage.rfftn` instead of `cryojax.ndimage.fftn`).
        """
        # Cast to JAX array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Preprocess to fourier grid
        fourier_voxel_grid, frequency_slice = _real_to_fourier_voxels(
            cls, real_voxel_grid, pad_scale, use_rfft=use_rfft
        )
        # Compute spline coefficients
        spline_coefficients = compute_spline_coefficients(fourier_voxel_grid)

        return cls(spline_coefficients, frequency_slice, is_rfft=use_rfft)


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
    unroll_gather: bool

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        outputs_integral: bool = True,
        out_of_bounds_mode: str = "fill",
        unroll_gather: bool = True,
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
        - `unroll_gather`:
            Passed to `cryojax.ndimage.map_coordinates`/
            `map_coordinates_spline`. Defaults to `True` everywhere, but for
            `FourierVoxelSplineVolume`, `unroll_gather=False` is often
            substantially faster and is usually worth trying first, falling
            back to `True` only if you hit GPU out-of-memory errors or are
            working with very large batches.
        """
        self.outputs_integral = outputs_integral
        self.out_of_bounds_mode = out_of_bounds_mode
        self.unroll_gather = unroll_gather

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
                is_rfft=volume_representation.is_rfft,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            fourier_projection = _extract_slice(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
                is_rfft=volume_representation.is_rfft,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
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
    unroll_gather: bool

    outputs_ewald_sphere: ClassVar[bool] = True

    def __init__(
        self,
        *,
        outputs_integral: bool = True,
        out_of_bounds_mode: str = "fill",
        unroll_gather: bool = True,
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
        - `unroll_gather`:
            Passed to `cryojax.ndimage.map_coordinates`/
            `map_coordinates_spline`. Defaults to `True` everywhere, but for
            `FourierVoxelSplineVolume`, `unroll_gather=False` is often
            substantially faster and is usually worth trying first, falling
            back to `True` only if you hit GPU out-of-memory errors or are
            working with very large batches.
        """
        self.outputs_integral = outputs_integral
        self.out_of_bounds_mode = out_of_bounds_mode
        self.unroll_gather = unroll_gather

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
                is_rfft=volume_representation.is_rfft,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            ewald_sphere_surface = _extract_ewald_sphere(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
                is_rfft=volume_representation.is_rfft,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
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
    is_rfft: bool = False,
    **kwargs: Any,
):
    # Convert to logical coordinates
    N = frequency_coordinates.shape[1]
    if is_rfft:
        # `voxel_grid`'s last axis only stores non-negative frequencies
        # along x (i.e. `F(-q) = conj(F(q))` is not stored, only `F(q)`
        # for `q_x >= 0`). Reflect the whole 3-vector through the origin
        # whenever `q_x < 0`, so we always look up a point with `q_x >= 0`,
        # then conjugate the interpolated result to correct for it. This is
        # exact, not an approximation: it's evaluated once per query point,
        # on the continuous coordinate, before any interpolation taps are
        # generated, so taps never straddle the truncation boundary.
        sign = jnp.where(frequency_coordinates[..., 0] < 0, -1.0, 1.0)
        reflected = sign[..., None] * frequency_coordinates
        k_x = reflected[..., 0] * N  # rfft/corner convention: no N // 2 offset
        k_y = reflected[..., 1] * N + N // 2
        k_z = reflected[..., 2] * N + N // 2
        # The centered axes' Nyquist bin (frequency -0.5) is stored only at
        # index 0, not also at index N -- +0.5 and -0.5 are the same
        # (aliased) physical frequency. Reflecting a coordinate that was
        # exactly at -0.5 lands exactly on index N, one past the valid
        # range; wrap that exact case back to index 0, without touching any
        # other (genuinely out-of-bounds) coordinate.
        k_y = jnp.where(k_y == N, 0.0, k_y)
        k_z = jnp.where(k_z == N, 0.0, k_z)
    else:
        logical_coordinates = (frequency_coordinates * N) + N // 2
        # Convert arguments to map_coordinates convention and compute
        k_x, k_y, k_z = jnp.transpose(logical_coordinates, axes=[3, 0, 1, 2])
        sign = None
    if is_spline_coefficients:
        spline_coefficients = voxel_grid
        surface = map_coordinates_spline(spline_coefficients, (k_z, k_y, k_x), **kwargs)[
            0, :, :
        ]
    else:
        fourier_voxel_grid = voxel_grid
        surface = map_coordinates(fourier_voxel_grid, (k_z, k_y, k_x), **kwargs)[0, :, :]
    if is_rfft:
        assert sign is not None
        surface = jnp.where(sign[0, :, :] < 0, jnp.conj(surface), surface)
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
    cls,
    real_voxel_grid: Array,
    pad_scale: float,
    apply_deconvolve: bool = False,
    use_rfft: bool = True,
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

    transform = rfftn(real_voxel_grid_p) if use_rfft else fftn(real_voxel_grid_p)
    return _prepare_fourier_voxel_arguments(transform, use_rfft=use_rfft)


def _prepare_fourier_voxel_arguments(
    fourier_voxel_grid: Array, use_rfft: bool = True
) -> tuple[Array, Array]:
    dim = fourier_voxel_grid.shape[0]
    if use_rfft:
        # Truncated (last) axis stays in rfft/corner convention -- only the
        # two full axes get fftshift'd to center convention.
        phase = make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
        shifted = jnp.fft.fftshift(phase * fourier_voxel_grid, axes=(0, 1))
        return shifted, make_frequency_slice((dim, dim), fftshifted=True)
    return (
        jnp.fft.fftshift(make_fftshift_phase((dim, dim, dim)) * fourier_voxel_grid),
        make_frequency_slice((dim, dim), fftshifted=True),
    )
