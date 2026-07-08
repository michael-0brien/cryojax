"""
Fourier voxel-based representations of a volume.
"""

import abc
from typing import ClassVar, Self
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import NDArrayLike
from ...ndimage import (
    central_slice_to_ewald_sphere,
    compute_spline_coefficients,
    enforce_rfftn_self_conjugates,
    fftn,
    ifftn,
    irfftn,
    make_fftshift_phase,
    make_frequency_slice,
    prepare_sampling_rfft,
    resize_with_crop_or_pad,
    rfftn,
    sample_rfft_surface,
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

    frequency_slice_in_pixels: eqx.AbstractVar[Float[Array, "1 dim dim//2+1 3"]]

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
    padding: int,
    cls_name: str,
    array_name: str,
) -> None:
    """Validate that `shape` (the raw stored array shape, e.g.
    `fourier_voxel_grid.shape` or `spline_coefficients.shape`) is consistent
    with a cubic, even-dimension volume stored as a half-space RFFT grid
    (last axis truncated to `dim // 2 + 1`), given `padding` (extra samples
    added per axis, e.g. `2` for cubic-spline coefficients).
    """
    d0, d1, d2 = (s - padding for s in shape)
    if d0 % 2 == 1:
        raise ValueError(
            f"`{cls_name}` does not support odd voxel map dimensions, but got "
            f"a voxel map with `{array_name}.shape = {shape}`. Please pass a "
            "voxel map with even dimensions."
        )
    expected_d2 = d0 // 2 + 1
    if d1 != d0 or d2 != expected_d2:
        expected_shape = tuple(s + padding for s in (d0, d0, expected_d2))
        # Common misuse: the array is a valid *full* (non-rfft) voxel grid,
        # e.g. the output of `fftn` rather than `rfftn`.
        if d1 == d0 and d2 == d0:
            raise AttributeError(
                f"`{array_name}` passed to `{cls_name}` has shape `{shape}`, "
                f"which is likely the full (non-rfft) FFT grid shape. Expected the "
                f"half-space RFFT grid shape `{expected_shape}` -- did you "
                "mean to pass `cryojax.ndimage.rfftn(real_voxel_grid)` "
                "instead of `cryojax.ndimage.fftn(real_voxel_grid)`?"
            )
        raise AttributeError(
            f"`{array_name}` passed to `{cls_name}` has an invalid shape "
            f"`{shape}`. Expected shape `{expected_shape}`."
        )


class FourierVoxelGridVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space.

    !!! note
        Prefer the class-method constructors over direct instantiation
        via ` volume = FourierVoxelGridVolume(...)`:

        - `from_real_voxel_grid`:
            Instantiate from a real-space map.
        - `from_fourier_voxel_grid`:
            Instantiate from the output of `cryojax.ndimage.rfftn`.

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
        phase = im.make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
        fourier_voxel_grid = jnp.fft.fftshift(phase * im.rfftn(real_voxel_grid), axes=(0, 1))
        frequency_slice = im.make_frequency_slice(
            (dim, dim), outputs_rfftfreqs=True, fftshifted=True
        )
        ```
    """  # noqa: E501

    fourier_voxel_grid: Complex[Array, "dim dim dim//2+1"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim//2+1 3"]

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        fourier_voxel_grid: Complex[NDArrayLike, "dim dim dim//2+1"],
        frequency_slice_in_pixels: Float[NDArrayLike, "1 dim dim//2+1 3"],
    ):
        """**Arguments:**

        - `fourier_voxel_grid`:
            The cubic voxel grid in fourier space, truncated to the
            half-space `(dim, dim, dim // 2 + 1)`, as returned by
            `cryojax.ndimage.rfftn`.
        - `frequency_slice_in_pixels`:
            The frequency slice coordinate system.
        """
        # Multiply by phase correction for interpolation logic
        self.fourier_voxel_grid = jnp.asarray(fourier_voxel_grid, dtype=complex)
        _check_voxel_array_shape(
            self.fourier_voxel_grid.shape,
            padding=0,
            cls_name=type(self).__name__,
            array_name="fourier_voxel_grid",
        )
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The cubic shape of the volume in real-space."""
        dim = self.fourier_voxel_grid.shape[0]
        return (dim, dim, dim)

    @classmethod
    def from_fourier_voxel_grid(cls, fourier_voxel_grid: NDArrayLike) -> Self:
        """Load from a fourier-domain 3D voxel grid.

        This should be the output of

        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage import im

        fourier_voxel_grid = im.rfftn(real_voxel_grid)
        volume = cxs.FourierVoxelGridVolume.from_fourier_voxel_grid(fourier_voxel_grid)
        ```

        **Arguments:**

        - `fourier_voxel_grid`:
            A voxel grid in fourier space, the output of `cryojax.ndimage.rfftn`.
        """  # noqa: E501
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
        # Preprocess to fourier grid, deconvolving after any padding so that
        # the sinc² correction uses the actual Fourier grid size.
        fourier_voxel_grid = prepare_sampling_rfft(
            jnp.asarray(real_voxel_grid, dtype=float),
            apply_deconvolve=apply_deconvolve,
            pad_scale=pad_scale,
        )
        dim = fourier_voxel_grid.shape[0]
        frequency_slice = make_frequency_slice((dim, dim), fftshifted=True)

        return cls(fourier_voxel_grid, frequency_slice)


class FourierVoxelSplineVolume(AbstractFourierVoxelVolume, strict=True):
    """A 3D voxel grid in fourier-space, represented
    by spline coefficients.
    """

    spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim//2+1"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim//2+1 3"]

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        spline_coefficients: Complex[NDArrayLike, "coeff_dim coeff_dim coeff_dim//2+1"],
        frequency_slice_in_pixels: Float[NDArrayLike, "1 dim dim//2+1 3"],
    ):
        """**Arguments:**

        - `spline_coefficients`:
            The spline coefficents computed from the cubic voxel grid in
            fourier space, i.e. the half-space RFFT grid (last axis of size
            `dim // 2 + 1`, before the `+ 2` spline padding). See
            `cryojax.ndimage.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency slice coordinate system.
            See `cryojax.coordinates.make_frequency_slice`.
        """
        self.spline_coefficients = jnp.asarray(spline_coefficients, dtype=complex)
        _check_voxel_array_shape(
            self.spline_coefficients.shape,
            padding=2,
            cls_name=type(self).__name__,
            array_name="spline_coefficients",
        )
        self.frequency_slice_in_pixels = jnp.asarray(
            frequency_slice_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The cubic shape of the original real-space `fourier_voxel_grid`
        from which `coefficients` were computed.
        """
        dim = self.spline_coefficients.shape[0] - 2
        return (dim, dim, dim)

    @classmethod
    def from_fourier_voxel_grid(cls, fourier_voxel_grid: NDArrayLike) -> Self:
        """Load from a fourier-domain 3D voxel grid.

        This should be the output of

        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage import im

        fourier_voxel_grid = im.rfftn(real_voxel_grid)
        volume = cxs.FourierVoxelSplineVolume.from_fourier_voxel_grid(fourier_voxel_grid)
        ```

        **Arguments:**

        - `fourier_voxel_grid`:
            A voxel grid in fourier space, the output of `cryojax.ndimage.rfftn`.
        """  # noqa: E501
        fourier_voxel_grid, frequency_slice = _prepare_fourier_voxel_arguments(
            jnp.asarray(fourier_voxel_grid)
        )
        # Compute spline coefficients
        spline_coefficients = compute_spline_coefficients(fourier_voxel_grid)

        return cls(spline_coefficients, frequency_slice)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        pad_scale: float = 1.0,
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`:
            A voxel grid in real space.
        - `pad_scale`:
            Scale factor at which to pad `real_voxel_grid` before fourier
            transform. Must be a value greater than `1.0`.
        """
        # Preprocess to fourier grid and compute spline coefficients
        spline_coefficients = prepare_sampling_rfft(
            jnp.asarray(real_voxel_grid, dtype=float),
            pad_scale=pad_scale,
            use_spline=True,
        )
        dim = spline_coefficients.shape[0] - 2
        frequency_slice = make_frequency_slice((dim, dim), fftshifted=True)

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
            fourier_projection = sample_rfft_surface(
                volume_representation.spline_coefficients,
                frequency_slice,
                use_spline=True,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            fourier_projection = sample_rfft_surface(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
                use_spline=False,
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
        # The extracted half-slice is already rfft-shaped (the query grid
        # itself was), so only self-conjugate (DC/Nyquist) realness needs
        # enforcing here -- no crop.
        fourier_projection = enforce_rfftn_self_conjugates(
            fourier_projection, (N, N), includes_dc=False, mode="zero"
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
        # The Ewald sphere surface curves the in-plane slice out of its own
        # plane, so unlike `FourierSliceExtraction`, its output isn't
        # Hermitian-symmetric as a whole and every output pixel is queried
        # independently. `central_slice_to_ewald_sphere` reconstructs the full
        # in-plane grid from the stored half one before curving.
        ewald_sphere_frequencies = central_slice_to_ewald_sphere(
            frequency_slice,
            image_config.pixel_size,
            image_config.wavelength_in_angstroms,
        )
        # Compute the fourier projection
        if isinstance(volume_representation, FourierVoxelSplineVolume):
            ewald_sphere_surface = sample_rfft_surface(
                volume_representation.spline_coefficients,
                ewald_sphere_frequencies,
                use_spline=True,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            ewald_sphere_surface = sample_rfft_surface(
                volume_representation.fourier_voxel_grid,
                ewald_sphere_frequencies,
                use_spline=False,
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


def _prepare_fourier_voxel_arguments(fourier_voxel_grid: Array) -> tuple[Array, Array]:
    dim = fourier_voxel_grid.shape[0]
    # Only the kept (non-negative local-x) half of the in-plane slice is
    # ever needed: `FourierSliceExtraction`'s output is itself rfft-shaped,
    # and `EwaldSphereExtraction` reconstructs the full grid on demand from
    # this half (see `_reconstruct_full_slice_from_half_slice`). This halves
    # the cost of rotating the slice to a pose.
    frequency_slice = make_frequency_slice((dim, dim), fftshifted=True)
    # Truncated (last) axis stays in rfft/corner convention -- only the
    # two full axes get fftshift'd to center convention.
    phase = make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
    shifted = jnp.fft.fftshift(phase * fourier_voxel_grid, axes=(0, 1))
    return shifted, frequency_slice
