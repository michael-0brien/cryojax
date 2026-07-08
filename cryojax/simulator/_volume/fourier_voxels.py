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
    enforce_rfftn_self_conjugates,
    fftn,
    ifftn,
    irfftn,
    make_1d_coordinate_grid,
    make_1d_frequency_grid,
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
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            fourier_projection = _extract_slice(
                volume_representation.fourier_voxel_grid,
                frequency_slice,
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
        # The Ewald sphere surface curves the in-plane slice out of its own
        # plane, so unlike `FourierSliceExtraction`, its output isn't
        # Hermitian-symmetric as a whole and every output pixel is queried
        # independently -- reconstruct the full in-plane grid from the
        # stored half one before curving.
        full_frequency_slice = _reconstruct_full_slice_from_half_slice(frequency_slice)
        # Compute the fourier projection
        if isinstance(volume_representation, FourierVoxelSplineVolume):
            ewald_sphere_surface = _extract_ewald_sphere_spline(
                volume_representation.spline_coefficients,
                full_frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
                out_of_bounds_mode=self.out_of_bounds_mode,
                unroll_gather=self.unroll_gather,
            )
        elif isinstance(volume_representation, FourierVoxelGridVolume):
            ewald_sphere_surface = _extract_ewald_sphere(
                volume_representation.fourier_voxel_grid,
                full_frequency_slice,
                image_config.pixel_size,
                image_config.wavelength_in_angstroms,
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
    N = frequency_slice.shape[1]
    surface = _extract_surface_from_voxel_grid(
        fourier_voxel_grid,
        frequency_slice,
        is_spline_coefficients=False,
        **kwargs,
    )
    # `surface` is already rfft-shaped (the query grid itself was), so only
    # self-conjugate (DC/Nyquist) realness needs enforcing here -- no crop.
    return enforce_rfftn_self_conjugates(surface, (N, N), includes_dc=False, mode="zero")


def _extract_slice_spline(
    spline_coefficients: Array, frequency_slice: Array, **kwargs: Any
) -> Complex[Array, "dim dim//2+1"]:
    N = frequency_slice.shape[1]
    surface = _extract_surface_from_voxel_grid(
        spline_coefficients, frequency_slice, is_spline_coefficients=True, **kwargs
    )
    return enforce_rfftn_self_conjugates(surface, (N, N), includes_dc=False, mode="zero")


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


def _reconstruct_full_slice_from_half_slice(
    half_slice: Float[Array, "1 dim dim//2+1 3"],
) -> Float[Array, "1 dim dim 3"]:
    """Reconstruct the full in-plane frequency grid from the half (rfft) one
    stored on the volume, for `EwaldSphereExtraction`, which needs the full
    grid to compute its local `xhat`/`yhat`/`zhat` basis and to produce its
    curved, non-Hermitian-symmetric-as-a-whole output surface.

    A rotated in-plane grid is an exactly linear function of the (unrotated)
    local x-coordinate, for any fixed y: `slice(x, y) = x * xhat_rot +
    slice(0, y)`, where `xhat_rot` is a single constant vector (the rotated
    local x unit vector). `xhat_rot` is recovered from any two adjacent
    columns of the half grid, then used to extrapolate every column of the
    full grid. This is exact (no interpolation, and no reflection of array
    indices, so no boundary case at the row-Nyquist frequency -- unlike a
    literal point-reflection, this only ever reads columns that are already
    stored in `half_slice`).
    """
    N = half_slice.shape[1]
    xhat_rot = N * (half_slice[:, :, 1, :] - half_slice[:, :, 0, :])
    x_full = make_1d_frequency_grid(N, outputs_rfftfreqs=False, fftshifted=True)
    x_term = x_full[None, None, :, None] * xhat_rot[:, :, None, :]
    return half_slice[:, :, 0:1, :] + x_term


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
    # `voxel_grid`'s last axis only stores non-negative frequencies along x
    # (i.e. `F(-q) = conj(F(q))` is not stored, only `F(q)` for `q_x >= 0`).
    # Reflect the whole 3-vector through the origin whenever `q_x < 0`, so we
    # always look up a point with `q_x >= 0`, then conjugate the interpolated
    # result to correct for it. This is exact, not an approximation: it's
    # evaluated once per query point, on the continuous coordinate, before
    # any interpolation taps are generated, so taps never straddle the
    # truncation boundary.
    sign = jnp.where(frequency_coordinates[..., 0] < 0, -1.0, 1.0)
    reflected = sign[..., None] * frequency_coordinates
    k_x = reflected[..., 0] * N  # rfft/corner convention: no N // 2 offset
    k_y = reflected[..., 1] * N + N // 2
    k_z = reflected[..., 2] * N + N // 2
    # The centered axes' Nyquist bin (frequency -0.5) is stored only at
    # index 0, not also at index N -- +0.5 and -0.5 are the same (aliased)
    # physical frequency. Reflecting a coordinate that was exactly at -0.5
    # lands exactly on index N, one past the valid range; wrap that exact
    # case back to index 0, without touching any other (genuinely
    # out-of-bounds) coordinate.
    k_y = jnp.where(k_y == N, 0.0, k_y)
    k_z = jnp.where(k_z == N, 0.0, k_z)
    if is_spline_coefficients:
        spline_coefficients = voxel_grid
        surface = map_coordinates_spline(spline_coefficients, (k_z, k_y, k_x), **kwargs)[
            0, :, :
        ]
    else:
        fourier_voxel_grid = voxel_grid
        surface = map_coordinates(fourier_voxel_grid, (k_z, k_y, k_x), **kwargs)[0, :, :]
    surface = jnp.where(sign[0, :, :] < 0, jnp.conj(surface), surface)
    # FFT shift and multiply by (-1)^k phase factors. `surface` is itself
    # rfft-shaped only when `frequency_coordinates` was (i.e. only for
    # `FourierSliceExtraction`'s half in-plane slice -- `EwaldSphereExtraction`
    # always reconstructs a full grid before calling this function), in which
    # case only the first axis is shifted, mirroring the same convention used
    # for the 3D volume storage.
    if surface.shape[0] == surface.shape[1]:
        surface = jnp.fft.ifftshift(make_fftshift_phase(surface.shape) * surface)
    else:
        surface = jnp.fft.ifftshift(
            make_fftshift_phase((N, N), outputs_rfft=True) * surface, axes=(0,)
        )

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

    return _prepare_fourier_voxel_arguments(rfftn(real_voxel_grid_p))


def _prepare_fourier_voxel_arguments(fourier_voxel_grid: Array) -> tuple[Array, Array]:
    dim = fourier_voxel_grid.shape[0]
    # Only the kept (non-negative local-x) half of the in-plane slice is
    # ever needed: `FourierSliceExtraction`'s output is itself rfft-shaped,
    # and `EwaldSphereExtraction` reconstructs the full grid on demand from
    # this half (see `_reconstruct_full_slice_from_half_slice`). This halves
    # the cost of rotating the slice to a pose.
    frequency_slice = make_frequency_slice(
        (dim, dim), outputs_rfftfreqs=True, fftshifted=True
    )
    # Truncated (last) axis stays in rfft/corner convention -- only the
    # two full axes get fftshift'd to center convention.
    phase = make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
    shifted = jnp.fft.fftshift(phase * fourier_voxel_grid, axes=(0, 1))
    return shifted, frequency_slice
