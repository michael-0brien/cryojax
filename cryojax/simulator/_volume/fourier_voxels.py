"""
Fourier voxel-based representations of a volume.
"""

from typing import ClassVar, Literal, Self
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import NDArrayLike
from ...ndimage import (
    enforce_rfftn_self_conjugates,
    ewald_sphere_from_slice,
    make_frequency_slice,
    prepare_sampling_fft,
    resize_with_crop_or_pad,
    sample_fft_slice,
)
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import (
    AbstractVolumeIntegrator,
    AbstractVoxelVolume,
    EwaldSphereArray,
    ProjectionArray,
)


def _check_voxel_array_shape(shape: tuple[int, ...], cls_name: str) -> None:
    """Validate that `values.shape` is consistent with a cubic, even-dimension
    volume stored as a half-space RFFT grid (last axis truncated to
    `dim // 2 + 1`).
    """
    d0, d1, d2 = shape
    if d0 % 2 == 1:
        raise ValueError(
            f"`{cls_name}` does not support odd voxel map dimensions, but got "
            f"a voxel map with `values.shape = {shape}`. Please pass "
            "a voxel map with even dimensions."
        )
    expected_d2 = d0 // 2 + 1
    if d1 != d0 or d2 != expected_d2:
        expected_shape = (d0, d0, expected_d2)
        # Common misuse: the array is a valid *full* (non-rfft) voxel grid,
        # e.g. the output of `fftn` rather than `rfftn`.
        if d1 == d0 and d2 == d0:
            raise AttributeError(
                f"`values` passed to `{cls_name}` has shape `{shape}`, "
                f"which is likely the full (non-rfft) FFT grid shape. Expected the "
                f"half-space RFFT grid shape `{expected_shape}` -- did you "
                "mean to pass `jax.numpy.fft.rfftn(real_voxel_grid)` "
                "instead of `jax.numpy.fft.fftn(real_voxel_grid)`?"
            )
        raise AttributeError(
            f"`values` passed to `{cls_name}` has an invalid shape "
            f"`{shape}`. Expected shape `{expected_shape}`."
        )


class FourierVoxelGridVolume(AbstractVoxelVolume, strict=True):
    """A volume representation for a 3D voxel grid in fourier-space.

    !!! note
        Prefer the class-method constructor `from_real_voxel_grid` over direct
        instantiation. This prepares values for interpolation; only use `__init__`
        assumes if more control is desired.
    """  # noqa: E501

    values: Complex[Array, "dim dim dim//2+1"]
    frequency_slice: Float[Array, "1 dim dim//2+1 3"]
    interp: Literal["linear", "cubic"] = eqx.field(static=True)

    is_frame_rotation: ClassVar[bool] = True

    def __init__(
        self,
        values: Complex[NDArrayLike, "dim dim dim//2+1"],
        frequency_slice: Float[NDArrayLike, "1 dim dim//2+1 3"],
        interp: Literal["linear", "cubic"] = "linear",
    ):
        """**Arguments:**

        - `values`:
            The cubic voxel grid in fourier space, truncated to the half-space
            `(dim, dim, dim // 2 + 1)` and already prepared for interpolation by
            [`cryojax.ndimage.prepare_sampling_fft`][].
        - `frequency_slice`:
            The frequency slice coordinate system, in pixel units. This should be
            the output of [`cryojax.ndimage.make_frequency_slice`][].
        - `interp`:
            The interpolation method used for fourier slice extraction, either
            `"linear"` (the default) or `"cubic"`. This should be the same value
            passed to [`cryojax.ndimage.prepare_sampling_fft`][].
        """
        self.values = jnp.asarray(values, dtype=complex)
        _check_voxel_array_shape(self.values.shape, cls_name=type(self).__name__)
        self.frequency_slice = jnp.asarray(frequency_slice, dtype=float)
        self.interp = interp

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new volume with a rotated `frequency_slice`."""
        return eqx.tree_at(
            lambda d: d.frequency_slice,
            self,
            pose.rotate_coordinates(self.frequency_slice, inverse=self.is_frame_rotation),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The cubic shape of the volume in real-space."""
        dim = self.values.shape[0]
        return (dim, dim, dim)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        /,
        *,
        interp: Literal["linear", "cubic"] = "linear",
        pad_scale: float = 1.0,
    ) -> Self:
        """Load from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`:
            A voxel grid in real space.
        - `interp`:
            The interpolation method used for fourier slice extraction, either
            `"linear"` (the default) or `"cubic"`. The corresponding
            interpolation kernel is deconvolved out of the voxel grid here,
            which is what makes slice extraction accurate --- see
            [`cryojax.ndimage.prepare_sampling_fft`][].
        - `pad_scale`:
            Scale factor at which to pad `real_voxel_grid` before fourier
            transform. Must be a value greater than `1.0`.
        """
        fourier_voxel_grid = prepare_sampling_fft(
            jnp.asarray(real_voxel_grid, dtype=float),
            interp=interp,
            pad_scale=pad_scale,
        )
        dim = fourier_voxel_grid.shape[0]
        frequency_slice = make_frequency_slice((dim, dim), fftshifted=True)

        return cls(fourier_voxel_grid, frequency_slice, interp=interp)


class FourierSliceExtraction(
    AbstractVolumeIntegrator[FourierVoxelGridVolume],
    strict=True,
):
    """Integrate points to the exit plane using the Fourier
    projection-slice theorem.

    The interpolation method is read from `FourierVoxelGridVolume.interp`.
    """

    boundary: str
    unroll: bool | Literal["auto"]

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        boundary: str = "fill",
        unroll: bool | Literal["auto"] = "auto",
    ):
        """**Arguments:**

        - `boundary`:
            What to return for frequencies outside the fourier box. See
            `cryojax.ndimage.sample_fft_slice`.
        - `unroll`:
            Passed to `cryojax.ndimage.sample_fft_slice`. With `"auto"` (the
            default), this is `True` for `interp="cubic"` and `False` otherwise.
        """
        self.boundary = boundary
        self.unroll = unroll

    @override
    def integrate(
        self,
        volume_representation: FourierVoxelGridVolume,
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
        if not isinstance(volume_representation, FourierVoxelGridVolume):
            raise ValueError(
                "Got unsupported type for `volume_representation` in "
                "`FourierSliceExtraction.integrate`. Expected "
                "`FourierVoxelGridVolume`, but got "
                f"`{volume_representation.__class__.__name__}`."
            )
        frequency_slice = volume_representation.frequency_slice
        N = frequency_slice.shape[1]
        # Compute the fourier projection
        fourier_projection = sample_fft_slice(
            volume_representation.values,
            frequency_slice,
            interp=volume_representation.interp,
            boundary=self.boundary,
            unroll=_resolve_unroll(self.unroll, volume_representation.interp),
        )
        # The extracted half-slice is already rfft-shaped (the query grid
        # itself was), so only self-conjugate (DC/Nyquist) realness needs
        # enforcing here -- no crop.
        fourier_projection = enforce_rfftn_self_conjugates(
            fourier_projection, (N, N), includes_dc=False, mode="zero"
        )

        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != (N, N):
            fourier_projection = jnp.fft.rfftn(
                resize_with_crop_or_pad(
                    jnp.fft.irfftn(fourier_projection, s=(N, N)),
                    image_config.padded_shape,
                )
            )
        # Scale by voxel size to convert from projection to integral
        fourier_projection *= image_config.pixel_size
        return (
            jnp.fft.irfftn(fourier_projection, s=image_config.padded_shape)
            if outputs_real_space
            else fourier_projection
        )


class EwaldSphereExtraction(
    AbstractVolumeIntegrator[FourierVoxelGridVolume],
    strict=True,
):
    """Integrate points to the exit plane by extracting a surface of
    the ewald sphere in fourier space.

    The interpolation method is read from `FourierVoxelGridVolume.interp`.
    """

    boundary: str
    unroll: bool | Literal["auto"]

    outputs_ewald_sphere: ClassVar[bool] = True

    def __init__(
        self,
        *,
        boundary: str = "fill",
        unroll: bool | Literal["auto"] = "auto",
    ):
        """**Arguments:**

        - `boundary`:
            What to return for frequencies outside the fourier box. See
            `cryojax.ndimage.sample_fft_slice`.
        - `unroll`:
            Passed to `cryojax.ndimage.sample_fft_slice`. With `"auto"` (the
            default), this is `True` for `interp="cubic"` and `False` otherwise.
        """
        self.boundary = boundary
        self.unroll = unroll

    @override
    def integrate(
        self,
        volume_representation: FourierVoxelGridVolume,
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
        if not isinstance(volume_representation, FourierVoxelGridVolume):
            raise ValueError(
                "Got unsupported type for `volume_representation` in "
                "`EwaldSphereExtraction.integrate`. Expected "
                "`FourierVoxelGridVolume`, but got "
                f"`{volume_representation.__class__.__name__}`."
            )
        frequency_slice = volume_representation.frequency_slice
        N = frequency_slice.shape[1]
        if volume_representation.shape != (N, N, N):
            raise AttributeError(
                "Only cubic boxes are supported for fourier slice extraction."
            )
        # The Ewald sphere surface curves the in-plane slice out of its own
        # plane, so unlike `FourierSliceExtraction`, its output isn't
        # Hermitian-symmetric as a whole and every output pixel is queried
        # independently. `ewald_sphere_from_slice` reconstructs the full
        # in-plane grid from the stored half one before curving.
        ewald_sphere_frequencies = ewald_sphere_from_slice(
            frequency_slice,
            image_config.pixel_size,
            image_config.wavelength_in_angstroms,
        )
        # Compute the fourier projection
        ewald_sphere_surface = sample_fft_slice(
            volume_representation.values,
            ewald_sphere_frequencies,
            interp=volume_representation.interp,
            boundary=self.boundary,
            unroll=_resolve_unroll(self.unroll, volume_representation.interp),
        )

        # Resize the image to match the AbstractImageConfig.padded_shape
        if image_config.padded_shape != (N, N):
            ewald_sphere_surface = jnp.fft.fftn(
                resize_with_crop_or_pad(
                    jnp.fft.ifftn(ewald_sphere_surface, s=(N, N)),
                    image_config.padded_shape,
                )
            )
        # Scale by voxel size to convert from projection to integral
        ewald_sphere_surface *= image_config.pixel_size
        return (
            jnp.fft.irfftn(ewald_sphere_surface, s=image_config.padded_shape)
            if outputs_real_space
            else ewald_sphere_surface
        )


def _resolve_unroll(unroll: bool | Literal["auto"], interp: str) -> bool:
    """Resolve `unroll="auto"`: unroll the gather for the cubic kernel's `4^3`
    neighborhood, and use the single consolidated gather otherwise.
    """
    if unroll == "auto":
        return interp == "cubic"
    return unroll
