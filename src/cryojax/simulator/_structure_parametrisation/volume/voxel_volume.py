"""
Voxel-based representations of a structure.
"""

from typing import Optional, cast
from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ....coordinates import make_coordinate_grid, make_frequency_slice
from ....internal import NDArrayLike
from ....ndimage import (
    compute_spline_coefficients,
    crop_to_shape,
    fftn,
    pad_to_shape,
)
from ....ndimage.transforms import AbstractFilter
from ..._pose import AbstractPose
from .base_volume import AbstractVoxelVolume


class AbstractFourierVoxelVolume(AbstractVoxelVolume, strict=True):
    """Abstract interface for a voxel-based structure."""

    frequency_slice_in_pixels: eqx.AbstractVar[Float[Array, "1 dim dim 3"]]

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new structure with a rotated `frequency_slice_in_pixels`."""
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
        filter: Optional[AbstractFilter] = None,
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
        filter: Optional[AbstractFilter] = None,
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
        spline_coefficients = compute_spline_coefficients(jnp.asarray(fourier_voxel_grid))
        # ... create in-plane frequency slice on the half space
        frequency_slice = make_frequency_slice(
            cast(tuple[int, int], padded_real_voxel_grid.shape[:-1]),
            outputs_rfftfreqs=False,
        )

        return cls(spline_coefficients, frequency_slice)


class AbstractRealVoxelVolume(AbstractVoxelVolume, strict=True):
    """Abstract interface for a voxel-based structure."""

    coordinate_grid_in_pixels: eqx.AbstractVar[Float[Array, "dim dim dim 3"]]

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new structure with a rotated
        `coordinate_grid_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_grid_in_pixels,
            self,
            pose.rotate_coordinates(self.coordinate_grid_in_pixels, inverse=inverse),
        )


class RealVoxelGridVolume(AbstractRealVoxelVolume, strict=True):
    """A 3D voxel grid in real-space."""

    real_voxel_grid: Float[Array, "dim dim dim"]
    coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"]

    def __init__(
        self,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        coordinate_grid_in_pixels: Float[NDArrayLike, "dim dim dim 3"],
    ):
        """**Arguments:**

        - `real_voxel_grid`: The voxel grid in fourier space.
        - `coordinate_grid_in_pixels`: A coordinate grid.
        """
        self.real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        self.coordinate_grid_in_pixels = jnp.asarray(
            coordinate_grid_in_pixels, dtype=float
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `real_voxel_grid`."""
        return cast(tuple[int, int, int], self.real_voxel_grid.shape)

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
        *,
        coordinate_grid_in_pixels: Optional[Float[Array, "dim dim dim 3"]] = None,
        crop_scale: Optional[float] = None,
    ) -> Self:
        """Load a `RealVoxelGridVolume` from a real-valued 3D voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A voxel grid in real space.
        - `crop_scale`: Scale factor at which to crop `real_voxel_grid`.
                        Must be a value greater than `1`.
        """
        # Cast to jax array
        real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
        # Make coordinates if not given
        if coordinate_grid_in_pixels is None:
            # Option for cropping template
            if crop_scale is not None:
                if crop_scale < 1.0:
                    raise ValueError("`crop_scale` must be greater than 1.0")
                cropped_shape = cast(
                    tuple[int, int, int],
                    tuple([int(s / crop_scale) for s in real_voxel_grid.shape[-3:]]),
                )
                real_voxel_grid = crop_to_shape(real_voxel_grid, cropped_shape)
            coordinate_grid_in_pixels = make_coordinate_grid(real_voxel_grid.shape[-3:])

        return cls(real_voxel_grid, coordinate_grid_in_pixels)
