"""
Voxel-based representations of the scattering potential.
"""

from abc import abstractmethod
from functools import cached_property
from typing import ClassVar, Optional, cast
from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox import AbstractClassVar, AbstractVar, field
from jaxtyping import Array, Complex, Float

from ...coordinates import make_coordinate_grid, make_frequency_slice
from ...internal import error_if_not_positive
from ...ndimage import (
    compute_spline_coefficients,
    crop_to_shape,
    fftn,
    pad_to_shape,
)
from ...ndimage.transforms import AbstractFilter
from .._pose import AbstractPose
from .base_potential import AbstractPotentialRepresentation


class AbstractVoxelPotential(AbstractPotentialRepresentation, strict=True):
    """Abstract interface for a voxel-based scattering potential representation."""

    voxel_size: AbstractVar[Float[Array, ""]]
    is_real_space: AbstractClassVar[bool]

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the voxel array."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    ) -> Self:
        """Load an `AbstractVoxelPotential` from real-valued 3D electron
        scattering potential.
        """
        raise NotImplementedError


# Not public API
class AbstractFourierVoxelGridPotential(AbstractVoxelPotential, strict=True):
    """Abstract interface of a 3D scattering potential voxel grid
    in fourier-space.
    """

    frequency_slice_in_pixels: AbstractVar[Float[Array, "1 dim dim 3"]]

    @abstractmethod
    def __init__(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""] | float,
    ):
        raise NotImplementedError


class FourierVoxelGridPotential(AbstractFourierVoxelGridPotential):
    """A 3D scattering potential voxel grid in fourier-space."""

    fourier_voxel_grid: Complex[Array, "dim dim dim"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real_space: ClassVar[bool] = False

    @override
    def __init__(
        self,
        fourier_voxel_grid: Complex[Array, "dim dim dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `fourier_voxel_grid`: The cubic voxel grid in fourier space.
        - `frequency_slice_in_pixels`: Frequency slice coordinate system.
        - `voxel_size`: The voxel size.
        """
        self.fourier_voxel_grid = jnp.asarray(fourier_voxel_grid)
        self.frequency_slice_in_pixels = frequency_slice_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `fourier_voxel_grid`."""
        return cast(tuple[int, int, int], self.fourier_voxel_grid.shape)

    @cached_property
    def frequency_slice_in_angstroms(self) -> Float[Array, "1 dim dim 3"]:
        """The `frequency_slice_in_pixels` in angstroms."""
        return _safe_multiply_grid_by_constant(
            self.frequency_slice_in_pixels, 1 / self.voxel_size
        )

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated `frequency_slice_in_pixels`."""
        return eqx.tree_at(
            lambda d: d.frequency_slice_in_pixels,
            self,
            pose.rotate_coordinates(self.frequency_slice_in_pixels, inverse=True),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
        filter: Optional[AbstractFilter] = None,
    ) -> Self:
        """Load an `AbstractFourierVoxelGridPotential` from real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `pad_scale`: Scale factor at which to pad `real_voxel_grid` before fourier
                     transform. Must be a value greater than `1.0`.
        - `pad_mode`: Padding method. See `jax.numpy.pad` for documentation.
        - `filter`: A filter to apply to the result of the fourier transform of
                  `real_voxel_grid`, i.e. `fftn(real_voxel_grid)`. Note that the zero
                  frequency component is assumed to be in the corner.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
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
        # Load potential and coordinates. For now, do not store the
        # fourier potential only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used.
        fourier_voxel_grid_with_zero_in_corner = (
            fftn(padded_real_voxel_grid)
            if filter is None
            else filter(fftn(padded_real_voxel_grid))
        )
        # ... store the potential grid with the zero frequency component in the center
        fourier_voxel_grid = jnp.fft.fftshift(fourier_voxel_grid_with_zero_in_corner)
        # ... create in-plane frequency slice on the half space
        frequency_slice = make_frequency_slice(
            cast(tuple[int, int], padded_real_voxel_grid.shape[:-1]),
            outputs_rfftfreqs=False,
        )

        return cls(fourier_voxel_grid, frequency_slice, voxel_size)


class FourierVoxelSplinePotential(AbstractFourierVoxelGridPotential):
    """A 3D scattering potential voxel grid in fourier-space, represented
    by spline coefficients.
    """

    spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim"]
    frequency_slice_in_pixels: Float[Array, "1 dim dim 3"]
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        spline_coefficients: Complex[Array, "coeff_dim coeff_dim coeff_dim"],
        frequency_slice_in_pixels: Float[Array, "1 dim dim 3"],
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `spline_coefficients`:
            The spline coefficents computed from the cubic voxel grid
            in fourier space. See `cryojax.image.compute_spline_coefficients`.
        - `frequency_slice_in_pixels`:
            Frequency slice coordinate system.
            See `cryojax.coordinates.make_frequency_slice`.
        - `voxel_size`: The voxel size.
        """
        self.spline_coefficients = spline_coefficients
        self.frequency_slice_in_pixels = frequency_slice_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the original `fourier_voxel_grid` from which
        `coefficients` were computed.
        """
        return cast(
            tuple[int, int, int], tuple([s - 2 for s in self.spline_coefficients.shape])
        )

    @cached_property
    def frequency_slice_in_angstroms(self) -> Float[Array, "1 dim dim 3"]:
        """The `frequency_slice_in_pixels` in angstroms."""
        return _safe_multiply_grid_by_constant(
            self.frequency_slice_in_pixels, 1 / self.voxel_size
        )

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated `frequency_slice_in_pixels`."""
        return eqx.tree_at(
            lambda d: d.frequency_slice_in_pixels,
            self,
            pose.rotate_coordinates(self.frequency_slice_in_pixels, inverse=True),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        pad_scale: float = 1.0,
        pad_mode: str = "constant",
        filter: Optional[AbstractFilter] = None,
    ) -> Self:
        """Load an `AbstractFourierVoxelGridPotential` from real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: A scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `pad_scale`: Scale factor at which to pad `real_voxel_grid` before fourier
                     transform. Must be a value greater than `1.0`.
        - `pad_mode`: Padding method. See `jax.numpy.pad` for documentation.
        - `filter`: A filter to apply to the result of the fourier transform of
                  `real_voxel_grid`, i.e. `fftn(real_voxel_grid)`. Note that the zero
                  frequency component is assumed to be in the corner.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
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
        # Load potential and coordinates. For now, do not store the
        # fourier potential only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used.
        fourier_voxel_grid_with_zero_in_corner = (
            fftn(padded_real_voxel_grid)
            if filter is None
            else filter(fftn(padded_real_voxel_grid))
        )
        # ... store the potential grid with the zero frequency component in the center
        fourier_voxel_grid = jnp.fft.fftshift(fourier_voxel_grid_with_zero_in_corner)
        # ... compute spline coefficients
        spline_coefficients = compute_spline_coefficients(jnp.asarray(fourier_voxel_grid))
        # ... create in-plane frequency slice on the half space
        frequency_slice = make_frequency_slice(
            cast(tuple[int, int], padded_real_voxel_grid.shape[:-1]),
            outputs_rfftfreqs=False,
        )

        return cls(spline_coefficients, frequency_slice, voxel_size)


class RealVoxelGridPotential(AbstractVoxelPotential, strict=True):
    """Abstraction of a 3D scattering potential voxel grid in real-space."""

    real_voxel_grid: Float[Array, "dim dim dim"]
    coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"]
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real_space: ClassVar[bool] = True

    def __init__(
        self,
        real_voxel_grid: Float[Array, "dim dim dim"],
        coordinate_grid_in_pixels: Float[Array, "dim dim dim 3"],
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `real_voxel_grid`: The voxel grid in fourier space.
        - `coordinate_grid_in_pixels`: A coordinate grid.
        - `voxel_size`: The voxel size.
        """
        self.real_voxel_grid = jnp.asarray(real_voxel_grid)
        self.coordinate_grid_in_pixels = coordinate_grid_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the `real_voxel_grid`."""
        return cast(tuple[int, int, int], self.real_voxel_grid.shape)

    @cached_property
    def coordinate_grid_in_angstroms(self) -> Float[Array, "dim dim dim 3"]:
        """The `coordinate_grid_in_pixels` in angstroms."""
        return _safe_multiply_grid_by_constant(
            self.coordinate_grid_in_pixels, self.voxel_size
        )

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated
        `coordinate_grid_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_grid_in_pixels,
            self,
            pose.rotate_coordinates(self.coordinate_grid_in_pixels, inverse=False),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        coordinate_grid_in_pixels: Optional[Float[Array, "dim dim dim 3"]] = None,
        crop_scale: Optional[float] = None,
    ) -> Self:
        """Load a `RealVoxelGridPotential` from a real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: An electron scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `crop_scale`: Scale factor at which to crop `real_voxel_grid`.
                        Must be a value greater than `1`.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
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

        return cls(real_voxel_grid, coordinate_grid_in_pixels, voxel_size)


class RealVoxelCloudPotential(AbstractVoxelPotential, strict=True):
    """Abstraction of a 3D electron scattering potential voxel point cloud.

    !!! info
        This object is similar to the `RealVoxelGridPotential`. Instead
        of storing the whole voxel grid, a `RealVoxelCloudPotential` need
        only store points of non-zero scattering potential. Therefore,
        a `RealVoxelCloudPotential` stores a point cloud of scattering potential
        voxel values. Instantiating with the `from_real_voxel_grid` constructor
        will automatically mask points of zero scattering potential.
    """

    voxel_weights: Float[Array, " size"]
    coordinate_list_in_pixels: Float[Array, "size 3"]
    voxel_size: Float[Array, ""] = field(converter=error_if_not_positive)

    is_real_space: ClassVar[bool] = True

    def __init__(
        self,
        voxel_weights: Float[Array, " size"],
        coordinate_list_in_pixels: Float[Array, "size 3"],
        voxel_size: Float[Array, ""] | float,
    ):
        """**Arguments:**

        - `voxel_weights`: A point-cloud of voxel scattering potential values.
        - `coordinate_list_in_pixels`: Coordinate list for the `voxel_weights`.
        - `voxel_size`: The voxel size.
        """
        self.voxel_weights = jnp.asarray(voxel_weights)
        self.coordinate_list_in_pixels = coordinate_list_in_pixels
        self.voxel_size = jnp.asarray(voxel_size)

    @property
    def shape(self) -> tuple[int]:
        """The shape of `voxel_weights`."""
        return cast(tuple[int], self.voxel_weights.shape)

    @cached_property
    def coordinate_list_in_angstroms(self) -> Float[Array, "size 3"]:
        """The `coordinate_list_in_pixels` in angstroms."""
        return _safe_multiply_list_by_constant(
            self.coordinate_list_in_pixels, self.voxel_size
        )

    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with a rotated
        `coordinate_list_in_pixels`.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_list_in_pixels,
            self,
            pose.rotate_coordinates(self.coordinate_list_in_pixels, inverse=False),
        )

    @classmethod
    def from_real_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
        *,
        coordinate_grid_in_pixels: Optional[Float[Array, "dim dim dim 3"]] = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        size: Optional[int] = None,
        fill_value: Optional[float] = None,
    ) -> Self:
        """Load an `RealVoxelCloudPotential` from a real-valued 3D electron
        scattering potential voxel grid.

        **Arguments:**

        - `real_voxel_grid`: An electron scattering potential voxel grid in real space.
        - `voxel_size`: The voxel size of `real_voxel_grid`.
        - `rtol`: Argument passed to `jnp.isclose`, used for masking
                  voxels of zero scattering potential.
        - `atol`: Argument passed to `jnp.isclose`, used for masking
                  voxels of zero scattering potential.
        - `size`: Argument passed to `jnp.where`, used for fixing the size
                  of the masked scattering potential. This argument is required
                  for using this function with a JAX transformation.
        - `fill_value`: Argument passed to `jnp.where`, used if `size` is specified and
                        the mask has fewer than the indicated number of elements.
        """
        # Cast to jax array
        real_voxel_grid, voxel_size = (
            jnp.asarray(real_voxel_grid),
            jnp.asarray(voxel_size),
        )
        # Make coordinates if not given
        if coordinate_grid_in_pixels is None:
            coordinate_grid_in_pixels = make_coordinate_grid(real_voxel_grid.shape)
        # ... mask zeros to store smaller arrays. This
        # option is not jittable.
        nonzero = jnp.where(
            ~jnp.isclose(real_voxel_grid, 0.0, rtol=rtol, atol=atol),
            size=size,
            fill_value=fill_value,
        )
        flat_potential = real_voxel_grid[nonzero]
        coordinate_list = coordinate_grid_in_pixels[nonzero]

        return cls(flat_potential, coordinate_list, voxel_size)


def _safe_multiply_grid_by_constant(
    grid: Float[Array, "z_dim y_dim x_dim 3"], constant: Float[Array, ""]
) -> Float[Array, "z_dim y_dim x_dim 3"]:
    """Multiplies a coordinate grid by a constant in a
    safe way for gradient computation.
    """
    return jnp.where(grid != 0.0, jnp.asarray(constant) * grid, 0.0)


def _safe_multiply_list_by_constant(
    coordinate_list: Float[Array, "size 3"], constant: Float[Array, ""]
) -> Float[Array, "size 3"]:
    """Multiplies a coordinate grid by a constant in a
    safe way for gradient computation.
    """
    return jnp.where(coordinate_list != 0.0, jnp.asarray(constant) * coordinate_list, 0.0)
