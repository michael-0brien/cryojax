"""The image configuration and utility manager."""

import math
from functools import cached_property
from typing import Any, Callable, Optional, TypedDict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from ..constants import convert_keV_to_angstroms
from ..coordinates import make_coordinate_grid, make_frequency_grid
from ..internal import error_if_not_positive
from ..ndimage import (
    crop_to_shape,
    pad_to_shape,
    resize_with_crop_or_pad,
)


class GridHelper(eqx.Module, strict=True):
    coordinate_grid: Float[Array, "y_dim x_dim 2"]
    frequency_grid: Float[Array, "y_dim x_dim//2+1 2"]
    full_frequency_grid: Optional[Float[Array, "y_dim x_dim 2"]]

    def __init__(self, shape: tuple[int, int], only_rfft: bool = True):
        self.coordinate_grid = make_coordinate_grid(shape)
        self.frequency_grid = make_frequency_grid(shape, outputs_rfftfreqs=True)
        if only_rfft:
            self.full_frequency_grid = None
        else:
            self.full_frequency_grid = make_frequency_grid(shape, outputs_rfftfreqs=False)


class PadOptions(TypedDict):
    shape: tuple[int, int]
    grid_helper: Optional[GridHelper]
    mode: str | Callable


class AbstractConfig(eqx.Module, strict=True):
    """Configuration and utilities for an electron microscopy image."""

    shape: eqx.AbstractVar[tuple[int, int]]
    pixel_size: eqx.AbstractVar[Float[Array, ""]]
    voltage_in_kilovolts: eqx.AbstractVar[Float[Array, ""]]

    grid_helper: eqx.AbstractVar[Optional[GridHelper]]
    pad_options: eqx.AbstractVar[PadOptions]

    def __check_init__(self):
        if self.padded_shape[0] < self.shape[0] or self.padded_shape[1] < self.shape[1]:
            raise AttributeError(
                "ImageConfig.padded_shape is less than ImageConfig.shape in one or "
                "more dimensions."
            )

    @property
    def wavelength_in_angstroms(self) -> Float[Array, ""]:
        """The incident electron wavelength corresponding to the beam
        energy `voltage_in_kilovolts`.
        """
        return convert_keV_to_angstroms(self.voltage_in_kilovolts)

    @property
    def wavenumber_in_inverse_angstroms(self) -> Float[Array, ""]:
        """The incident electron wavenumber corresponding to the beam
        energy `voltage_in_kilovolts`.
        """
        return 2 * jnp.pi / self.wavelength_in_angstroms

    @cached_property
    def coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """A spatial coordinate system for the `shape`."""
        if self.grid_helper is None:
            return make_coordinate_grid(self.shape)
        else:
            return self.grid_helper.coordinate_grid

    @cached_property
    def coordinate_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """Convenience property for `pixel_size * coordinate_grid_in_pixels`"""
        return _safe_multiply_by_constant(self.coordinate_grid_in_pixels, self.pixel_size)

    @cached_property
    def frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim//2+1} 2"]:
        """A spatial frequency coordinate system for the `shape`,
        with hermitian symmetry.
        """
        if self.grid_helper is None:
            return make_frequency_grid(self.shape, outputs_rfftfreqs=True)
        else:
            return self.grid_helper.frequency_grid

    @cached_property
    def frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim//2+1} 2"]:
        """Convenience property for `frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self.frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @cached_property
    def full_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """A spatial frequency coordinate system for the `shape`,
        without hermitian symmetry.
        """
        if self.grid_helper is None or self.grid_helper.full_frequency_grid is None:
            return make_frequency_grid(shape=self.shape, outputs_rfftfreqs=False)
        else:
            return self.grid_helper.full_frequency_grid

    @cached_property
    def full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """Convenience property for `full_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self.full_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @property
    def padded_shape(self):
        return self.pad_options["shape"]

    @property
    def pad_mode(self):
        return self.pad_options["mode"]

    @cached_property
    def padded_coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial coordinate system for the `padded_shape`."""
        grid_helper = (
            self.grid_helper
            if self.shape == self.padded_shape
            else self.pad_options["grid_helper"]
        )
        if grid_helper is None:
            return make_coordinate_grid(shape=self.padded_shape)
        else:
            return grid_helper.coordinate_grid

    @cached_property
    def padded_coordinate_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """Convenience property for `pixel_size * padded_coordinate_grid_in_pixels`"""
        return _safe_multiply_by_constant(
            self.padded_coordinate_grid_in_pixels, self.pixel_size
        )

    @cached_property
    def padded_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1} 2"]:
        """A spatial frequency coordinate system for the `padded_shape`,
        with hermitian symmetry.
        """
        grid_helper = (
            self.grid_helper
            if self.shape == self.padded_shape
            else self.pad_options["grid_helper"]
        )
        if grid_helper is None:
            return make_frequency_grid(shape=self.padded_shape, outputs_rfftfreqs=True)
        else:
            return grid_helper.frequency_grid

    @cached_property
    def padded_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1} 2"]:
        """Convenience property for `padded_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self.padded_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @cached_property
    def padded_full_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial frequency coordinate system for the `padded_shape`,
        without hermitian symmetry.
        """
        grid_helper = (
            self.grid_helper
            if self.shape == self.padded_shape
            else self.pad_options["grid_helper"]
        )
        if grid_helper is None or grid_helper.full_frequency_grid is None:
            return make_frequency_grid(shape=self.padded_shape, outputs_rfftfreqs=False)
        else:
            return grid_helper.full_frequency_grid

    @cached_property
    def padded_full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """Convenience property for `padded_full_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self.padded_full_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    def crop_to_shape(
        self, image: Inexact[Array, "y_dim x_dim"]
    ) -> Inexact[Array, "{self.y_dim} {self.x_dim}"]:
        """Crop an image to `shape`."""
        return crop_to_shape(image, self.shape)

    def pad_to_padded_shape(
        self, image: Inexact[Array, "y_dim x_dim"], **kwargs: Any
    ) -> Inexact[Array, "{self.padded_y_dim} {self.padded_x_dim}"]:
        """Pad an image to `padded_shape`."""
        return pad_to_shape(image, self.padded_shape, mode=self.pad_mode, **kwargs)

    def crop_or_pad_to_padded_shape(
        self, image: Inexact[Array, "y_dim x_dim"], **kwargs: Any
    ) -> Inexact[Array, "{self.padded_y_dim} {self.padded_x_dim}"]:
        """Reshape an image to `padded_shape` using cropping or padding."""
        return resize_with_crop_or_pad(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )

    @property
    def n_pixels(self) -> int:
        """Convenience property for `math.prod(shape)`"""
        return math.prod(self.shape)

    @property
    def y_dim(self) -> int:
        """Convenience property for `shape[0]`"""
        return self.shape[0]

    @property
    def x_dim(self) -> int:
        """Convenience property for `shape[1]`"""
        return self.shape[1]

    @property
    def padded_y_dim(self) -> int:
        """Convenience property for `padded_shape[0]`"""
        return self.padded_shape[0]

    @property
    def padded_x_dim(self) -> int:
        """Convenience property for `padded_shape[1]`"""
        return self.padded_shape[1]

    @property
    def padded_n_pixels(self) -> int:
        """Convenience property for `math.prod(padded_shape)`"""
        return math.prod(self.padded_shape)


class BasicConfig(AbstractConfig, strict=True):
    """Configuration and utilities for a basic electron microscopy
    image."""

    shape: tuple[int, int] = eqx.field(static=True)
    pixel_size: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""]

    grid_helper: Optional[GridHelper]
    pad_options: PadOptions

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: float | Float[Array, ""],
        voltage_in_kilovolts: float | Float[Array, ""],
        *,
        grid_helper: Optional[GridHelper] = None,
        pad_options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
        - `pixel_size`:
            The pixel size of the image in angstroms.
        - `voltage_in_kilovolts`:
            The incident energy of the electron beam.
        - `electrons_per_angstrom_squared`:
            The integrated dose rate of the electron beam.
        - `grid_helper`:
            The `GridHelper` object, which stores the coordinate grids
            for image shape `shape`.
            If not passed, grid are computed at run-time.
        - `pad_options`:
            Options that control image padding.
            - `shape`:
                The shape of the image after padding. By default, equal
                to `shape`.
            - `mode`:
                The method of image padding. By default, equal
                to `"constant"`. For all options, see `jax.numpy.pad`.
            - `grid_helper`:
                The `GridHelper` object, which stores coordinate grids
                for the padded shape. If not passed, grid are computed
                at run-time. By default, equal to `None`.
        """
        # Set parameters
        self.pixel_size = error_if_not_positive(jnp.asarray(pixel_size, dtype=float))
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts, dtype=float)
        )
        # Set shape
        self.shape = shape
        # Set pad options
        self.pad_options = _dict_to_pad_options(pad_options, shape)
        # Finally, grid helper
        self.grid_helper = grid_helper


class DoseConfig(AbstractConfig, strict=True):
    """Configuration and utilities for an electron microscopy image,
    including the electron dose."""

    shape: tuple[int, int] = eqx.field(static=True)
    pixel_size: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""]
    electrons_per_angstrom_squared: Float[Array, ""]

    grid_helper: Optional[GridHelper]
    pad_options: PadOptions

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: float | Float[Array, ""],
        voltage_in_kilovolts: float | Float[Array, ""],
        electrons_per_angstrom_squared: float | Float[Array, ""],
        *,
        grid_helper: Optional[GridHelper] = None,
        pad_options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
        - `pixel_size`:
            The pixel size of the image in angstroms.
        - `voltage_in_kilovolts`:
            The incident energy of the electron beam.
        - `electrons_per_angstrom_squared`:
            The integrated dose rate of the electron beam.
        - `grid_helper`:
            The `GridHelper` object, which stores the coordinate grids
            for image shape `shape`.
            If not passed, grid are computed at run-time.
        - `pad_options`:
            Options that control image padding.
            - `shape`:
                The shape of the image after padding. By default, equal
                to `shape`.
            - `mode`:
                The method of image padding. By default, equal
                to `"constant"`. For all options, see `jax.numpy.pad`.
            - `grid_helper`:
                The `GridHelper` object, which stores coordinate grids
                for the padded shape. If not passed, grid are computed
                at run-time. By default, equal to `None`.
        """
        # Set parameters
        self.pixel_size = error_if_not_positive(jnp.asarray(pixel_size, dtype=float))
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts, dtype=float)
        )
        self.electrons_per_angstrom_squared = jnp.asarray(
            electrons_per_angstrom_squared, dtype=float
        )
        # Set shape
        self.shape = shape
        # Set pad options
        self.pad_options = _dict_to_pad_options(pad_options, shape)
        # Finally, grid helper
        self.grid_helper = grid_helper


def _safe_multiply_by_constant(
    grid: Float[Array, "y_dim x_dim 2"], constant: Float[Array, ""]
) -> Float[Array, "y_dim x_dim 2"]:
    """Multiplies a coordinate grid by a constant in a
    safe way for gradient computation.
    """
    return jnp.where(grid != 0.0, jnp.asarray(constant) * grid, 0.0)


def _dict_to_pad_options(d: dict[str, Any], default_shape: tuple[int, int]) -> PadOptions:
    shape = d["shape"] if "shape" in d else default_shape
    mode = d["mode"] if "mode" in d else "constant"
    grid_helper = d["grid_helper"] if "grid_helper" in d else None

    return PadOptions(shape=shape, mode=mode, grid_helper=grid_helper)
