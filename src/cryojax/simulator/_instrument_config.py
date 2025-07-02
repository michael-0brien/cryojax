"""
The image configuration and utility manager.
"""

import math
from functools import cached_property
from typing import Any, Callable, Optional, Union

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


class InstrumentConfig(eqx.Module, strict=True):
    """Configuration and utilities for an electron microscopy image."""

    shape: tuple[int, int] = eqx.field(static=True)
    pixel_size: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""]
    electrons_per_angstrom_squared: Optional[Float[Array, ""]]

    padded_shape: tuple[int, int] = eqx.field(static=True)
    pad_mode: Union[str, Callable]

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: float | Float[Array, ""],
        voltage_in_kilovolts: float | Float[Array, ""],
        electrons_per_angstrom_squared: Optional[float | Float[Array, ""]] = None,
        padded_shape: Optional[tuple[int, int]] = None,
        *,
        pad_scale: float = 1.0,
        pad_mode: Union[str, Callable] = "constant",
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
        - `padded_shape`:
            The shape of the image after padding. If this argument is
            not given, it can be set by the `pad_scale` argument.
        - `pad_scale`: A scale factor at which to pad the image. This is
                       optionally used to set `padded_shape` and must be
                       greater than `1`. If `padded_shape` is set, this
                       argument is ignored.
        - `pad_mode`:
            The method of image padding. By default, `"constant"`.
            For all options, see `jax.numpy.pad`.
        """
        self.shape = shape
        self.pixel_size = error_if_not_positive(jnp.asarray(pixel_size))
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts)
        )
        if electrons_per_angstrom_squared is None:
            self.electrons_per_angstrom_squared = None
        else:
            self.electrons_per_angstrom_squared = error_if_not_positive(
                jnp.asarray(electrons_per_angstrom_squared)
            )
        self.pad_mode = pad_mode
        # Set shape after padding
        if padded_shape is None:
            self.padded_shape = (int(pad_scale * shape[0]), int(pad_scale * shape[1]))
        else:
            self.padded_shape = padded_shape

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
        return make_coordinate_grid(shape=self.shape)

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
        return make_frequency_grid(shape=self.shape)

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
        return make_frequency_grid(shape=self.shape, outputs_rfftfreqs=False)

    @cached_property
    def full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """Convenience property for `full_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self.full_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @cached_property
    def padded_coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial coordinate system for the `padded_shape`."""
        return make_coordinate_grid(shape=self.padded_shape)

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
        return make_frequency_grid(shape=self.padded_shape)

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
        return make_frequency_grid(shape=self.padded_shape, outputs_rfftfreqs=False)

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


def _safe_multiply_by_constant(
    grid: Float[Array, "y_dim x_dim 2"], constant: Float[Array, ""]
) -> Float[Array, "y_dim x_dim 2"]:
    """Multiplies a coordinate grid by a constant in a
    safe way for gradient computation.
    """
    return jnp.where(grid != 0.0, jnp.asarray(constant) * grid, 0.0)
