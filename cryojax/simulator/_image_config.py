"""The image configuration and utility manager."""

import math
import warnings
from functools import cached_property
from typing import Any, Literal, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constants import (
    interaction_constant_from_kilovolts,
    lorentz_factor_from_kilovolts,
    wavelength_from_kilovolts,
)
from ..jax_util import FloatLike, error_if_not_positive
from ..ndimage import make_coordinate_grid, make_frequency_grid


_get_deprecation_msg = lambda self, prop, func: (
    f"`{self.__class__.__name__}.{prop}` has been deprecated and will be "
    "removed in cryoJAX 0.6.0. Instead, make "
    f"the appropriate call to `{self.__class__.__name__}.{func}`."
)


# Not currently public API
class PrecomputedGrids(eqx.Module, strict=True):
    only_rfft: bool
    only_fourier: bool

    _frequency_grid: Float[Array, "_ _ 2"]
    _coordinate_grid: Float[Array, "_ _ 2"] | None
    _full_frequency_grid: Float[Array, "_ _ 2"] | None

    _padded_coordinate_grid: Float[Array, "_ _ 2"] | None
    _padded_frequency_grid: Float[Array, "_ _ 2"] | None
    _padded_full_frequency_grid: Float[Array, "_ _ 2"] | None

    def __init__(
        self,
        shape: tuple[int, int],
        padded_shape: tuple[int, int] | None = None,
        only_fourier: bool = True,
        only_rfft: bool = True,
    ):
        if only_fourier:
            self._coordinate_grid = None
        else:
            self._coordinate_grid = make_coordinate_grid(shape)
        self._frequency_grid = make_frequency_grid(shape, outputs_rfftfreqs=True)
        if only_rfft:
            self._full_frequency_grid = None
        else:
            self._full_frequency_grid = make_frequency_grid(
                shape, outputs_rfftfreqs=False
            )
        if padded_shape is None or padded_shape == shape:
            self._padded_coordinate_grid = None
            self._padded_frequency_grid = None
            self._padded_full_frequency_grid = None
        else:
            if only_fourier:
                self._padded_coordinate_grid = None
            else:
                self._padded_coordinate_grid = make_coordinate_grid(padded_shape)
            self._padded_frequency_grid = make_frequency_grid(
                padded_shape, outputs_rfftfreqs=True
            )
            if only_rfft:
                self._padded_full_frequency_grid = None
            else:
                self._padded_full_frequency_grid = make_frequency_grid(
                    padded_shape, outputs_rfftfreqs=False
                )
        self.only_fourier = only_fourier
        self.only_rfft = only_rfft

    def get(
        self, *, real_space: bool, full: bool = False, padding: bool = False
    ) -> Array:
        _error_msg = (
            "Internal cryoJAX error when fetching "
            "precomputed grids in the `image_config`. "
            "Please report this issue."
        )
        if padding:
            if real_space:
                _coordinate_grid = (
                    self._coordinate_grid
                    if self._padded_coordinate_grid is None
                    else self._padded_coordinate_grid
                )
                if _coordinate_grid is None:
                    raise Exception(_error_msg)
                return _coordinate_grid
            else:
                if full:
                    _full_frequency_grid = (
                        self._full_frequency_grid
                        if self._padded_full_frequency_grid is None
                        else self._padded_full_frequency_grid
                    )
                    if _full_frequency_grid is None:
                        raise Exception(_error_msg)
                    return _full_frequency_grid
                else:
                    return (
                        self._frequency_grid
                        if self._padded_frequency_grid is None
                        else self._padded_frequency_grid
                    )
        else:
            if real_space:
                if self._coordinate_grid is None:
                    raise Exception(_error_msg)
                return self._coordinate_grid
            else:
                if full:
                    if self._full_frequency_grid is None:
                        raise Exception(_error_msg)
                    return self._full_frequency_grid
                else:
                    return self._frequency_grid


class PadOptions(TypedDict):
    shape: tuple[int, int]


class AbstractImageConfig(eqx.Module, strict=True):
    """Configuration and utilities for an electron microscopy image."""

    shape: eqx.AbstractVar[tuple[int, int]]
    pixel_size: eqx.AbstractVar[Float[Array, ""]]
    voltage_in_kilovolts: eqx.AbstractVar[Float[Array, ""]]

    pad_options: eqx.AbstractVar[dict[str, Any]]

    precompute_mode: eqx.AbstractVar[
        Literal["none", "rfft", "fft", "all", "compile_time_eval"]
    ]
    precomputed_grids: eqx.AbstractVar[PrecomputedGrids | None]

    def __check_init__(self):
        cls = self.__class__.__name__
        if self.padded_shape[0] < self.shape[0] or self.padded_shape[1] < self.shape[1]:
            raise AttributeError(
                f"`{cls}.padded_shape` is less than `{cls}.shape` in one or "
                " more dimensions."
            )
        if set(self.pad_options.keys()) != {"shape"}:
            raise AttributeError(
                f"Found that `{cls}.pad_options` was not a dictionary with "
                "key 'shape'. "
                f"Instead, had keys {set(self.grid_options.keys())}."
            )

    @property
    def wavelength_in_angstroms(self) -> Float[Array, ""]:
        """The incident electron wavelength corresponding to the beam
        energy `voltage_in_kilovolts`.
        """
        return wavelength_from_kilovolts(self.voltage_in_kilovolts)

    @property
    def lorentz_factor(self) -> Float[Array, ""]:
        """The lorenz factor at the given `voltage_in_kilovolts`."""
        return lorentz_factor_from_kilovolts(self.voltage_in_kilovolts)

    @property
    def interaction_constant(self) -> Float[Array, ""]:
        """The electron interaction constant at the given `voltage_in_kilovolts`."""
        return interaction_constant_from_kilovolts(self.voltage_in_kilovolts)

    def get_coordinate_grid(self, *, padding: bool = False, physical: bool = True):
        """Return the image coordinate system. See
        [`cryojax.ndimage.make_coordinate_grid`][] for more
        information.

        **Arguments:**

        - `padding`:
            If `True`, return coordinates with shape
            `image_config.padded_shape`. Otherwise, return with
            shape `image_config.shape`.
        - `physical`:
            If `True`, return coordinates in units of angstroms.
            Otherwise, return on the unit box.

        **Returns:**

        The coordinate grid.
        """

        def _get_grid_impl(_self):
            if padding:
                if physical:
                    return _self._padded_coordinate_grid_in_angstroms
                else:
                    return _self._padded_coordinate_grid_in_pixels
            else:
                if physical:
                    return _self._coordinate_grid_in_angstroms
                else:
                    return _self._coordinate_grid_in_pixels

        if self.precompute_mode == "compile_time_eval":
            with jax.ensure_compile_time_eval():
                return _get_grid_impl(self)
        else:
            return _get_grid_impl(self)

    def get_frequency_grid(
        self, *, padding: bool = False, physical: bool = True, full: bool = False
    ):
        """Return a grid of FFT frequencies. See
        [`cryojax.ndimage.make_frequency_grid`] for more
        information.

        **Arguments:**

        - `padding`:
            If `True`, return frequencies corresponding to shape
            `image_config.padded_shape`. Otherwise, use
            `image_config.shape`.
        - `physical`:
            If `True`, return frequencies in units of inverse
            angstroms. Otherwise, return unitless where Nyquist
            is equal to 0.5.
        - `full`:
            If `True`, return the full plane of frequencies
            for usage with `jax.numpy.fft.fftn`.
            Otherwise, return the half plane for usage with
            `jax.numpy.fft.rfftn`.

        **Returns:**

        The frequency grid.
        """

        def _get_grid_impl(_self):
            if padding:
                if physical:
                    if full:
                        return _self._padded_full_frequency_grid_in_angstroms
                    else:
                        return _self._padded_frequency_grid_in_angstroms
                else:
                    if full:
                        return _self._padded_full_frequency_grid_in_pixels
                    else:
                        return _self._padded_frequency_grid_in_pixels
            else:
                if physical:
                    if full:
                        return _self._full_frequency_grid_in_angstroms
                    else:
                        return _self._frequency_grid_in_angstroms
                else:
                    if full:
                        return _self._full_frequency_grid_in_pixels
                    else:
                        return _self._frequency_grid_in_pixels

        if self.precompute_mode == "compile_time_eval":
            with jax.ensure_compile_time_eval():
                return _get_grid_impl(self)
        else:
            return _get_grid_impl(self)

    @property
    def padded_shape(self):
        return self.pad_options["shape"]

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

    @cached_property
    def _coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """A spatial coordinate system for the `shape`."""
        if self.precomputed_grids is None or self.precomputed_grids.only_fourier:
            return make_coordinate_grid(self.shape)
        else:
            return self.precomputed_grids.get(real_space=True)

    @cached_property
    def _coordinate_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """Property for `pixel_size * coordinate_grid_in_pixels`"""
        return _safe_multiply_by_constant(
            self._coordinate_grid_in_pixels, self.pixel_size
        )

    @cached_property
    def _frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim//2+1} 2"]:
        """A spatial frequency coordinate system for the `shape`,
        with hermitian symmetry.
        """
        if self.precomputed_grids is None:
            return make_frequency_grid(self.shape, outputs_rfftfreqs=True)
        else:
            return self.precomputed_grids.get(real_space=False)

    @cached_property
    def _frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim//2+1} 2"]:
        """Convenience property for `frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self._frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @cached_property
    def _full_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """A spatial frequency coordinate system for the `shape`,
        without hermitian symmetry.
        """
        if self.precomputed_grids is None or self.precomputed_grids.only_rfft:
            return make_frequency_grid(shape=self.shape, outputs_rfftfreqs=False)
        else:
            return self.precomputed_grids.get(real_space=False, full=True)

    @cached_property
    def _full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """Property for `full_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self._full_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @cached_property
    def _padded_coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial coordinate system for the `padded_shape`."""
        if self.precomputed_grids is None or self.precomputed_grids.only_fourier:
            return make_coordinate_grid(shape=self.padded_shape)
        else:
            return self.precomputed_grids.get(real_space=True, padding=True)

    @cached_property
    def _padded_coordinate_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """Property for `pixel_size * padded_coordinate_grid_in_pixels`"""
        return _safe_multiply_by_constant(
            self._padded_coordinate_grid_in_pixels, self.pixel_size
        )

    @cached_property
    def _padded_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1} 2"]:
        """A spatial frequency coordinate system for the `padded_shape`,
        with hermitian symmetry.
        """
        if self.precomputed_grids is None:
            return make_frequency_grid(shape=self.padded_shape, outputs_rfftfreqs=True)
        else:
            return self.precomputed_grids.get(real_space=False, padding=True)

    @cached_property
    def _padded_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1} 2"]:
        """Property for `padded_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self._padded_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @cached_property
    def _padded_full_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial frequency coordinate system for the `padded_shape`,
        without hermitian symmetry.
        """
        if self.precomputed_grids is None or self.precomputed_grids.only_rfft:
            return make_frequency_grid(shape=self.padded_shape, outputs_rfftfreqs=False)
        else:
            return self.precomputed_grids.get(real_space=False, full=True, padding=True)

    @cached_property
    def _padded_full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """Property for `padded_full_frequency_grid_in_pixels / pixel_size`"""
        return _safe_multiply_by_constant(
            self._padded_full_frequency_grid_in_pixels, 1 / self.pixel_size
        )

    @property
    def coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "coordinate_grid_in_pixels", "get_coordinate_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._coordinate_grid_in_pixels

    @property
    def coordinate_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "coordinate_grid_in_angstroms", "get_coordinate_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._coordinate_grid_in_angstroms

    @property
    def frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim//2+1} 2"]:
        warnings.warn(
            _get_deprecation_msg(self, "frequency_grid_in_pixels", "g"),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._frequency_grid_in_pixels

    @property
    def frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim//2+1} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "frequency_grid_in_angstroms", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._frequency_grid_in_angstroms

    @property
    def full_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "full_frequency_grid_in_pixels", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._full_frequency_grid_in_pixels

    @property
    def full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "full_frequency_grid_in_angstroms", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._full_frequency_grid_in_angstroms

    @property
    def padded_coordinate_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "padded_coordinate_grid_in_pixels", "get_coordinate_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._padded_coordinate_grid_in_pixels

    @property
    def padded_coordinate_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "coordinate_grid_in_angstroms", "get_coordinate_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._padded_coordinate_grid_in_angstroms

    @property
    def padded_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "padded_frequency_grid_in_pixels", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._padded_frequency_grid_in_pixels

    @property
    def padded_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "padded_frequency_grid_in_angstroms", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._padded_frequency_grid_in_angstroms

    @property
    def padded_full_frequency_grid_in_pixels(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "padded_full_frequency_grid_in_pixels", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._padded_full_frequency_grid_in_pixels

    @property
    def padded_full_frequency_grid_in_angstroms(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        warnings.warn(
            _get_deprecation_msg(
                self, "padded_full_frequency_grid_in_angstroms", "get_frequency_grid"
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._padded_full_frequency_grid_in_angstroms


class BasicImageConfig(AbstractImageConfig, strict=True):
    """Configuration and utilities for a basic electron microscopy
    image.
    """

    shape: tuple[int, int]
    pixel_size: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""]

    pad_options: PadOptions

    precompute_mode: Literal["none", "rfft", "fft", "all", "compile_time_eval"]
    precomputed_grids: PrecomputedGrids | None

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: FloatLike,
        voltage_in_kilovolts: FloatLike,
        *,
        precompute_mode: Literal[
            "none", "rfft", "fft", "all", "compile_time_eval"
        ] = "none",
        pad_options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
        - `pixel_size`:
            The pixel size of the image in angstroms.
        - `voltage_in_kilovolts`:
            The incident energy of the electron beam.
        - `pad_options`:
            Options that control image padding.
            - 'shape':
                The shape of the image after padding. By default, equal
                to `shape`.
        - `precompute_mode`:
            How to pre-compute coordinate and frequency grids stored in
            the `image_config`. Options are
            - 'none':
                Compute grids at runtime and cache the result.
            - 'rfft':
                Only precompute frequencies
                in the Fourier domain for a real-valued function (i.e.
                for use with `jax.numpy.fft.rfftn`). This is the best option
                for most use cases.
            - 'fft':
                Precompute frequencies
                in the Fourier domain for both real and complex-valued functions
                (i.e. for use with `jax.numpy.fft.rfftn` and `jax.numpy.fft.fftn`).
                Relevant for use with Ewald sphere extraction.
            - 'all':
                Precompute all grids in both real-space and frequency-space.
            - 'compile_time_eval':
                Evaluate grids as needed at compile time using
                `jax.ensure_compile_time_eval`.
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
        # Finally, grid precompute
        if precompute_mode == "rfft":
            self.precomputed_grids = PrecomputedGrids(
                self.shape, self.padded_shape, only_rfft=True
            )
        elif precompute_mode == "fft":
            self.precomputed_grids = PrecomputedGrids(
                self.shape, self.padded_shape, only_rfft=False
            )
        elif precompute_mode == "all":
            self.precomputed_grids = PrecomputedGrids(
                self.shape, self.padded_shape, only_fourier=False, only_rfft=False
            )
        else:
            self.precomputed_grids = None
        self.precompute_mode = precompute_mode


class DoseImageConfig(AbstractImageConfig, strict=True):
    """Configuration and utilities for an electron microscopy image,
    including the electron dose."""

    shape: tuple[int, int]
    pixel_size: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""]
    electron_dose: Float[Array, ""]

    pad_options: PadOptions

    precompute_mode: Literal["none", "rfft", "fft", "all", "compile_time_eval"]
    precomputed_grids: PrecomputedGrids | None

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: FloatLike,
        voltage_in_kilovolts: FloatLike,
        electron_dose: FloatLike,
        *,
        precompute_mode: Literal[
            "none", "rfft", "fft", "all", "compile_time_eval"
        ] = "none",
        pad_options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
        - `pixel_size`:
            The pixel size of the image in angstroms.
        - `voltage_in_kilovolts`:
            The incident energy of the electron beam.
        - `electron_dose`:
            The integrated dose rate of the electron beam in
            $e^-/A^2$
        - `pad_options`:
            Options that control image padding.
            - `shape`:
                The shape of the image after padding. By default, equal
                to `shape`.
        - `precompute_mode`:
            How to pre-compute coordinate and frequency grids stored in
            the `image_config`. Options are
            - 'none':
                Compute grids at runtime and cache the result.
            - 'rfft':
                Only precompute frequencies
                in the Fourier domain for a real-valued function (i.e.
                for use with `jax.numpy.fft.rfftn`). This is the best option
                for most use cases.
            - 'fft':
                Precompute frequencies
                in the Fourier domain for both real and complex-valued functions
                (i.e. for use with `jax.numpy.fft.rfftn` and `jax.numpy.fft.fftn`).
                Relevant for use with Ewald sphere extraction.
            - 'all':
                Precompute all grids in both real-space and frequency-space.
            - 'compile_time_eval':
                Evaluate grids as needed at compile time using
                `jax.ensure_compile_time_eval`.
        """
        # Set parameters
        self.pixel_size = error_if_not_positive(jnp.asarray(pixel_size, dtype=float))
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts, dtype=float)
        )
        self.electron_dose = jnp.asarray(electron_dose, dtype=float)
        # Set shape
        self.shape = shape
        # Set pad options
        self.pad_options = _dict_to_pad_options(pad_options, shape)
        # Finally, grid precompute
        if precompute_mode == "rfft":
            self.precomputed_grids = PrecomputedGrids(
                self.shape, self.padded_shape, only_rfft=True
            )
        elif precompute_mode == "fft":
            self.precomputed_grids = PrecomputedGrids(
                self.shape, self.padded_shape, only_rfft=False
            )
        elif precompute_mode == "all":
            self.precomputed_grids = PrecomputedGrids(
                self.shape, self.padded_shape, only_fourier=False, only_rfft=False
            )
        else:
            self.precomputed_grids = None
        self.precompute_mode = precompute_mode


def _safe_multiply_by_constant(
    grid: Float[Array, "y_dim x_dim 2"], constant: Float[Array, ""]
) -> Float[Array, "y_dim x_dim 2"]:
    """Multiplies a coordinate grid by a constant in a
    safe way for gradient computation.
    """
    return jnp.where(grid != 0.0, constant * grid, 0.0)


def _dict_to_pad_options(d: dict[str, Any], default_shape: tuple[int, int]) -> PadOptions:
    shape = d["shape"] if "shape" in d else default_shape

    return PadOptions(shape=shape)
