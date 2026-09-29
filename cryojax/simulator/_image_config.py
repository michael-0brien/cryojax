"""The image configuration and utility manager."""

import math
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any, ClassVar, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .._internal import error_if_not_positive, leaf_asarray, leaf_asarray_vector
from ..constants import (
    interaction_constant_from_kilovolts,
    lorentz_factor_from_kilovolts,
    wavelength_from_kilovolts,
)
from ..jax_util import FloatLike, NDArrayLike
from ..ndimage import (
    make_coordinate_grid,
    make_frequency_grid,
    query_efficient_grid_size,
)


# Not currently public API
class PrecomputedGrids(eqx.Module, strict=True):
    only_rfft: bool = eqx.field(static=True)
    only_fourier: bool = eqx.field(static=True)

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


class AbstractImageConfig(eqx.Module, strict=True):
    """Configuration and utilities for an electron microscopy image."""

    shape: eqx.AbstractVar[tuple[int, int]]
    pixel_size: eqx.AbstractVar[Float[NDArrayLike, "..."]]
    voltage_in_kilovolts: eqx.AbstractVar[Float[NDArrayLike, "..."]]

    padded_shape: eqx.AbstractVar[tuple[int, int]]
    precompute_mode: eqx.AbstractVar[
        Literal["none", "rfft", "fft", "all", "compile_time_eval"]
    ]
    precomputed_grids: eqx.AbstractVar[PrecomputedGrids | None]
    options: eqx.AbstractVar[Mapping[str, Any]]

    is_anisotropic: eqx.AbstractClassVar[bool]

    @property
    @abstractmethod
    def magnification_matrix(self) -> Float[Array, "2 2"]:
        """The linear map `D` from the specimen to the detector plane, so that the
        detector records an image `p(x)` as `p(D⁻¹ x)`. The identity if the
        magnification is isotropic."""
        raise NotImplementedError

    def __check_init__(self):
        cls = self.__class__.__name__
        if not all(type(s) == int for s in self.padded_shape):
            raise AttributeError(
                f"Found that `{cls}.padded_shape` was not a tuple of Python integers. "
            )
        if not all(type(s) == int for s in self.shape):
            raise AttributeError(
                f"Found that `{cls}.shape` was not a tuple of Python integers. "
            )
        if self.padded_shape[0] < self.shape[0] or self.padded_shape[1] < self.shape[1]:
            raise AttributeError(
                f"Found that `{cls}.padded_shape` is less than `{cls}.shape` in one or "
                " more dimensions."
            )

    @property
    def wavelength_in_angstroms(self) -> Float[Array, ""]:
        """The incident electron wavelength corresponding to the beam
        energy `voltage_in_kilovolts`.
        """
        return wavelength_from_kilovolts(
            error_if_not_positive(jnp.asarray(self.voltage_in_kilovolts))
        )

    @property
    def lorentz_factor(self) -> Float[Array, ""]:
        """The lorenz factor at the given `voltage_in_kilovolts`."""
        return lorentz_factor_from_kilovolts(
            error_if_not_positive(jnp.asarray(self.voltage_in_kilovolts))
        )

    @property
    def interaction_constant(self) -> Float[Array, ""]:
        """The electron interaction constant at the given `voltage_in_kilovolts`."""
        return interaction_constant_from_kilovolts(
            error_if_not_positive(jnp.asarray(self.voltage_in_kilovolts))
        )

    def get_coordinate_grid(
        self, *, padding: bool = False, physical: bool = True, anisotropy: bool = True
    ) -> Float[Array, "y_dim x_dim 2"]:
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
        - `anisotropy`:
            If `True` and `physical = True`, apply the anisotropic magnification.

        **Returns:**

        The coordinate grid.
        """

        def _get_grid_impl(_self):
            if padding:
                return _self._padded_coordinate_grid
            else:
                return _self._coordinate_grid

        if self.precompute_mode == "compile_time_eval":
            with jax.ensure_compile_time_eval():
                coordinate_grid = _get_grid_impl(self)
        else:
            coordinate_grid = _get_grid_impl(self)

        if physical:
            coordinate_grid = self.apply_magnification(
                coordinate_grid, is_real_space=True, anisotropy=anisotropy
            )

        return coordinate_grid

    def get_frequency_grid(
        self,
        *,
        padding: bool = False,
        physical: bool = True,
        full: bool = False,
        anisotropy: bool = True,
    ) -> Float[Array, "y_dim x_dim 2"]:
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
        - `anisotropy`:
            If `True` and `physical = True`, apply the anisotropic magnification.

        **Returns:**

        The frequency grid.
        """

        def _get_grid_impl(_self):
            if padding:
                if full:
                    return _self._padded_full_frequency_grid
                else:
                    return _self._padded_frequency_grid
            else:
                if full:
                    return _self._full_frequency_grid
                else:
                    return _self._frequency_grid

        if self.precompute_mode == "compile_time_eval":
            with jax.ensure_compile_time_eval():
                frequency_grid = _get_grid_impl(self)
        else:
            frequency_grid = _get_grid_impl(self)

        if physical:
            frequency_grid = self.apply_magnification(
                frequency_grid, anisotropy=anisotropy
            )

        return frequency_grid

    def apply_magnification(
        self,
        grid: Float[Array, "y_dim x_dim 2"],
        *,
        is_real_space: bool = False,
        anisotropy: bool = True,
    ) -> Float[Array, "y_dim x_dim 2"]:
        """Convert a unitless grid to physical units in the specimen frame. Frequencies
        (in FFT order) map to `Dᵀk / pixel_size` and coordinates (centered) to
        `D⁻¹x * pixel_size`, for the `magnification_matrix` `D`.

        **Arguments:**

        - `grid`:
            A grid of frequencies in cycles per pixel, or of coordinates in pixels.
        - `is_real_space`:
            If `True`, `grid` is a coordinate grid. Otherwise, it is a frequency grid.
        - `anisotropy`:
            If `True`, apply the anisotropic magnification.
        """
        pixel_size = error_if_not_positive(jnp.asarray(self.pixel_size))
        constant = pixel_size if is_real_space else 1 / pixel_size
        grid = _safe_constant_multiply(grid, constant, is_fft_grid=not is_real_space)
        if not (self.is_anisotropic and anisotropy):
            return grid
        matrix = self.magnification_matrix
        matrix = _inverse_2x2(matrix).T if is_real_space else matrix
        return _safe_matrix_multiply(grid, matrix, is_fft_grid=not is_real_space)

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
    def _coordinate_grid(
        self,
    ) -> Float[Array, "{self.y_dim} {self.x_dim} 2"]:
        """A spatial coordinate system for the `shape`."""
        if self.precomputed_grids is None or self.precomputed_grids.only_fourier:
            return make_coordinate_grid(self.shape)
        else:
            return self.precomputed_grids.get(real_space=True)

    @cached_property
    def _frequency_grid(
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
    def _full_frequency_grid(
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
    def _padded_coordinate_grid(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial coordinate system for the `padded_shape`."""
        if self.precomputed_grids is None or self.precomputed_grids.only_fourier:
            return make_coordinate_grid(shape=self.padded_shape)
        else:
            return self.precomputed_grids.get(real_space=True, padding=True)

    @cached_property
    def _padded_frequency_grid(
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
    def _padded_full_frequency_grid(
        self,
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim} 2"]:
        """A spatial frequency coordinate system for the `padded_shape`,
        without hermitian symmetry.
        """
        if self.precomputed_grids is None or self.precomputed_grids.only_rfft:
            return make_frequency_grid(shape=self.padded_shape, outputs_rfftfreqs=False)
        else:
            return self.precomputed_grids.get(real_space=False, full=True, padding=True)


class BasicImageConfig(AbstractImageConfig, strict=True):
    """Configuration and utilities for a basic electron microscopy
    image.
    """

    shape: tuple[int, int]
    pixel_size: Float[NDArrayLike, "..."]
    voltage_in_kilovolts: Float[NDArrayLike, "..."]

    padded_shape: tuple[int, int]
    precompute_mode: Literal["none", "rfft", "fft", "all", "compile_time_eval"] = (
        eqx.field(static=True)
    )
    precomputed_grids: PrecomputedGrids | None
    options: Mapping[str, Any] = eqx.field(static=True)

    is_anisotropic: ClassVar[bool] = False

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: FloatLike,
        voltage_in_kilovolts: FloatLike = 300.0,
        *,
        padded_shape: tuple[int, int] | None = None,
        pad_scale: float = 1.0,
        precompute_mode: Literal[
            "none", "rfft", "fft", "all", "compile_time_eval"
        ] = "none",
        options: Mapping[str, Any] | None = None,
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
        - `pixel_size`:
            The pixel size of the image in angstroms.
        - `voltage_in_kilovolts`:
            The incident energy of the electron beam.
        - `padded_shape`:
            The shape of the image after padding. By default, equal
            to `shape`.
        - `pad_scale`:
            A scale factor used to determine the `padded_shape`.
            Used only if `padded_shape` is not directly provided.
            Unless `pad_scale = 1.0`, it is only an approximation
            of the resulting `padded_shape`. By default, a good
            array size for FFT efficiency is chosen.
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
        - `options`:
            Advanced options for the simulation. No keys are currently accepted.
        """
        # Set parameters
        self.pixel_size = leaf_asarray(pixel_size, dtype=float)
        self.voltage_in_kilovolts = leaf_asarray(voltage_in_kilovolts, dtype=float)
        # Set shape and padded shape
        self.shape = shape
        self.padded_shape = _set_padded_shape(type(self), shape, padded_shape, pad_scale)
        self.options = _resolve_options_dict(type(self), options, supported={})
        # Finally, grid precompute
        self.precomputed_grids = _make_precomputed_grids(
            self.shape, self.padded_shape, precompute_mode
        )
        self.precompute_mode = precompute_mode

    @property
    def magnification_matrix(self) -> Float[Array, "2 2"]:
        """The identity: the magnification is isotropic."""
        return jnp.eye(2)


class DoseImageConfig(AbstractImageConfig, strict=True):
    """Configuration and utilities for an electron microscopy image,
    including the electron dose."""

    shape: tuple[int, int]
    pixel_size: Float[NDArrayLike, "..."]
    voltage_in_kilovolts: Float[NDArrayLike, "..."]
    electron_dose: Float[NDArrayLike, "..."]

    padded_shape: tuple[int, int]
    precompute_mode: Literal["none", "rfft", "fft", "all", "compile_time_eval"] = (
        eqx.field(static=True)
    )
    precomputed_grids: PrecomputedGrids | None
    options: Mapping[str, Any] = eqx.field(static=True)

    is_anisotropic: ClassVar[bool] = False

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: FloatLike,
        voltage_in_kilovolts: FloatLike = 300.0,
        electron_dose: FloatLike = 50.0,
        *,
        padded_shape: tuple[int, int] | None = None,
        pad_scale: float = 1.0,
        precompute_mode: Literal[
            "none", "rfft", "fft", "all", "compile_time_eval"
        ] = "none",
        options: Mapping[str, Any] | None = None,
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
        - `padded_shape`:
            The shape of the image after padding. By default, equal
            to `shape`.
        - `pad_scale`:
            A scale factor used to determine the `padded_shape`.
            Used only if `padded_shape` is not directly provided.
            Unless `pad_scale = 1.0`, it is only an approximation
            of the resulting `padded_shape`. By default, a good
            array size for FFT efficiency is chosen.
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
        - `options`:
            Advanced options for the simulation. No keys are currently accepted.
        """
        # Set parameters
        self.pixel_size = leaf_asarray(pixel_size, dtype=float)
        self.voltage_in_kilovolts = leaf_asarray(voltage_in_kilovolts, dtype=float)
        self.electron_dose = leaf_asarray(electron_dose, dtype=float)
        # Set shape and padded shape
        self.shape = shape
        self.padded_shape = _set_padded_shape(type(self), shape, padded_shape, pad_scale)
        self.options = _resolve_options_dict(type(self), options, supported={})
        # Finally, grid precompute
        self.precomputed_grids = _make_precomputed_grids(
            self.shape, self.padded_shape, precompute_mode
        )
        self.precompute_mode = precompute_mode

    @property
    def magnification_matrix(self) -> Float[Array, "2 2"]:
        """The identity: the magnification is isotropic."""
        return jnp.eye(2)

    @property
    def electrons_per_pixel(self) -> Float[Array, ""]:
        """The `electron_dose` in a given pixel area."""
        return (
            error_if_not_positive(jnp.asarray(self.electron_dose))
            * jnp.asarray(self.pixel_size) ** 2
        )


class AnisotropicImageConfig(AbstractImageConfig, strict=True):
    """Configuration and utilities for an electron microscopy image with an
    anisotropic magnification.

    The magnification stretches the image by `1 + a` along `angle` and compresses it
    by `1 - a` perpendicular to it, with `angle` measured as for
    `AstigmaticCTF.astigmatism_angle`. It is parametrized by the vector
    `anisotropy_xy = a * (cos(2 * angle), sin(2 * angle))`.
    """

    shape: tuple[int, int]
    pixel_size: Float[NDArrayLike, "..."]
    voltage_in_kilovolts: Float[NDArrayLike, "..."]
    anisotropy_xy: Float[NDArrayLike, "... 2"]

    padded_shape: tuple[int, int]
    precompute_mode: Literal["none", "rfft", "fft", "all", "compile_time_eval"] = (
        eqx.field(static=True)
    )
    precomputed_grids: PrecomputedGrids | None
    options: Mapping[str, Any] = eqx.field(static=True)

    is_anisotropic: ClassVar[bool] = True

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: FloatLike,
        voltage_in_kilovolts: FloatLike = 300.0,
        anisotropy_xy: Float[NDArrayLike, "... 2"] | Sequence[float] = (0.0, 0.0),
        *,
        padded_shape: tuple[int, int] | None = None,
        pad_scale: float = 1.0,
        precompute_mode: Literal[
            "none", "rfft", "fft", "all", "compile_time_eval"
        ] = "none",
        options: Mapping[str, Any] | None = None,
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
        - `pixel_size`:
            The pixel size of the image in angstroms.
        - `voltage_in_kilovolts`:
            The incident energy of the electron beam.
        - `anisotropy_xy`:
            The magnification anisotropy, `a * (cos(2 * angle), sin(2 * angle))`.
        - `padded_shape`:
            The shape of the image after padding. By default, equal
            to `shape`.
        - `pad_scale`:
            A scale factor used to determine the `padded_shape`, as for
            `BasicImageConfig`.
        - `precompute_mode`:
            How to pre-compute coordinate and frequency grids, as for
            `BasicImageConfig`.
        - `options`:
            Advanced options for the simulation. The accepted keys are:
            - `'nufft'`:
                A dictionary of keyword arguments for
                [`cryojax.ndimage.nufft_resample`][], with keys `'eps'` and
                `'upsampfac'`.
        """
        # Set parameters
        self.pixel_size = leaf_asarray(pixel_size, dtype=float)
        self.voltage_in_kilovolts = leaf_asarray(voltage_in_kilovolts, dtype=float)
        self.anisotropy_xy = leaf_asarray_vector(
            anisotropy_xy, 2, name="AnisotropicImageConfig(..., anisotropy_xy=...)"
        )
        # Set shape and padded shape
        self.shape = shape
        self.padded_shape = _set_padded_shape(type(self), shape, padded_shape, pad_scale)
        self.options = _resolve_options_dict(
            type(self), options, supported={"nufft": ("eps", "upsampfac")}
        )
        # Finally, grid precompute
        self.precomputed_grids = _make_precomputed_grids(
            self.shape, self.padded_shape, precompute_mode
        )
        self.precompute_mode = precompute_mode

    @property
    def magnification_matrix(self) -> Float[Array, "... 2 2"]:
        """`D = I + [[-e0, e1], [e1, e0]]`, for `anisotropy_xy = (e0, e1)`."""
        anisotropy_xy = jnp.asarray(self.anisotropy_xy)
        e0, e1 = anisotropy_xy[..., 0], anisotropy_xy[..., 1]
        anisotropy = jnp.stack(
            [jnp.stack([-e0, e1], axis=-1), jnp.stack([e1, e0], axis=-1)], axis=-2
        )
        return jnp.eye(2) + anisotropy


def _safe_constant_multiply(
    grid: Float[Array, "y_dim x_dim 2"], constant: Float[Array, ""], is_fft_grid: bool
) -> Float[Array, "y_dim x_dim 2"]:
    """Multiply a coordinate grid by a constant, keeping zero-valued
    components independent of `constant` so that gradients through
    `jnp.linalg.norm(grid, axis=-1)` remain finite at the origin.

    For an FFT frequency grid (DC at corner): the x-component is zero in
    column 0 and the y-component is zero in row 0; those entries are left
    untouched.  For a real-space coordinate grid (center at N//2): the
    x-component is zero in the center column and the y-component is zero
    in the center row; those entries are left untouched.
    """
    y_dim, x_dim = grid.shape[0], grid.shape[1]
    row_idx = jnp.arange(y_dim)
    col_idx = jnp.arange(x_dim)
    if is_fft_grid:
        scale_x = jnp.where(col_idx > 0, constant, 1.0)
        scale_y = jnp.where(row_idx > 0, constant, 1.0)
    else:
        scale_x = jnp.where(col_idx != x_dim // 2, constant, 1.0)
        scale_y = jnp.where(row_idx != y_dim // 2, constant, 1.0)
    return jnp.stack(
        [grid[..., 0] * scale_x[None, :], grid[..., 1] * scale_y[:, None]],
        axis=-1,
    )


def _safe_matrix_multiply(
    grid: Float[Array, "y_dim x_dim 2"], matrix: Float[Array, "2 2"], is_fft_grid: bool
) -> Float[Array, "y_dim x_dim 2"]:
    """Multiply the vectors of a grid by a matrix, `grid @ matrix`, keeping the origin
    independent of `matrix` so that gradients through `jnp.linalg.norm(grid, axis=-1)`
    remain finite there. The origin is at the corner for an FFT grid and at the
    center (N//2) for a real-space grid.
    """
    y_dim, x_dim = grid.shape[0], grid.shape[1]
    row_idx = jnp.arange(y_dim)
    col_idx = jnp.arange(x_dim)
    if is_fft_grid:
        is_origin = (row_idx == 0)[:, None] & (col_idx == 0)[None, :]
    else:
        is_origin = (row_idx == y_dim // 2)[:, None] & (col_idx == x_dim // 2)[None, :]
    return jnp.where(is_origin[..., None], grid, grid @ matrix)


def _inverse_2x2(matrix: Float[Array, "2 2"]) -> Float[Array, "2 2"]:
    """The inverse of a 2x2 matrix, by its adjugate."""
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    adjugate = jnp.stack(
        [jnp.stack([d, -b], axis=-1), jnp.stack([-c, a], axis=-1)], axis=-2
    )
    return adjugate / (a * d - b * c)[..., None, None]


def _resolve_options_dict(
    cls: type,
    options: Mapping[str, Any] | None,
    supported: Mapping[str, tuple[str, ...]],
) -> Mapping[str, Any]:
    """Check the `options` of an image config against the `supported` keys (each with
    the keys of its sub-dictionary), and freeze it so that it may be a static field."""
    options = {} if options is None else options
    for key, value in options.items():
        if key not in supported:
            raise ValueError(
                f"Found invalid value for `{cls.__name__}(..., options=...)`. "
                f"Supported keys are {list(supported)}, but got key {key!r}."
            )
        subkeys = supported[key]
        if not isinstance(value, Mapping) or not set(value).issubset(subkeys):
            raise ValueError(
                f"Found invalid value for `{cls.__name__}(..., options=...)`. The "
                f"value of {key!r} must be a dictionary with keys in {list(subkeys)}, "
                f"but got {value!r}."
            )
    return _FrozenDict(
        {
            key: _FrozenDict(value) if isinstance(value, Mapping) else value
            for key, value in options.items()
        }
    )


class _FrozenDict(Mapping):
    """A read-only, hashable dictionary, for the static `options` of an image config."""

    def __init__(self, mapping: Mapping[str, Any]):
        self._dict = dict(mapping)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._dict.items())))

    def __repr__(self) -> str:
        return repr(self._dict)


def _make_precomputed_grids(
    shape: tuple[int, int],
    padded_shape: tuple[int, int],
    precompute_mode: Literal["none", "rfft", "fft", "all", "compile_time_eval"],
) -> PrecomputedGrids | None:
    if precompute_mode == "rfft":
        return PrecomputedGrids(shape, padded_shape, only_rfft=True)
    elif precompute_mode == "fft":
        return PrecomputedGrids(shape, padded_shape, only_rfft=False)
    elif precompute_mode == "all":
        return PrecomputedGrids(shape, padded_shape, only_fourier=False, only_rfft=False)
    else:
        return None


def _set_padded_shape(
    cls, shape: tuple[int, int], padded_shape: tuple[int, int] | None, pad_scale: float
):
    if padded_shape is not None:
        return padded_shape
    elif pad_scale == 1.0:
        return shape
    elif pad_scale > 1.0:
        return cast(
            tuple[int, int],
            query_efficient_grid_size(shape, pad_scale=pad_scale, only_even=True),
        )
    else:
        raise ValueError(
            f"Invalid value for `{cls.__name__}(..., pad_scale=...)`. "
            f"This must be greater than `1.0`, but got value `{pad_scale}`."
        )
