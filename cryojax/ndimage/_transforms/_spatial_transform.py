from typing import Any, ClassVar, Literal
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import FloatLike, NDArrayLike
from ...rotations import SO2
from .._fourier_utils import enforce_rfftn_self_conjugates
from .._map_coordinates import map_coordinates
from .._operators import FourierPhaseShifts
from ._base_transform import AbstractImageTransform


class PhaseShiftFFT(AbstractImageTransform, strict=True):
    """Apply a phase shift to an image in Fourier space, effectively
    applying an in-plane shift to the image in real space. Only square
    images are supported.

    !!! example "Apply a translation in real-space"

        ```python

        from cryojax.ndimage import PhaseShiftFFT, rfftn, irfftn

        offset_in_angstroms = jnp.array([50.0, -30.0])
        frequency_grid = ... # in angstroms
        fft = rfftn(...) # e.g., fft of a real 2D image

        shift_fn = PhaseShiftFFT(
            offset=offset_in_angstroms, frequency_grid=frequency_grid
        )

        shifted_fft = shift_fn(fft)
        shifted_image = irfftn(shifted_image_fft)
        ```
    """

    translation_operator: Complex[Array, "_ _ _"] | Complex[Array, "_ _"]
    is_rfft: bool = eqx.field(static=True)

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        offset: Float[NDArrayLike, "2"] | Float[NDArrayLike, "3"],
        frequency_grid: Float[NDArrayLike, "_ _ 2"] | Float[NDArrayLike, "_ _ _ 3"],
    ):
        """**Arguments:**

        - `offset`: The offset by which to shift the image, in pixels or angstroms.
        - `frequency_grid`: The frequency grid in pixels or angstroms.
        """
        if _is_square_rfft_grid(frequency_grid):
            # It's worth noting that this condition is breakable and rfftn/fftn
            # inference can fail! One can pass a grid meant for use with fftn
            # with these exact shapes.
            self.is_rfft = True
        elif _is_square_fft_grid(frequency_grid):
            self.is_rfft = False
        else:
            raise ValueError(
                "The `frequency_grid` argument to `PhaseShiftFFT` did not have a valid "
                f"shape {frequency_grid.shape}. `PhaseShiftFFT` only supports square "
                "images as input; you may have passed a grid that does not correspond "
                "to a square image."
            )
        compute_operator = FourierPhaseShifts(offset)
        self.translation_operator = compute_operator(
            jnp.asarray(frequency_grid, dtype=float)
        )

    @override
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]:
        """Apply the phase shift to the input image in Fourier space.

        **Arguments:**

        - `image`:
            The input image in Fourier space.

        **Returns:**

        The phase shifted image in Fourier space.
        """
        if image.shape != self.translation_operator.shape:
            raise ValueError(
                "The image passed to `PhaseShiftFFT` did not have a valid "
                f"shape. The shape of the image was {image.shape}, "
                "but that of the translation operator was "
                f"{self.translation_operator.shape}."
            )
        if self.is_rfft:
            ndim, dim = self.translation_operator.ndim, self.translation_operator.shape[0]
            shape = tuple(ndim * [dim])
            image = enforce_rfftn_self_conjugates(
                image,
                shape,  # pyright: ignore[reportArgumentType]
                includes_dc=False,
                mode="zero",
            )

        return image * self.translation_operator


class RotateFFT(AbstractImageTransform, strict=True):
    """Rotate an image in Fourier space using interpolation.
    Only square images are supported.

    !!! example

        ```python

        from cryojax.ndimage import RotateFFT, fftn, ifftn

        frequency_grid = ... # in pixels
        image = ... # e.g., a real 2D image

        rotation_fn = RotateFFT(
            rotation_angle=45.0, frequency_grid=frequency_grid
        )

        rotated_image_fft = rotation_fn(fftn(image))
        rotated_image = ifftn(rotated_image_fft).real
        ```
    """

    rotation_angle: Float[Array, ""]
    frequency_grid: Float[Array, "_ _ 2"]
    rotation_convention: Literal["frame", "object"]
    map_coordinates_options: dict[str, Any]

    is_rfft: bool = eqx.field(static=True)

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        rotation_angle: FloatLike,
        frequency_grid: Float[NDArrayLike, "y_dim x_dim 2"],
        *,
        pixel_size: FloatLike = 1.0,
        rotation_convention: Literal["frame", "object"] = "object",
        map_coordinates_options: dict[str, Any] = {},
    ):
        """
        **Arguments:**

        - `rotation_angle`: The angle by which to rotate the image, in degrees.
        - `frequency_grid`: The frequency grid.
        - `pixel_size`: The pixel size of the `frequency_grid`.
        - `rotation_convention`:
            If `'object'`, the rotation is with respect to the object in the image.
            If `'frame'`, it is with respect to the frame. These are related by transpose.
        - `map_coordinates_options`:
            A dictionary of options passed to [`cryojax.ndimage.map_coordinates`][].
        """
        if rotation_convention not in ["object", "frame"]:
            raise ValueError(
                "Invalid value for the `rotation_convention` argument to `RotateFFT`. "
                f"Found {rotation_convention}, but options are 'object' or 'frame'."
            )
        if _is_square_rfft_grid(frequency_grid, only_2d=True):
            # It's worth noting that this condition is breakable and rfftn/fftn
            # inference can fail! One can pass a grid meant for use with fftn
            # with these exact shapes.
            self.is_rfft = True
        elif _is_square_fft_grid(frequency_grid, only_2d=True):
            self.is_rfft = False
        else:
            raise ValueError(
                "The `frequency_grid` argument to `RotateFFT` did not have a valid "
                f"shape {frequency_grid.shape}. `RotateFFT` only supports square "
                "2D images as input; you may have passed a grid that does not "
                "correspond to a square image, or you may have tried to use `RotateFFT` "
                "with a volume."
            )
        self.rotation_angle = jnp.asarray(rotation_angle, dtype=float)
        self.rotation_convention = rotation_convention
        self.frequency_grid = jnp.asarray(frequency_grid * pixel_size, dtype=float)
        self.map_coordinates_options = map_coordinates_options

    @override
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Rotate the input image in Fourier space.

        **Arguments:**

        `image`:
            The image in Fourier space.

        **Returns:**

        The rotated image in Fourier space.
        """
        if image.shape != self.frequency_grid.shape[0:-1]:
            raise ValueError(
                "The image passed to `RotateFFT` did not have a valid "
                f"shape. The shape of the image was {image.shape}, "
                "but that of the `frequency_grid` was "
                f"{self.frequency_grid.shape}."
            )
        shift_axes = (0,) if self.is_rfft else (0, 1)
        # Shift images and grid so that zero is in center
        fourier_image = image
        fourier_image_c = jnp.fft.fftshift(fourier_image, axes=shift_axes)
        frequency_grid_c = jnp.fft.fftshift(self.frequency_grid, axes=shift_axes)
        # Rotate the grid
        rotation_matrix = _get_rotation_matrix(
            self.rotation_angle, self.rotation_convention
        )
        rotated_grid = frequency_grid_c @ rotation_matrix
        # Interpolate at new coordinates
        logical_grid = _frequencies_to_indices(rotated_grid, self.is_rfft)
        k_x, k_y = jnp.transpose(logical_grid, axes=[2, 0, 1])
        rotated_image_c = map_coordinates(
            fourier_image_c, (k_y, k_x), **self.map_coordinates_options
        )
        # Shift back, ensure that rfft components are real-valued where they
        # should be, and return
        rotated_image = jnp.fft.ifftshift(rotated_image_c, axes=shift_axes)
        if self.is_rfft:
            dim = rotated_image.shape[0]
            rotated_image = enforce_rfftn_self_conjugates(
                rotated_image, (dim, dim), includes_dc=True, mode="real"
            )
        return rotated_image


def _get_rotation_matrix(angle, convention):
    if convention == "object":
        angle *= -1.0
    angle = jnp.deg2rad(angle)
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = SO2([c, s])
    return rotation.as_matrix()


def _frequencies_to_indices(freqs, is_rfft):
    dim = freqs.shape[0]
    if is_rfft:
        freqsx, freqsy = freqs[..., 0], freqs[..., 1]
        return jnp.stack([freqsx * dim, freqsy * dim + dim // 2], axis=-1)
    else:
        return dim * freqs + dim // 2


def _is_square_rfft_grid(grid, only_2d: bool = False):
    shape, dim = grid.shape, grid.shape[0]
    shapes_2d = [(dim, dim // 2 + 1, 2), (dim, dim // 2, 2)]
    if only_2d:
        return shape in shapes_2d
    else:
        return shape in [*shapes_2d, (dim, dim, dim // 2, 3), (dim, dim, dim // 2 + 1, 3)]


def _is_square_fft_grid(grid, only_2d: bool = False):
    shape, dim = grid.shape, grid.shape[0]
    if only_2d:
        return shape == (dim, dim, 2)
    else:
        return shape in [(dim, dim, 2), (dim, dim, dim, 3)]
