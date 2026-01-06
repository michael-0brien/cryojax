from typing import ClassVar, Literal
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .. import enforce_rfftn_self_conjugates
from .._map_coordinates import map_coordinates
from ._base_transform import AbstractImageTransform


class PhaseShiftFourierImage(AbstractImageTransform, strict=True):
    """Apply a phase shift to an image in Fourier space.

    This class implements a phase shift of an image in Fourier space.
    The shift is specified in pixels or angstroms along each axis.

    Attributes:
        translation_operator: The precomputed translation operator.
        is_rfft: Whether the input image is in RFFT format.
    """

    translation_operator: Complex[Array, "y_dim x_dim"]
    is_rfft: bool
    shape: tuple[int, int]
    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        offset: Float[Array, "2"],
        frequency_grid: Float[Array, "y_dim x_dim 2"],
        is_rfft: bool,
    ):
        """
        **Arguments:**

            - `offset`: The offset by which to shift the image, in pixels or angstroms.
            - `frequency_grid`: The frequency grid in pixels or angstroms.
            - `is_rfft`: Whether the frequency grid is in full or rfft format.
                Right now only full format is supported for image rotation.

        **Example:**
            ```python

            from cryojax.ndimage import PhaseShiftFourierImage, fftn, ifftn

            frequency_grid = ... # in angstroms
            image = ... # e.g., a real 2D image

            shift_op = PhaseShiftFourierImage(
                offset=jnp.array([50.0, -30.0]),
                frequency_grid=frequency_grid,
            )

            shifted_image_fft = shift_op(fftn(image))
            shifted_image = ifftn(shifted_image_fft).real
            ```

        """
        self.translation_operator = jnp.exp(
            -1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid, offset))
        )
        self.is_rfft = is_rfft
        if is_rfft:
            self.shape = (frequency_grid.shape[0], frequency_grid.shape[0])
        else:
            self.shape = (frequency_grid.shape[0], frequency_grid.shape[1])

    @override
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Apply the phase shift to the input image in Fourier space.

        Args:
            image: The input image in Fourier space.
            shape: The shape of the Fourier Image in real space.

        Returns:
            The phase-shifted image in Fourier space.
        """
        if self.is_rfft:
            image = enforce_rfftn_self_conjugates(
                image, self.shape, includes_dc=False, mode="zero"
            )

        return image * self.translation_operator


class RotateFourierImage(AbstractImageTransform, strict=True):
    """Rotate an image in Fourier space.

    This class implements rotation of an image in Fourier space using
    interpolation. The rotation is specified by an angle in degrees.

    Attributes:
        angle_degrees: The angle by which to rotate the image, in degrees.
        frequency_grid: The full frequency grid in pixels or angstroms.
    """

    angle_degrees: float
    frequency_grid: Float[Array, "y_dim x_dim 2"]
    order: int
    mode: str
    cval: float | complex
    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        angle_degrees: float,
        frequency_grid: Float[Array, "y_dim x_dim 2"],
        is_rfft: bool,
        *,
        pixel_size: float = 1.0,
        order: Literal[0, 1] = 1,
        mode: str = "fill",
        cval: float | complex = 0.0,
    ):
        """
        **Arguments:**

            - `angle_degrees`: The angle by which to rotate the image, in degrees.
            - `frequency_grid`: The frequency grid in pixels or angstroms.
            - `is_rfft`: Whether the frequency grid is in full or rfft format.
                Right now only full format is supported for image rotation.
            - `pixel_size`: The pixel size in angstroms or the unit of choice.
            - `order`: The order of the spline interpolation. Only 0 and 1 are supported.
            - `mode`: The mode to use for points outside the boundaries.
            - `cval`: The constant value to use when `mode` is 'fill'.

        **Example:**
            ```python

            from cryojax.ndimage import RotateFourierImage, fftn, ifftn

            frequency_grid = ... # in pixels
            image = ... # e.g., a real 2D image

            rotation_op = RotateFourierImage(
                angle_degrees=45.0,
                frequency_grid=frequency_grid,
            )

            rotated_image_fft = rotation_op(fftn(image))
            rotated_image = ifftn(rotated_image_fft).real
            ```

        """
        assert order in (0, 1), "Only order 0 and 1 are supported."

        if is_rfft:
            raise NotImplementedError("RotateFourierImage does not support rfft arrays.")
        self.angle_degrees = angle_degrees
        self.frequency_grid = frequency_grid * pixel_size
        self.order = order
        self.mode = mode
        self.cval = cval

    @override
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Rotate the input image in Fourier space.

        Args:
            image: The input image in Fourier space.

        Returns:
            The rotated image in Fourier space.
        """
        rotated_image = _rotate_fourier_image(
            image,
            angle_degrees=self.angle_degrees,
            frequency_grid=self.frequency_grid,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )
        return rotated_image


def _rotate_fourier_image(
    fourier_image: Complex[Array, "y_dim x_dim"],
    angle_degrees: float,
    frequency_grid: Float[Array, "y_dim x_dim 2"],
    order: int = 1,
    mode: str = "fill",
    cval: float | complex = 0.0,
) -> Complex[Array, "y_dim x_dim"]:
    angle = jnp.deg2rad(angle_degrees)
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    rotation_matrix = jnp.array([[c, -s], [s, c]])

    rotated_grid = frequency_grid @ rotation_matrix
    N = rotated_grid.shape[1]
    logical_frequency_coordinates = (rotated_grid * N) + N // 2
    k_y, k_x = jnp.transpose(logical_frequency_coordinates, axes=[2, 0, 1])

    image_fft = jnp.fft.fftshift(fourier_image)
    rotated_image = map_coordinates(
        image_fft, (k_x, k_y), order=order, mode=mode, cval=cval
    )
    return rotated_image
