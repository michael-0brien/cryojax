from typing import ClassVar
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .. import enforce_rfftn_self_conjugates
from .._fft import fftn, ifftn, irfftn, rfftn
from .._map_coordinates import map_coordinates
from ._base_transform import AbstractImageTransform


class PhaseShiftFourierImage(AbstractImageTransform, strict=True):
    """Apply a phase shift to an image in Fourier space.

    This class implements a phase shift of an image in Fourier space.
    The shift is specified in pixels along each axis.

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
        shape: tuple[int, int],
    ):
        self.translation_operator = jnp.exp(
            -1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid, offset))
        )
        self.is_rfft = is_rfft
        self.shape = shape

    @override
    def __call__(
        self, fourier_image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Apply the phase shift to the input image in Fourier space.

        Args:
            fourier_image: The input image in Fourier space.
            shape: The shape of the Fourier Image in real space.

        Returns:
            The phase-shifted image in Fourier space.
        """
        if self.is_rfft:
            fourier_image = enforce_rfftn_self_conjugates(
                fourier_image, self.shape, includes_dc=False, mode="zero"
            )

        return fourier_image * self.translation_operator


class TranslateImage(AbstractImageTransform, strict=True):
    """Translate an image in real space.

    This class implements translation of an image in real space using
    phase shifts in Fourier space. The translation is specified in pixels
    along each axis.

    Attributes:
        phase_shift_fft_image_op: The phase shift operator in Fourier space.
        is_rfft: Whether the input image is in RFFT format.
    """

    translation_operator: Complex[Array, "y_dim x_dim"]
    is_rfft: bool
    shape: tuple[int, int]
    is_real_space: ClassVar[bool] = True

    def __init__(
        self,
        offset: Float[Array, "2"],
        frequency_grid: Float[Array, "y_dim x_dim 2"],
        is_rfft: bool,
        shape: tuple[int, int],
    ):
        self.translation_operator = jnp.exp(
            -1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid, offset))
        )
        self.is_rfft = is_rfft
        self.shape = shape

    @override
    def __call__(self, image: Float[Array, "y_dim x_dim"]) -> Float[Array, "y_dim x_dim"]:
        """Translate the input image.

        Args:
            image: The input image.

        Returns:
            The translated image.
        """
        if self.is_rfft:
            fourier_image = enforce_rfftn_self_conjugates(
                rfftn(image), self.shape, includes_dc=False, mode="zero"
            )
            translated_image = irfftn(fourier_image * self.translation_operator)
        else:
            translated_image = ifftn(fftn(image) * self.translation_operator).real
        return translated_image


class RotateFourierImage(AbstractImageTransform, strict=True):
    """Rotate an image in Fourier space.

    This class implements rotation of an image in Fourier space using
    interpolation. The rotation is specified by an angle in degrees.

    Attributes:
        angle_degrees: The angle by which to rotate the image, in degrees.
        full_frequency_grid_in_pixels: The full frequency grid in pixels.
    """

    angle_degrees: float
    full_frequency_grid_in_pixels: Float[Array, "y_dim x_dim 2"]
    order: int
    mode: str
    cval: float | complex
    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        full_frequency_grid_in_pixels: Float[Array, "y_dim x_dim 2"],
        angle_degrees: float,
        order: int = 1,
        mode: str = "fill",
        cval: float | complex = 0.0,
    ):
        self.angle_degrees = angle_degrees
        self.full_frequency_grid_in_pixels = full_frequency_grid_in_pixels
        self.order = order
        self.mode = mode
        self.cval = cval

    @override
    def __call__(
        self, fourier_image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Rotate the input image in Fourier space.

        Args:
            fourier_image: The input image in Fourier space.

        Returns:
            The rotated image in Fourier space.
        """
        rotated_image = _rotate_image(
            fourier_image,
            angle_degrees=self.angle_degrees,
            full_frequency_grid_in_pixels=self.full_frequency_grid_in_pixels,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )
        return rotated_image


class RotateImage(AbstractImageTransform, strict=True):
    """Rotate an image in real space.

    This class implements rotation of an image in real space using
    interpolation. The rotation is specified by an angle in degrees.

    Attributes:
        angle_degrees: The angle by which to rotate the image, in degrees.
        full_frequency_grid_in_pixels: The full frequency grid in pixels.
    """

    angle_degrees: float
    full_frequency_grid_in_pixels: Float[Array, "y_dim x_dim 2"]
    order: int
    mode: str
    cval: float | complex
    is_real_space: ClassVar[bool] = True

    def __init__(
        self,
        full_frequency_grid_in_pixels: Float[Array, "y_dim x_dim 2"],
        angle_degrees: float,
        order: int = 1,
        mode: str = "fill",
        cval: float | complex = 0.0,
    ):
        self.angle_degrees = angle_degrees
        self.full_frequency_grid_in_pixels = full_frequency_grid_in_pixels
        self.order = order
        self.mode = mode
        self.cval = cval

    @override
    def __call__(self, image: Float[Array, "y_dim x_dim"]) -> Float[Array, "y_dim x_dim"]:
        """Rotate the input image.

        Args:
            image: The input image.

        Returns:
            The rotated image.
        """
        rotated_image = _rotate_image(
            fftn(image),
            angle_degrees=self.angle_degrees,
            full_frequency_grid_in_pixels=self.full_frequency_grid_in_pixels,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )
        return ifftn(rotated_image).real


def _rotate_image(
    fourier_image: Complex[Array, "y_dim x_dim"],
    angle_degrees: float,
    full_frequency_grid_in_pixels: Float[Array, "y_dim x_dim 2"],
    order: int = 1,
    mode: str = "fill",
    cval: float | complex = 0.0,
) -> Complex[Array, "y_dim x_dim"]:
    angle = jnp.deg2rad(angle_degrees)
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    rotation_matrix = jnp.array([[c, -s], [s, c]])

    rotated_grid = full_frequency_grid_in_pixels @ rotation_matrix
    N = rotated_grid.shape[1]
    logical_frequency_coordinates = (rotated_grid * N) + N // 2
    k_y, k_x = jnp.transpose(logical_frequency_coordinates, axes=[2, 0, 1])

    # image_fft = jnp.fft.fftshift(cxim.fftn(image))
    image_fft = jnp.fft.fftshift(fourier_image)
    rotated_image = map_coordinates(
        image_fft, (k_x, k_y), order=order, mode=mode, cval=cval
    )

    # rotated_image = jnp.fft.ifftshift(jnp.fft.ifftn(rotated_image)).real
    return rotated_image
