from typing import ClassVar
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import FloatLike, NDArrayLike
from ...rotations import SO2
from .._fourier_utils import enforce_rfftn_self_conjugates, make_fftshift_phase
from .._interpolation import map_frequencies
from .._operators import FourierPhaseShifts
from ._base_transform import AbstractImageTransform


class PhaseShiftFFT(AbstractImageTransform, strict=True):
    """Apply a phase shift to an image in Fourier space, effectively
    applying an in-plane shift to the image in real space. Only square
    images are supported.

    !!! example "Apply a translation in real-space"

        ```python

        import jax.numpy as jnp
        from cryojax.ndimage import PhaseShiftFFT

        offset_in_angstroms = jnp.array([50.0, -30.0])
        frequency_grid = ... # in angstroms
        fft = jnp.fft.rfftn(...) # e.g., fft of a real 2D image

        shift_fn = PhaseShiftFFT(
            offset=offset_in_angstroms, frequency_grid=frequency_grid
        )

        shifted_fft = shift_fn(fft)
        shifted_image = jnp.fft.irfftn(shifted_image_fft)
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
    Only square, even-dimension images are supported.

    Rotation is done by interpolating the image's fourier transform with
    [`cryojax.ndimage.map_frequencies`][].

    !!! example

        ```python

        import jax.numpy as jnp
        from cryojax.ndimage import RotateFFT, make_frequency_grid

        image = ...  # e.g., a real 2D image of shape (dim, dim)
        frequency_grid = make_frequency_grid((dim, dim))  # in pixels

        rotation_fn = RotateFFT(
            rotation_angle=45.0, frequency_grid=frequency_grid
        )

        rotated_image_fft = jnp.fft.irfftn(
            rotation_fn(jnp.fft.rfftn(image)), s=image.shape
        )
        ```
    """

    rotation_angle: Float[Array, ""]
    frequency_grid: Float[Array, "_ _ 2"]

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        rotation_angle: FloatLike,
        frequency_grid: Float[NDArrayLike, "y_dim x_dim 2"],
        *,
        pixel_size: FloatLike = 1.0,
    ):
        """
        **Arguments:**

        - `rotation_angle`: The angle by which to rotate the image, in degrees.
        - `frequency_grid`:
            The frequency grid, of the half-space (rfft) shape
            `(dim, dim // 2 + 1, 2)`, as returned by
            [`cryojax.ndimage.make_frequency_grid`][].
        - `pixel_size`: The pixel size of the `frequency_grid`.
        """  # noqa: E501
        if not _is_square_rfft_grid(frequency_grid, only_2d=True):
            raise ValueError(
                "The `frequency_grid` argument to `RotateFFT` did not have a valid "
                f"shape {frequency_grid.shape}. `RotateFFT` only supports square 2D "
                "images stored as a half-space (rfft) DFT, so `frequency_grid` must "
                "have shape `(dim, dim // 2 + 1, 2)` -- as returned by "
                "`cryojax.ndimage.make_frequency_grid((dim, dim))`."
            )
        self.rotation_angle = jnp.asarray(rotation_angle, dtype=float)
        self.frequency_grid = jnp.asarray(frequency_grid * pixel_size, dtype=float)

    @override
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Rotate the input image in Fourier space.

        **Arguments:**

        `image`:
            The image in Fourier space, i.e. the output of
            `jax.numpy.fft.rfftn`.

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
        dim = image.shape[0]
        if dim % 2 == 1:
            raise ValueError(
                "Only even parity images are supported in `RotateFFT`. Got "
                f"an image of shape `{image.shape}`."
            )
        # Shift image and grid so that zero is in the center. Only the full axis
        # is shifted; the rfft-truncated axis stays in the corner convention.
        factors = make_fftshift_phase((dim, dim), outputs_rfft=True)
        fourier_image_c = jnp.fft.fftshift(factors * image, axes=(0,))
        frequency_grid_c = jnp.fft.fftshift(self.frequency_grid, axes=(0,))
        # Rotate the grid
        rotated_grid = frequency_grid_c @ _get_rotation_matrix(self.rotation_angle)
        # Interpolate at the rotated frequencies. `q_x` may be negative -- those
        # frequencies are not stored in the half space, but `map_frequencies`
        # recovers them exactly from the DFT's Hermitian symmetry.
        rotated_image_c = map_frequencies(fourier_image_c, rotated_grid)
        # Shift back, ensure that rfft components are real-valued where they
        # should be, and return
        rotated_image = jnp.fft.ifftshift(rotated_image_c, axes=(0,))
        rotated_image = enforce_rfftn_self_conjugates(
            rotated_image, (dim, dim), includes_dc=True, mode="real"
        )
        return factors * rotated_image


def _get_rotation_matrix(angle):
    """The in-plane rotation matrix for `angle` degrees, in the 'object'
    convention: the rotation is with respect to the object in the image."""
    angle = jnp.deg2rad(-angle)
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = SO2([c, s])
    return rotation.as_matrix()


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
