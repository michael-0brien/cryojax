from typing import ClassVar, Literal
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...jax_util import FloatLike, NDArrayLike
from .. import enforce_rfftn_self_conjugates
from .._map_coordinates import map_coordinates
from .._operators import FourierPhaseShifts
from ._base_transform import AbstractImageTransform


class PhaseShiftFFT(AbstractImageTransform, strict=True):
    """Apply a phase shift to an image in Fourier space, effectively
    applying an in-plane shift to the image in real space.

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
    is_rfft: bool

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
        shape, dim = frequency_grid.shape, frequency_grid.shape[0]
        print(shape)
        if shape == (dim, dim // 2 + 1, 2) or shape == (dim, dim, dim // 2 + 1, 3):
            # It's worth noting that this condition is breakable and rfftn/fftn
            # inference can fail! One can pass a grid meant for use with fftn
            # with these exact shapes.
            self.is_rfft = True
        elif shape == (dim, dim, 2) or shape == (dim, dim, dim, 3):
            self.is_rfft = False
        else:
            raise ValueError(
                "The `frequency_grid` argument to `PhaseShiftFFT` did not have a valid "
                f"shape {shape}. `PhaseShiftFFT` only supports square images as input; "
                "you may have passed a grid that does not correspond to a square image."
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
    """Rotate an image in Fourier space. This corresponds to a rotation
    in real space as well.

    **Example:**
    ```python

    from cryojax.ndimage import RotateFFT, fftn, ifftn

    frequency_grid = ... # in pixels
    image = ... # e.g., a real 2D image

    rotation_fn = RotateFFT(
        rotation_angle=45.0,
        frequency_grid=frequency_grid,
    )

    rotated_image_fft = rotation_fn(fftn(image))
    rotated_image = ifftn(rotated_image_fft).real
    ```
    """

    rotation_angle: Float[Array, ""]
    frequency_grid: Float[Array, "_ _ 2"] | Float[Array, "_ _ _ 3"]
    order: Literal[0, 1]
    mode: str
    cval: float | complex

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        rotation_angle: FloatLike,
        frequency_grid: (
            Float[NDArrayLike, "y_dim x_dim 2"]
            | Float[NDArrayLike, "z_dim y_dim x_dim 3"]
        ),
        *,
        pixel_size: FloatLike = 1.0,
        order: Literal[0, 1] = 1,
        mode: str = "fill",
        cval: float | complex = 0.0,
    ):
        """
        **Arguments:**

        - `rotation_angle`: The angle by which to rotate the image, in degrees.
        - `frequency_grid`: The frequency grid.
        - `pixel_size`: The pixel size of the `frequency_grid`.
        - `order`: The interpolation order.
            See [`cryojax.ndimage.map_coordinates`][] for details.
        - `mode`: The mode to use for points outside the boundaries.
            See [`cryojax.ndimage.map_coordinates`][] for details.
        - `cval`: The constant value to use when `mode` is 'fill'.
            See [`cryojax.ndimage.map_coordinates`][] for details.

        !!! Warning
            Only square frequency grids are currently supported.

        """
        assert order in (0, 1), "Only order 0 and 1 are supported."

        if not (frequency_grid.shape[0] == frequency_grid.shape[1]):
            raise NotImplementedError(
                "Only square frequency grids are currently supported."
            )

        self.rotation_angle = jnp.asarray(rotation_angle)
        self.frequency_grid = jnp.asarray(frequency_grid * pixel_size)
        self.order = order
        self.mode = mode
        self.cval = cval

    @override
    def __call__(
        self, image: Complex[Array, "y_dim x_dim"]
    ) -> Complex[Array, "y_dim x_dim"]:
        """Rotate the input image in Fourier space.

        **Arguments:**
            image: The input image in Fourier space.

        **Returns:**
            The rotated image in Fourier space.
        """
        rotated_image = _rotate_fourier_image(
            image,
            rotation_angle=self.rotation_angle,
            frequency_grid=self.frequency_grid,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )
        return rotated_image


def _rotate_fourier_image(
    fourier_image: Complex[Array, "y_dim x_dim"],
    rotation_angle: FloatLike,
    frequency_grid: Float[Array, "y_dim x_dim 2"],
    order: Literal[0, 1] = 1,
    mode: str = "fill",
    cval: float | complex = 0.0,
) -> Complex[Array, "y_dim x_dim"]:
    angle = jnp.deg2rad(rotation_angle)
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    # shift images and grid to center
    fourier_image = jnp.fft.fftshift(fourier_image, axes=(0, 1))
    frequency_grid = jnp.fft.fftshift(frequency_grid, axes=(0, 1))
    rotation_matrix = jnp.array([[c, -s], [s, c]])

    rotated_grid = frequency_grid @ rotation_matrix
    N = rotated_grid.shape[1]
    logical_frequency_coordinates = rotated_grid.at[...].set(rotated_grid * N + N // 2)
    k_y, k_x = jnp.transpose(logical_frequency_coordinates, axes=[2, 0, 1])

    rotated_image = map_coordinates(
        fourier_image, (k_x, k_y), order=order, mode=mode, cval=cval
    )

    rotated_image = jnp.fft.fftshift(rotated_image, axes=(0, 1))
    return rotated_image
