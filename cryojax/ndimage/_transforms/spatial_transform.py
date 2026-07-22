from typing import ClassVar
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..._internal import leaf_asarray
from ...jax_util import FloatLike, NDArrayLike
from ...rotations import SO2
from .._coordinates import make_1d_frequency_grid, make_frequency_grid
from .._fourier_utils import enforce_rfftn_self_conjugates, make_fftshift_phase
from .._interpolation import map_frequencies
from .._operators import FourierPhaseShifts
from .base_transform import AbstractImageTransform


class PhaseShiftFFT(AbstractImageTransform, strict=True):
    """Apply a phase shift to an image in Fourier space, effectively
    applying an in-plane shift to the image in real space. Only square
    images are supported.

    !!! example "Apply a translation in real-space"

        ```python

        import jax.numpy as jnp
        from cryojax.ndimage import PhaseShiftFFT

        offset_in_angstroms = jnp.array([50.0, -30.0])
        fft = jnp.fft.rfftn(...) # e.g., fft of a real 2D image

        shift_fn = PhaseShiftFFT(
            offset=offset_in_angstroms, pixel_size=1.1
        )

        shifted_fft = shift_fn(fft)
        shifted_image = jnp.fft.irfftn(shifted_image_fft)
        ```
    """

    offset: Float[NDArrayLike, "2"] | Float[NDArrayLike, "3"]

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        offset: Float[NDArrayLike, "2"] | Float[NDArrayLike, "3"],
        *,
        pixel_size: FloatLike = 1.0,
    ):
        """**Arguments:**

        - `offset`: The offset by which to shift the image, in pixels or angstroms.
        - `pixel_size`: The pixel size of the image. Set `pixel_size` if `offset`
            is given in Angstroms, and leave as `1.0` if `offset` is given in
            pixel units.
        """
        self.offset = leaf_asarray(offset, dtype=float) / pixel_size

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
        ndim, dim = image.ndim, image.shape[0]
        offset = jnp.asarray(self.offset)
        if offset.size != ndim:
            raise ValueError(
                "The image passed to `PhaseShiftFFT` had dimensionality "
                f"`{ndim}`, but `PhaseShiftFFT.offset` was an array of "
                f"size {offset.size}."
            )
        # Infer whether `image` is an rfftn or fftn output from its shape.
        # It's worth noting that this inference can fail! An fftn output of
        # shape (2, 2) also has the rfftn shape (dim, dim // 2 + 1).
        if _is_square_rfft_shape(image.shape):
            is_rfft = True
        elif _is_square_fft_shape(image.shape):
            is_rfft = False
        else:
            raise ValueError(
                "The image passed to `PhaseShiftFFT` did not have a valid "
                f"shape {image.shape}. `PhaseShiftFFT` only supports square "
                "images stored as fftn or rfftn outputs."
            )
        # Build the phase factor as an outer product of 1D phase factors.
        # Image axes are ordered (y, x) or (z, y, x), while `offset` is
        # ordered (x, y) or (x, y, z).
        frequencies = make_1d_frequency_grid(dim, outputs_rfftfreqs=False)
        frequencies_x = make_1d_frequency_grid(dim, outputs_rfftfreqs=is_rfft)[
            : image.shape[-1]
        ]
        phase_x = FourierPhaseShifts(offset[0])(frequencies_x)
        phase_y = FourierPhaseShifts(offset[1])(frequencies)
        if ndim == 2:
            translation_operator = phase_y[:, None] * phase_x[None, :]
        else:
            phase_z = FourierPhaseShifts(offset[2])(frequencies)
            translation_operator = (
                phase_z[:, None, None] * phase_y[None, :, None] * phase_x[None, None, :]
            )
        if is_rfft:
            image = enforce_rfftn_self_conjugates(
                image,
                tuple(ndim * [dim]),  # pyright: ignore[reportArgumentType]
                includes_dc=False,
                mode="zero",
            )

        return image * translation_operator


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

    rotation_angle: Float[NDArrayLike, ""]
    frequency_grid: Float[Array, "_ _ 2"] | None

    is_real_space: ClassVar[bool] = False

    def __init__(
        self,
        rotation_angle: FloatLike,
        *,
        frequency_grid: Float[NDArrayLike, "y_dim x_dim 2"] | None = None,
        pixel_size: FloatLike = 1.0,
    ):
        """
        **Arguments:**

        - `rotation_angle`: The angle by which to rotate the image, in degrees.
        - `frequency_grid`:
            The frequency grid, of the half-space (rfft) shape
            `(dim, dim // 2 + 1, 2)`, as returned by
            [`cryojax.ndimage.make_frequency_grid`][]. If not provided,
            generate on-the-fly.
        - `pixel_size`: The pixel size of the `frequency_grid`.
        """  # noqa: E501
        if frequency_grid is not None and not (
            frequency_grid.ndim == 3
            and frequency_grid.shape[-1] == 2
            and _is_square_rfft_shape(frequency_grid.shape[:-1])
        ):
            raise ValueError(
                "The `frequency_grid` argument to `RotateFFT` did not have a valid "
                f"shape {frequency_grid.shape}. `RotateFFT` only supports square 2D "
                "images stored as a half-space (rfft) DFT, so `frequency_grid` must "
                "have shape `(dim, dim // 2 + 1, 2)` -- as returned by "
                "`cryojax.ndimage.make_frequency_grid((dim, dim))`."
            )
        self.rotation_angle = leaf_asarray(rotation_angle, dtype=float)
        self.frequency_grid = (
            None
            if frequency_grid is None
            else jnp.asarray(frequency_grid * pixel_size, dtype=float)
        )

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
        if not (image.ndim == 2 and _is_square_rfft_shape(image.shape)):
            raise ValueError(
                "The `image` argument to `RotateFFT.__call__` did not have a valid "
                f"shape {image.shape}. `RotateFFT` only supports square 2D "
                "images stored as a half-space (rfft) DFT, so `image` must "
                "have shape `(dim, dim // 2 + 1)`."
            )
        dim = image.shape[0]
        if self.frequency_grid is None:
            frequencies = make_frequency_grid((dim, dim), fftshifted=True)
        else:
            frequencies = jnp.fft.fftshift(self.frequency_grid, axes=(0,))
            if image.shape != frequencies.shape[0:-1]:
                raise ValueError(
                    "The image passed to `RotateFFT` did not have a valid "
                    f"shape. The shape of the image was {image.shape}, "
                    "but that of the `frequency_grid` via "
                    "`RotateFFT(..., frequency_grid=...)` was "
                    f"{frequencies.shape}."
                )
        if dim % 2 == 1:
            raise ValueError(
                "Only even parity images are supported in `RotateFFT`. Got "
                f"an image corresponding to shape `{(dim, dim)}`."
            )
        # Shift image and grid so that zero is in the center. Only the full axis
        # is shifted; the rfft-truncated axis stays in the corner convention.
        fftshift_phase = make_fftshift_phase((dim, dim), outputs_rfft=True)
        sampling_fft = jnp.fft.fftshift(fftshift_phase * image, axes=(0,))
        # Rotate the grid
        rotated_frequencies = frequencies @ _rotation_matrix(self.rotation_angle)
        # Interpolate at the rotated frequencies. `q_x` may be negative -- those
        # frequencies are not stored in the half space, but `map_frequencies`
        # recovers them exactly from the DFT's Hermitian symmetry.
        # Then shift back, ensure that rfft components are real-valued where they
        # should be, and return
        rotated_image = fftshift_phase * enforce_rfftn_self_conjugates(
            jnp.fft.ifftshift(
                map_frequencies(
                    sampling_fft,
                    (rotated_frequencies[..., 1], rotated_frequencies[..., 0]),
                ),
                axes=(0,),
            ),
            (dim, dim),
            includes_dc=True,
            mode="real",
        )
        return rotated_image


def _rotation_matrix(angle):
    """The in-plane rotation matrix for `angle` degrees, in the 'object'
    convention: the rotation is with respect to the object in the image."""
    angle = jnp.deg2rad(-angle)
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = SO2([c, s])
    return rotation.as_matrix()


def _is_square_rfft_shape(shape):
    ndim = len(shape)
    if ndim not in (2, 3):
        return False
    dim = shape[0]
    return shape in [
        (*(ndim - 1) * (dim,), dim // 2 + 1),
        (*(ndim - 1) * (dim,), dim // 2),
    ]


def _is_square_fft_shape(shape):
    ndim = len(shape)
    if ndim not in (2, 3):
        return False
    return shape == ndim * (shape[0],)
