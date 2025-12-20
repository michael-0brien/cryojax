"""
Masks to apply to images in real space.
"""

import abc
import functools
import operator
from typing import ClassVar
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import FloatLike, NDArrayLike
from ._base_transform import AbstractImageTransform


class AbstractMask(AbstractImageTransform, strict=True):
    """Base class for computing and applying an image mask."""

    is_real_space: ClassVar[bool] = True

    @abc.abstractmethod
    def get(self) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError

    def __call__(
        self, image: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        return image * self.get()


class CustomMask(AbstractMask, strict=True):
    """Pass a custom mask as an array."""

    array: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self, mask_array: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ):
        self.array = mask_array

    @override
    def get(self) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        return self.array


class CircularCosineMask(AbstractMask, strict=True):
    """Apply a circular mask to an image with a cosine
    soft-edge.
    """

    array: Float[Array, "y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid: Float[Array, "y_dim x_dim 2"],
        radius: FloatLike,
        rolloff_width: FloatLike,
        xy_offset: tuple[float, float] | Float[NDArrayLike, " 2"] = (0.0, 0.0),
    ):
        """**Arguments:**

        - `coordinate_grid`:
            The image coordinates.
        - `radius`:
            The radius of the circular mask.
        - `rolloff_width`:
            The rolloff width of the soft edge.
        """
        self.array = _compute_circular_or_spherical_mask(
            coordinate_grid,
            jnp.asarray(radius),
            jnp.asarray(rolloff_width),
            offset=jnp.asarray(xy_offset),
        )

    @override
    def get(self) -> Float[Array, "y_dim x_dim"]:
        return self.array


class SphericalCosineMask(AbstractMask, strict=True):
    """Apply a spherical mask to a volume with a cosine
    soft-edge.
    """

    array: Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"],
        radius: FloatLike,
        rolloff_width: FloatLike,
    ):
        """**Arguments:**

        - `coordinate_grid`:
            The volume coordinates.
        - `radius`:
            The radius of the spherical mask.
        - `rolloff_width`:
            The rolloff width of the soft edge.
        """
        self.array = _compute_circular_or_spherical_mask(
            coordinate_grid,
            jnp.asarray(radius),
            jnp.asarray(rolloff_width),
            offset=None,
        )

    @override
    def get(self) -> Float[Array, "z_dim y_dim x_dim"]:
        return self.array


class SquareCosineMask(AbstractMask, strict=True):
    """Apply a square mask to an image with a cosine
    soft-edge.
    """

    array: Float[Array, "y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid: Float[Array, "y_dim x_dim 2"],
        side_length: FloatLike,
        rolloff_width: FloatLike,
    ):
        """**Arguments:**

        - `coordinate_grid`:
            The image coordinates.
        - `side_length`:
            The side length of the square.
        - `rolloff_width`:
            The rolloff width of the soft edge.
        """
        self.array = _compute_square_mask(
            coordinate_grid, jnp.asarray(side_length), jnp.asarray(rolloff_width)
        )

    @override
    def get(self) -> Float[Array, "y_dim x_dim"]:
        return self.array


class Cylindrical2DCosineMask(AbstractMask, strict=True):
    """Apply a cylindrical mask to an image with a cosine
    soft-edge. This implements an infinite in-plane cylinder,
    rotated at a given angle.
    """

    array: Float[Array, "y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid: Float[Array, "y_dim x_dim 2"],
        radius: FloatLike,
        rolloff_width: FloatLike,
        in_plane_rotation_angle: FloatLike = 0.0,
        length: FloatLike | None = None,
    ):
        """**Arguments:**

        - `coordinate_grid`:
            The image coordinates.
        - `radius`:
            The radius of the cylinder.
        - `rolloff_width`:
            The rolloff width of the soft edge.
        - `in_plane_rotation_angle`:
            The in-plane rotation angle of the cylinder in degrees. By default,
            `0.0`.
        - `length`:
            The length of the cylinder. If `None`, do not mask the cylinder length-wise.
        """
        length = None if length is None else jnp.asarray(length)
        if length is None:
            self.array = _compute_cylindrical_mask_2d_without_length(
                coordinate_grid,
                jnp.asarray(radius),
                jnp.asarray(in_plane_rotation_angle),
                jnp.asarray(rolloff_width),
            )
        else:
            self.array = _compute_cylindrical_mask_2d_with_length(
                coordinate_grid,
                jnp.asarray(radius),
                length,
                jnp.asarray(in_plane_rotation_angle),
                jnp.asarray(rolloff_width),
            )

    @override
    def get(self) -> Float[Array, "y_dim x_dim"]:
        return self.array


class Rectangular2DCosineMask(AbstractMask, strict=True):
    """Apply a rectangular mask in 2D to an image with a cosine
    soft-edge. Optionally, rotate the rectangle by an angle.
    """

    array: Float[Array, "y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid: Float[Array, "y_dim x_dim 2"],
        x_width: FloatLike,
        y_width: FloatLike,
        rolloff_width: FloatLike,
        in_plane_rotation_angle: FloatLike = 0.0,
    ):
        """**Arguments:**

        - `coordinate_grid`:
            The image coordinates.
        - `x_width`:
            The width of the rectangle along the x-axis.
        - `y_width`:
            The width of the rectangle along the y-axis.
        - `rolloff_width`:
            The rolloff width of the soft edge.
        - `in_plane_rotation_angle`:
            The in-plane rotation angle of the rectangle in degrees. By default,
            `0.0`.
        """
        self.array = _compute_cylindrical_mask_2d_with_length(
            coordinate_grid,
            jnp.asarray(y_width / 2),
            jnp.asarray(x_width),
            jnp.asarray(in_plane_rotation_angle),
            jnp.asarray(rolloff_width),
        )

    @override
    def get(self) -> Float[Array, "y_dim x_dim"]:
        return self.array


class Rectangular3DCosineMask(AbstractMask, strict=True):
    """Apply a rectangular mask to a volume with a cosine
    soft-edge.
    """

    array: Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"],
        x_width: FloatLike,
        y_width: FloatLike,
        z_width: FloatLike,
        rolloff_width: FloatLike,
    ):
        """**Arguments:**

        - `coordinate_grid`:
            The volume coordinates.
        - `x_width`:
            The width of the rectangle along the x-axis.
        - `y_width`:
            The width of the rectangle along the y-axis.
        - `z_width`:
            The width of the rectangle along the z-axis.
        - `rolloff_width`:
            The rolloff width of the soft edge.
        """
        self.array = _compute_rectangular_mask_3d(
            coordinate_grid,
            jnp.asarray(x_width),
            jnp.asarray(y_width),
            jnp.asarray(z_width),
            jnp.asarray(rolloff_width),
        )

    @override
    def get(self) -> Float[Array, "z_dim y_dim x_dim"]:
        return self.array


class SincCorrectionMask(AbstractMask, strict=True):
    """Divide an image or volume by a 2D or 3D rectangular
    squared sinc function computed on the unit box. This is used
    for correcting scaling in [`cryojax.simulator.FourierSliceExtraction`][].

    Linear interpolation in the Fourier domain can be thought of as
    a convolution with a triangular kernel, whose Fourier
    transform pair is the squared sinc. This mask
    acts to deconvolve this function with a division in real-space.
    """

    array: Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]

    def __init__(
        self,
        coordinate_grid_in_pixels: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ):
        """**Arguments:**

        - `coordinate_grid_in_pixels`:
            The image or volume coordinates. This should be
            generated with `make_coordinate_grid(shape)`, *not*
            `make_coordinate_grid(shape, grid_spacing)`.
        """
        ndim = coordinate_grid_in_pixels.ndim - 1
        box_dims = coordinate_grid_in_pixels.shape[0:ndim][::-1]
        self.array = (
            jax.lax.reciprocal(
                functools.reduce(
                    operator.mul,
                    [
                        jnp.sinc(coordinate_grid_in_pixels[..., i] / box_dims[i])
                        for i in range(ndim)
                    ],
                )
            )
            ** 2
        )

    @override
    def get(self) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        return self.array


def _compute_circular_or_spherical_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"],
    radius: Float[Array, ""],
    rolloff_width: Float[Array, ""],
    offset: Float[Array, "2"] | Float[Array, "3"] | None = None,
) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
    if offset is not None:
        coordinate_grid -= offset[None, None, :]
    radial_coordinate_grid = jnp.linalg.norm(coordinate_grid, axis=-1)

    def compute_mask_at_coordinate(radial_coordinate):
        return jnp.where(
            radial_coordinate <= radius,
            1.0,
            jnp.where(
                radial_coordinate > radius + rolloff_width,
                0.0,
                0.5
                * (1 + jnp.cos(jnp.pi * (radial_coordinate - radius) / rolloff_width)),
            ),
        )

    compute_mask = (
        jax.vmap(jax.vmap(compute_mask_at_coordinate))
        if radial_coordinate_grid.ndim == 2
        else jax.vmap(jax.vmap(jax.vmap(compute_mask_at_coordinate)))
    )

    return compute_mask(radial_coordinate_grid)


def _compute_square_mask(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    side_length: Float[Array, ""],
    rolloff_width: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"]:
    is_in_square_fn = lambda abs_x, abs_y, s: jnp.logical_and(
        abs_x <= s / 2, abs_y <= s / 2
    )
    is_in_edge_fn = lambda abs_x_or_y, s, w: jnp.logical_and(
        abs_x_or_y > s / 2, abs_x_or_y < s / 2 + w
    )
    compute_edge_fn = lambda abs_x_or_y, s, w: 0.5 * (
        1 + jnp.cos(jnp.pi * (abs_x_or_y - s / 2) / w)
    )

    def compute_mask_at_coordinate(coordinate):
        x, y = coordinate
        abs_x, abs_y = jnp.abs(x), jnp.abs(y)
        # Check coordinate is in either the square of the unmasked region
        is_in_unmasked_square = is_in_square_fn(abs_x, abs_y, side_length)
        # ... or the square of the unmasked region, plus the rolloff width
        # of the soft edge
        is_in_unmasked_plus_soft_edge_square = is_in_square_fn(
            abs_x, abs_y, side_length + 2 * rolloff_width
        )
        # Next, compute where (if anywhere) the coordinate is in the soft edge
        # region
        is_in_edge_x = is_in_edge_fn(abs_x, side_length, rolloff_width)
        is_in_edge_y = is_in_edge_fn(abs_y, side_length, rolloff_width)
        # Compute the soft edges
        edge_x, edge_y = (
            compute_edge_fn(abs_x, side_length, rolloff_width),
            compute_edge_fn(abs_y, side_length, rolloff_width),
        )

        return jnp.where(
            is_in_unmasked_square,
            1.0,
            jnp.where(
                is_in_unmasked_plus_soft_edge_square,
                jnp.where(
                    jnp.logical_and(is_in_edge_x, is_in_edge_y),
                    edge_x * edge_y,
                    jnp.where(is_in_edge_x, edge_x, edge_y),
                ),
                0.0,
            ),
        )

    compute_mask = jax.vmap(jax.vmap(compute_mask_at_coordinate))

    return compute_mask(coordinate_grid)


def _compute_cylindrical_mask_2d_without_length(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    radius: Float[Array, ""],
    angle: Float[Array, ""],
    rolloff_width: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"]:
    # Compute rotated radial coordinate grid
    angle_in_radians = jnp.deg2rad(angle)
    cos_angle = jnp.cos(angle_in_radians)
    sin_angle = jnp.sin(angle_in_radians)
    x, y = coordinate_grid[..., 0], coordinate_grid[..., 1]
    y_rotated = -x * sin_angle + y * cos_angle
    # ... the mask in 2D is just a rectangle; the cylindrical coordinate is
    # the rotated y coordinate
    cylinder_radial_coordinate_grid = jnp.sqrt(y_rotated**2)

    def compute_mask_at_coordinate(cylinder_radial_coordinate):
        r = cylinder_radial_coordinate
        return jnp.where(
            r < radius,
            1.0,
            jnp.where(
                r < radius + rolloff_width,
                0.5 * (1 + jnp.cos(jnp.pi * (r - radius) / rolloff_width)),
                0.0,
            ),
        )

    compute_mask = jax.vmap(jax.vmap(compute_mask_at_coordinate))

    return compute_mask(cylinder_radial_coordinate_grid)


def _compute_cylindrical_mask_2d_with_length(
    coordinate_grid: Float[Array, "y_dim x_dim 2"],
    radius: Float[Array, ""],
    length: Float[Array, ""],
    angle: Float[Array, ""],
    rolloff_width: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"]:
    # Compute rotated radial coordinate grid
    diameter = 2 * radius
    angle_in_radians = jnp.deg2rad(angle)
    cos_angle = jnp.cos(angle_in_radians)
    sin_angle = jnp.sin(angle_in_radians)
    x, y = coordinate_grid[..., 0], coordinate_grid[..., 1]
    x_rotated, y_rotated = x * cos_angle + y * sin_angle, -x * sin_angle + y * cos_angle
    cylinder_r_coordinate_grid = y_rotated
    cylinder_z_coordinate_grid = x_rotated

    is_in_rect_fn = lambda abs_x, abs_y, s_x, s_y: jnp.logical_and(
        abs_x <= s_x / 2, abs_y <= s_y / 2
    )
    is_in_edge_fn = lambda abs_x_or_y, s, w: jnp.logical_and(
        abs_x_or_y > s / 2, abs_x_or_y < s / 2 + w
    )
    compute_edge_fn = lambda abs_x_or_y, s, w: 0.5 * (
        1 + jnp.cos(jnp.pi * (abs_x_or_y - s / 2) / w)
    )

    def compute_mask_at_coordinate(r, z):
        abs_r, abs_z = jnp.abs(r), jnp.abs(z)
        # Check coordinate is in either the rectangle of the unmasked region
        is_in_unmasked_rect = is_in_rect_fn(abs_r, abs_z, diameter, length)
        # ... or the square of the unmasked region, plus the rolloff width
        # of the soft edge
        is_in_unmasked_plus_soft_edge_rect = is_in_rect_fn(
            abs_r, abs_z, diameter + 2 * rolloff_width, length + 2 * rolloff_width
        )
        # Next, compute where (if anywhere) the coordinate is in the soft edge
        # region
        is_in_edge_r = is_in_edge_fn(abs_r, diameter, rolloff_width)
        is_in_edge_z = is_in_edge_fn(abs_z, length, rolloff_width)
        # Compute the soft edges
        edge_r, edge_z = (
            compute_edge_fn(abs_r, diameter, rolloff_width),
            compute_edge_fn(abs_z, length, rolloff_width),
        )

        return jnp.where(
            is_in_unmasked_rect,
            1.0,
            jnp.where(
                is_in_unmasked_plus_soft_edge_rect,
                jnp.where(
                    jnp.logical_and(is_in_edge_r, is_in_edge_z),
                    edge_r * edge_z,
                    jnp.where(is_in_edge_r, edge_r, edge_z),
                ),
                0.0,
            ),
        )

    compute_mask = jax.vmap(jax.vmap(compute_mask_at_coordinate))

    return compute_mask(cylinder_r_coordinate_grid, cylinder_z_coordinate_grid)


def _compute_rectangular_mask_3d(
    coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"],
    x_width: Float[Array, ""],
    y_width: Float[Array, ""],
    z_width: Float[Array, ""],
    rolloff_width: Float[Array, ""],
):
    is_in_rect_fn = lambda abs_x, abs_y, abs_z, s_x, s_y, s_z: jnp.logical_and(
        jnp.logical_and(abs_x <= s_x / 2, abs_y <= s_y / 2), abs_z <= s_z / 2
    )
    is_in_edge_fn = lambda abs_x_or_y_or_z, s, w: jnp.logical_and(
        abs_x_or_y_or_z > s / 2, abs_x_or_y_or_z < s / 2 + w
    )
    compute_edge_fn = lambda abs_x_or_y_or_z, s, w: 0.5 * (
        1 + jnp.cos(jnp.pi * (abs_x_or_y_or_z - s / 2) / w)
    )

    def compute_mask_at_coordinate(coordinate):
        x, y, z = coordinate
        abs_x, abs_y, abs_z = jnp.abs(x), jnp.abs(y), jnp.abs(z)
        # Check coordinate is in either the rectangular of the unmasked region
        is_in_unmasked_rect = is_in_rect_fn(
            abs_x, abs_y, abs_z, x_width, y_width, z_width
        )
        # ... or the rectangle of the unmasked region, plus the rolloff width
        # of the soft edge
        is_in_unmasked_plus_soft_edge_rect = is_in_rect_fn(
            abs_x,
            abs_y,
            abs_z,
            x_width + 2 * rolloff_width,
            y_width + 2 * rolloff_width,
            z_width + 2 * rolloff_width,
        )
        # Next, compute where (if anywhere) the coordinate is in the soft edge
        # region
        is_in_edge_x = is_in_edge_fn(abs_x, x_width, rolloff_width)
        is_in_edge_y = is_in_edge_fn(abs_y, y_width, rolloff_width)
        is_in_edge_z = is_in_edge_fn(abs_z, z_width, rolloff_width)
        # Compute the soft edges
        edge_x, edge_y, edge_z = (
            compute_edge_fn(abs_x, x_width, rolloff_width),
            compute_edge_fn(abs_y, y_width, rolloff_width),
            compute_edge_fn(abs_z, z_width, rolloff_width),
        )

        return jnp.where(
            is_in_unmasked_rect,
            1.0,
            jnp.where(
                is_in_unmasked_plus_soft_edge_rect,
                jnp.where(
                    jnp.logical_and(
                        jnp.logical_and(is_in_edge_x, is_in_edge_y), is_in_edge_z
                    ),
                    edge_x * edge_y * edge_z,
                    jnp.where(
                        jnp.logical_and(is_in_edge_x, is_in_edge_y),
                        edge_x * edge_y,
                        jnp.where(
                            jnp.logical_and(is_in_edge_x, is_in_edge_z),
                            edge_x * edge_z,
                            jnp.where(
                                jnp.logical_and(is_in_edge_y, is_in_edge_z),
                                edge_y * edge_z,
                                jnp.where(
                                    is_in_edge_x,
                                    edge_x,
                                    jnp.where(is_in_edge_y, edge_y, edge_z),
                                ),
                            ),
                        ),
                    ),
                ),
                0.0,
            ),
        )

    compute_mask = jax.vmap(jax.vmap(jax.vmap(compute_mask_at_coordinate)))
    return compute_mask(coordinate_grid)
