import jax
import jax.numpy as jnp
import pytest

from cryojax.coordinates import make_coordinate_grid, make_frequency_grid
from cryojax.ndimage import transforms as tf


def test_mask_2d_running():
    classes = [
        tf.InverseSincMask,
        tf.SquareCosineMask,
        tf.CircularCosineMask,
        tf.Cylindrical2DCosineMask,
        tf.Rectangular2DCosineMask,
    ]
    kwargs = [
        dict(),
        dict(side_length=5, rolloff_width=2),
        dict(radius=5, rolloff_width=2),
        dict(radius=5, rolloff_width=2, length=5, in_plane_rotation_angle=2.0),
        dict(x_width=5, y_width=5, rolloff_width=2, in_plane_rotation_angle=2.0),
    ]
    coordinate_grid = make_coordinate_grid((10, 10))
    image = jnp.zeros((10, 10))
    for i, cls in enumerate(classes):
        mask = cls(coordinate_grid, **kwargs[i])
        _ = mask.get()
        _ = mask(image)


def test_mask_3d_running():
    classes = [tf.InverseSincMask, tf.SphericalCosineMask, tf.Rectangular3DCosineMask]
    kwargs = [
        dict(),
        dict(radius=5, rolloff_width=2),
        dict(x_width=5, y_width=5, z_width=5, rolloff_width=2),
    ]
    coordinate_grid = make_coordinate_grid((10, 10, 10))
    image = jnp.zeros((10, 10, 10))
    for i, cls in enumerate(classes):
        mask = cls(coordinate_grid, **kwargs[i])
        _ = mask.get()
        _ = mask(image)


def test_filter_running():
    classes = [tf.LowpassFilter, tf.HighpassFilter]
    kwargs = [dict(), dict()]
    frequency_grid_2d, fourier_image_2d = (
        make_frequency_grid((10, 10)),
        jnp.zeros((10, 10 // 2 + 1)),
    )
    frequency_grid_3d, fourier_image_3d = (
        make_frequency_grid((10, 10, 10)),
        jnp.zeros((10, 10, 10 // 2 + 1)),
    )
    for i, cls in enumerate(classes):
        f_2d = cls(frequency_grid_2d, **kwargs[i])
        _ = f_2d.get()
        _ = f_2d(fourier_image_2d)
        f_3d = cls(frequency_grid_3d, **kwargs[i])
        _ = f_3d.get()
        _ = f_3d(fourier_image_3d)


def test_custom_filter_and_mask_initialization():
    classes = [tf.CustomFilter, tf.CustomMask]
    array = jnp.zeros((10, 10))
    for cls in classes:
        _ = cls(array)


@pytest.mark.parametrize(
    "image_shape, filter_shape, mode, square",
    (
        ((10, 10), None, "linear", False),
        ((2, 10, 10), None, "linear", False),
        ((2, 10, 10), (9, 9), "linear", False),
        ((2, 10, 10), (11, 11), "linear", False),
        ((2, 10, 10), None, "nearest", False),
        ((2, 10, 10), None, "linear", True),
    ),
)
def test_whitening_filter(image_shape, filter_shape, mode, square):
    rng_key = jax.random.key(1234)
    image = jax.random.normal(rng_key, image_shape)
    f = tf.WhiteningFilter(
        image, shape=filter_shape, interpolation_mode=mode, outputs_squared=square
    )
    _ = f.get()
