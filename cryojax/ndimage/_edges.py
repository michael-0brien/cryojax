"""
Routines for dealing with image cropping and padding.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, Int

from ..jax_util import NDArrayLike


def crop_to_shape(
    image_or_volume: (
        Inexact[NDArrayLike, "y_dim x_dim"] | Inexact[NDArrayLike, "z_dim y_dim x_dim"]
    ),
    shape: tuple[int, int] | tuple[int, int, int],
    center: (
        tuple[int, int]
        | tuple[int, int, int]
        | tuple[Int[NDArrayLike, ""], Int[NDArrayLike, ""]]
        | tuple[Int[NDArrayLike, ""], Int[NDArrayLike, ""], Int[NDArrayLike, ""]]
        | None
    ) = None,
) -> (
    Inexact[Array, " {shape[0]} {shape[1]}"]
    | Inexact[Array, " {shape[0]} {shape[1]} {shape[2]}"]
):
    """Crop an image or volume to a new shape around its
    center.

    !!! info

        The behavior of this function is slightly different
        depending on the type of `center`. If `center` is
        a tuple of `jax.Array`s, wraps `jax.lax.dynamic_slice`.
        If `center` is a tuple of `int` or `numpy.ndarrays`,
        crops with array indexing and throws an error if
        the resulting shape isn't equal to the requested shape.
    """
    ndim = image_or_volume.ndim
    if ndim not in [2, 3]:
        raise ValueError(
            "`cryojax.ndimage.crop_to_shape` can only crop images "
            f" and volumes, but got array shape of {image_or_volume.shape}."
        )
    if len(shape) != len(image_or_volume.shape):
        raise ValueError(
            "Mismatch between number of dimensions of desired "
            "crop shape and image or volume shape. "
            f"Desired crop shape was {shape} and "
            f"image or volume shape was {image_or_volume.shape}."
        )
    if center is None:
        center = tuple(n // 2 for n in image_or_volume.shape[::-1])  # type: ignore
        assert center is not None
    if any(isinstance(x, Array) for x in center):
        start_indices = tuple(
            jnp.asarray(x - s // 2) for x, s in zip(center[::-1], shape)
        )
        cropped = jax.lax.dynamic_slice(image_or_volume, start_indices, shape)
    else:
        if len(shape) == 2:
            assert len(center) == 2
            image = image_or_volume
            xc, yc = center
            h, w = shape
            x0, y0 = (max(xc - w // 2, 0), max(yc - h // 2, 0))
            xn, yn = (
                min(xc + w // 2 + w % 2, image.shape[1]),
                min(yc + h // 2 + h % 2, image.shape[0]),
            )
            cropped = image[y0:yn, x0:xn]
        else:
            assert len(center) == 3
            volume = image_or_volume
            xc, yc, zc = center
            d, h, w = shape
            x0, y0, z0 = (max(xc - w // 2, 0), max(yc - h // 2, 0), max(zc - d // 2, 0))
            xn, yn, zn = (
                min(xc + w // 2 + w % 2, volume.shape[2]),
                min(yc + h // 2 + h % 2, volume.shape[1]),
                min(zc + d // 2 + d % 2, volume.shape[0]),
            )
            cropped = volume[z0:zn, y0:yn, x0:xn]
    if cropped.shape != tuple(shape):
        raise ValueError(
            f"The cropped shape {cropped.shape} was not equal to the desired shape "
            f"{shape} in "
            "`cryojax.ndimage.crop_to_shape`. This can happen if the crop is "
            "near the image edges."
        )
    return jnp.asarray(cropped)


def pad_to_shape(
    image_or_volume: (
        Inexact[NDArrayLike, "y_dim x_dim"] | Inexact[NDArrayLike, "z_dim y_dim x_dim"]
    ),
    shape: tuple[int, int] | tuple[int, int, int],
    **kwargs: Any,
) -> (
    Inexact[Array, " {shape[0]} {shape[1]}"]
    | Inexact[Array, " {shape[0]} {shape[1]} {shape[2]}"]
):
    """Pad an image or volume to a new shape."""
    if image_or_volume.ndim not in [2, 3]:
        raise ValueError(
            "pad_to_shape can only pad images and volumes. Got array shape "
            f"of {image_or_volume.shape}."
        )
    if len(shape) != len(image_or_volume.shape):
        raise ValueError(
            "Mismatch between ndim of desired shape and "
            f"array shape. Got a shape of {shape} after padding and "
            f"an array shape of {image_or_volume.shape}."
        )
    if len(shape) == 2:
        image = image_or_volume
        y_padding = _get_left_vs_right_pad(shape[0], image.shape[0])
        x_padding = _get_left_vs_right_pad(shape[1], image.shape[1])
        padding = (y_padding, x_padding)
    elif len(shape) == 3:
        volume = image_or_volume
        z_padding = _get_left_vs_right_pad(shape[0], volume.shape[0])
        y_padding = _get_left_vs_right_pad(shape[1], volume.shape[1])
        x_padding = _get_left_vs_right_pad(shape[2], volume.shape[2])
        padding = (z_padding, y_padding, x_padding)
    else:
        raise ValueError(
            f"pad_to_shape can only pad images and volumes. Got desired shape of {shape}."
        )
    return jnp.pad(image_or_volume, padding, **kwargs)


def resize_with_crop_or_pad(
    image: Inexact[NDArrayLike, "y_dim x_dim"], shape: tuple[int, int], **kwargs
) -> Inexact[Array, " {shape[0]} {shape[1]}"]:
    """Resize an image to a new shape using padding and cropping."""
    if image.ndim != 2 or len(shape) != 2:
        raise ValueError(
            "resize_with_crop_or_pad can only resize images. Got array shape "
            f"of {image.shape} and desired shape {shape}."
        )
    N1, N2 = image.shape
    M1, M2 = shape
    if N1 >= M1 and N2 >= M2:
        image = crop_to_shape(image, shape)
    elif N1 <= M1 and N2 <= M2:
        image = pad_to_shape(image, shape, **kwargs)
    elif N1 <= M1 and N2 >= M2:
        image = crop_to_shape(image, (N1, M2))
        image = pad_to_shape(image, (M1, M2), **kwargs)
    else:
        image = crop_to_shape(image, (M1, N2))
        image = pad_to_shape(image, (M1, M2), **kwargs)

    return image


def _get_left_vs_right_pad(pad_dim, image_dim):
    pad = pad_dim - image_dim
    if pad_dim % 2 == 0 and image_dim % 2 == 0:
        pad_l, pad_r = (pad // 2, pad // 2)
    elif pad_dim % 2 == 1 and image_dim % 2 == 0:
        pad_l, pad_r = (pad // 2, pad // 2 + 1)
    elif pad_dim % 2 == 0 and image_dim % 2 == 1:
        pad_l, pad_r = (pad // 2 + 1, pad // 2)
    else:
        pad_l, pad_r = (pad // 2, pad // 2)
    return pad_l, pad_r
