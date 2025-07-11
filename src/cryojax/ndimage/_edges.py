"""
Routines for dealing with image cropping and padding.
"""

import warnings
from typing import Any, overload

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, Int


@overload
def crop_to_shape(
    image_or_volume: Inexact[Array, "y_dim x_dim"],
    shape: tuple[int, int],
) -> Inexact[Array, " {shape[0]} {shape[1]}"]: ...


@overload
def crop_to_shape(
    image_or_volume: Inexact[Array, "z_dim y_dim x_dim"],
    shape: tuple[int, int, int],
) -> Inexact[Array, " {shape[0]} {shape[1]} {shape[2]}"]: ...


def crop_to_shape(
    image_or_volume: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    shape: tuple[int, int] | tuple[int, int, int],
) -> (
    Inexact[Array, " {shape[0]} {shape[1]}"]
    | Inexact[Array, " {shape[0]} {shape[1]} {shape[2]}"]
):
    """Crop an image or volume to a new shape around its
    center.
    """
    if image_or_volume.ndim not in [2, 3]:
        raise ValueError(
            "crop_to_shape can only crop images and volumes. Got array shape of "
            f"{image_or_volume.shape}."
        )
    if len(shape) != len(image_or_volume.shape):
        raise ValueError(
            "Mismatch between ndim of desired crop shape and "
            f"array shape. Got a crop shape of {shape} and "
            f"an array shape of {image_or_volume.shape}."
        )
    if len(shape) == 2:
        image = image_or_volume
        Ny, Nx = image.shape
        xc, yc = Nx // 2, Ny // 2
        h, w = shape
        cropped = image[
            yc - h // 2 : yc + h // 2 + h % 2,
            xc - w // 2 : xc + w // 2 + w % 2,
        ]
    elif len(shape) == 3:
        volume = image_or_volume
        Nz, Ny, Nx = volume.shape
        xc, yc, zc = Nx // 2, Ny // 2, Nz // 2
        d, h, w = shape
        cropped = volume[
            zc - d // 2 : zc + d // 2 + d % 2,
            yc - h // 2 : yc + h // 2 + h % 2,
            xc - w // 2 : xc + w // 2 + w % 2,
        ]
    else:
        raise ValueError(
            "crop_to_shape can only crop images and volumes. Got desired crop shape of "
            f"of {shape}."
        )
    return cropped


def crop_to_shape_with_center(
    image: Inexact[Array, "y_dim x_dim"],
    shape: tuple[int, int],
    center: tuple[int, int] | tuple[Int[Array, ""], Int[Array, ""]],
    do_safe_crop: bool = True,
) -> Inexact[Array, "{shape[0]} {shape[1]}"]:
    """Crop an image to a new shape, given a center."""
    if image.ndim != 2:
        raise ValueError(
            "crop_to_shape_with_center can only crop images. Got array shape of "
            f"{image.shape}."
        )
    if len(shape) == 2:
        xc, yc = center
        h, w = shape
        if do_safe_crop:
            if not isinstance(xc, int) or not isinstance(yc, int):
                raise ValueError(
                    "If `do_safe_crop = True`, then `center` must be passed as "
                    "a tuple of python integers. Found instead a tuple with "
                    f"type {type(xc).__name__}."
                )
            x0, y0 = max(xc - w // 2, 0), max(yc - h // 2, 0)
            xn, yn = (
                min(xc + w // 2 + w % 2, image.shape[1] - 1),
                min(yc + h // 2 + h % 2, image.shape[0] - 1),
            )
            cropped = image[y0:yn, x0:xn]
        else:
            x0, y0 = jnp.asarray(xc - w // 2), jnp.asarray(yc - h // 2)
            cropped = jax.lax.dynamic_slice(image, (y0, x0), shape)
    else:
        raise ValueError(
            "`crop_to_shape_with_center` can only crop images. Got desired crop shape of "
            f"{shape}."
        )
    if cropped.shape != shape:
        warnings.warn(
            "The cropped shape is not equal to the desired shape in "
            "crop_to_shape_with_center. Usually, this happens because the crop was "
            "near the image edges."
        )
    return cropped


@overload
def pad_to_shape(
    image_or_volume: Inexact[Array, "y_dim x_dim"],
    shape: tuple[int, int],
    **kwargs: Any,
) -> Inexact[Array, " {shape[0]} {shape[1]}"]: ...


@overload
def pad_to_shape(
    image_or_volume: Inexact[Array, "z_dim y_dim x_dim"],
    shape: tuple[int, int, int],
    **kwargs: Any,
) -> Inexact[Array, " {shape[0]} {shape[1]} {shape[2]}"]: ...


def pad_to_shape(
    image_or_volume: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
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
            "pad_to_shape can only pad images and volumes. Got desired shape of "
            f"{shape}."
        )
    return jnp.pad(image_or_volume, padding, **kwargs)


def resize_with_crop_or_pad(
    image: Inexact[Array, "y_dim x_dim"], shape: tuple[int, int], **kwargs
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
