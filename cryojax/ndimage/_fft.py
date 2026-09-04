"""
Routines to compute FFTs, in cryojax conventions.
"""

import warnings

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ..jax_util import NDArrayLike


def _warn_fft_deprecated(name: str):
    warnings.warn(
        f"`cryojax.ndimage.{name}` is deprecated and will be removed in a future "
        f"release. Use `jax.numpy.fft.{name}` instead.",
        FutureWarning,
        stacklevel=3,
    )


def ifftn(
    ft: Inexact[NDArrayLike, "..."],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> Complex[Array, "..."]:
    """The equivalent of `jax.numpy.fft.ifftn` in `cryojax` conventions.

    !!! warning "Deprecated"
        This function is deprecated and will be removed in a future release. Use
        `jax.numpy.fft.ifftn` instead.

    Arguments
    ---------
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the corner.

    Returns
    -------
    ift :
        Inverse fourier transform.
    """
    _warn_fft_deprecated("ifftn")
    ift = jnp.fft.ifftn(ft, s=s, axes=axes)

    return ift


def fftn(
    ift: Inexact[NDArrayLike, "..."],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> Complex[Array, "..."]:
    """The equivalent of `jax.numpy.fft.fftn` in `cryojax` conventions.

    !!! warning "Deprecated"
        This function is deprecated and will be removed in a future release. Use
        `jax.numpy.fft.fftn` instead.

    Arguments
    ---------
    ift :
        Array in real space. Assumes that the zero
        frequency component is in the center.

    Returns
    -------
    ft :
        Fourier transform of array.
    """
    _warn_fft_deprecated("fftn")
    ft = jnp.fft.fftn(ift, s=s, axes=axes)

    return ft


def irfftn(
    ft: Inexact[NDArrayLike, "..."],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> Float[Array, "..."]:
    """The equivalent of `jax.numpy.fft.irfftn` in `cryojax` conventions.

    !!! warning "Deprecated"
        This function is deprecated and will be removed in a future release. Use
        `jax.numpy.fft.irfftn` instead.

    Arguments
    ---------
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the corner.

    Returns
    -------
    ift :
        Inverse fourier transform.
    """
    _warn_fft_deprecated("irfftn")
    ift = jnp.fft.irfftn(ft, s=s, axes=axes)

    return ift


def rfftn(
    ift: Float[NDArrayLike, "..."],
    s: tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> Complex[Array, "..."]:
    """The equivalent of `jax.numpy.fft.rfftn` in `cryojax` conventions.

    !!! warning "Deprecated"
        This function is deprecated and will be removed in a future release. Use
        `jax.numpy.fft.rfftn` instead.

    Arguments
    ---------
    ift :
        Array in real space.

    Returns
    -------
    ft :
        Fourier transform of array.
    """
    _warn_fft_deprecated("rfftn")
    ft = jnp.fft.rfftn(ift, axes=axes, s=s)

    return ft
