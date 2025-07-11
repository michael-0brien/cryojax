"""
Routines to compute FFTs, in cryojax conventions.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ..internal import NDArrayLike


def ifftn(
    ft: Inexact[NDArrayLike, "..."],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
) -> Complex[Array, "..."]:
    """The equivalent of `jax.numpy.fft.ifftn` in `cryojax` conventions.

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
    ift = jnp.fft.fftshift(jnp.fft.ifftn(ft, s=s, axes=axes), axes=axes)

    return ift


def fftn(
    ift: Inexact[NDArrayLike, "..."],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
) -> Complex[Array, "..."]:
    """The equivalent of `jax.numpy.fft.fftn` in `cryojax` conventions.

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
    ft = jnp.fft.fftn(jnp.fft.ifftshift(ift, axes=axes), s=s, axes=axes)

    return ft


def irfftn(
    ft: Inexact[NDArrayLike, "..."],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
) -> Float[Array, "..."]:
    """The equivalent of `jax.numpy.fft.irfftn` in `cryojax` conventions.

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
    ift = jnp.fft.fftshift(jnp.fft.irfftn(ft, s=s, axes=axes), axes=axes)

    return ift


def rfftn(
    ift: Float[NDArrayLike, "..."],
    s: Optional[tuple[int, ...]] = None,
    axes: Optional[tuple[int, ...]] = None,
) -> Complex[Array, "..."]:
    """The equivalent of `jax.numpy.fft.rfftn` in `cryojax` conventions.

    Arguments
    ---------
    ift :
        Array in real space.

    Returns
    -------
    ft :
        Fourier transform of array.
    """
    ft = jnp.fft.rfftn(jnp.fft.ifftshift(ift, axes=axes), s=s, axes=axes)

    return ft
