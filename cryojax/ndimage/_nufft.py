"""Non-uniform FFTs of point sources, in cryoJAX conventions."""

from typing import Any

import jax.numpy as jnp
import nufftax
from jaxtyping import Array, Complex, Float

from .._config import CRYOJAX_FINUFFT_BACKEND
from ..jax_util import FloatLike


try:
    import jax_finufft
    from jax_finufft.options import NestedOpts, Opts

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jax_finufft, Opts, NestedOpts = None, None, None
    JAX_FINUFFT_IMPORT_ERROR = err


def dispatch_nufft1(
    positions: Float[Array, "M d"],
    strengths: Complex[Array, " M"],
    shape: tuple[int, ...],
    *,
    pixel_size: FloatLike = 1.0,
    fftshifted: bool = False,
    eps: float = 1e-6,
    upsampfac: float = 1.25,
    options: dict[str, Any] | None = None,
) -> Complex[Array, "*shape"]:
    """Compute the FFT of point sources, $\\sum_j c_j e^{-2 \\pi i k \\cdot r_j}$, with a
    type-1 non-uniform FFT.

    Unlike [`jax_finufft.nufft1`](https://github.com/flatironinstitute/jax-finufft),
    this takes positions rather than angles, measured from the grid center `N//2` as
    elsewhere in cryoJAX, and returns the modes in the order of
    `jax.numpy.fft.fftn`. The backend is set by the `CRYOJAX_FINUFFT_BACKEND`
    environment variable.

    **Arguments:**

    - `positions`:
        The positions of the points, as `(x, y)` or `(x, y, z)`, in the units of
        `pixel_size`.
    - `strengths`:
        The strength of each point.
    - `shape`:
        The shape of the output grid, as `(y, x)` or `(z, y, x)`.
    - `pixel_size`:
        The grid spacing.
    - `fftshifted`:
        If `True`, return the modes in the order of `jax.numpy.fft.fftshift`.
    - `eps`:
        The tolerance of the non-uniform FFT.
    - `upsampfac`:
        The upsampling factor of the non-uniform FFT's internal grid. The default of
        `1.25` is faster than `finufft`'s `2.0`, but does not reach small `eps` for
        signals with significant power near Nyquist.
    - `options`:
        Options for the backend, for advanced usage, which take precedence. See
        [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html).

    **Returns:**

    The FFT of the points, with shape `shape`.
    """
    ndim = len(shape)
    if ndim not in (2, 3) or positions.shape[-1] != ndim:
        raise ValueError(
            "Found invalid value for `dispatch_nufft1(positions, strengths, shape)`. The "
            "`shape` must be 2D or 3D, and `positions` must have one coordinate per "
            f"dimension, but got `shape = {shape}` and `positions.shape = "
            f"{positions.shape}`."
        )
    # Positions to angles in [-pi, pi), with the grid center at index N//2
    n = jnp.asarray(shape[::-1], dtype=float)
    offsets = jnp.asarray([2 * jnp.pi * (s // 2) / s for s in shape[::-1]])
    angles = 2 * jnp.pi * positions / (jnp.asarray(pixel_size) * n) + offsets
    options = {} if options is None else dict(options)
    if ndim == 2:
        fourier_image = _nufft2d1(
            shape, strengths, angles, eps=eps, upsampfac=upsampfac, options=options
        )
    else:
        fourier_image = _nufft3d1(
            shape, strengths, angles, eps=eps, upsampfac=upsampfac, options=options
        )
    return fourier_image if fftshifted else jnp.fft.ifftshift(fourier_image)


def _nufft2d1(
    shape: tuple[int, ...],
    strengths: Complex[Array, " M"],
    angles: Float[Array, "M 2"],
    *,
    eps: float,
    upsampfac: float,
    options: dict[str, Any],
) -> Complex[Array, "y x"]:
    if CRYOJAX_FINUFFT_BACKEND == "jax-finufft":
        _check_jax_finufft()
        opts = options.pop("opts", None) or _make_jax_finufft_opts(upsampfac)
        return jax_finufft.nufft1(  # type: ignore
            shape,
            strengths,
            angles[:, 1],
            angles[:, 0],
            eps=eps,
            iflag=-1,
            opts=opts,
            **options,
        )
    upsampfac = options.pop("upsampfac", upsampfac)
    return nufftax.nufft2d1(
        n_modes=shape[::-1],  # type: ignore
        c=strengths,
        x=angles[:, 0],
        y=angles[:, 1],
        eps=eps,
        isign=-1,
        upsampfac=upsampfac,
        **options,
    )


def _nufft3d1(
    shape: tuple[int, ...],
    strengths: Complex[Array, " M"],
    angles: Float[Array, "M 3"],
    *,
    eps: float,
    upsampfac: float,
    options: dict[str, Any],
) -> Complex[Array, "z y x"]:
    if CRYOJAX_FINUFFT_BACKEND == "jax-finufft":
        _check_jax_finufft()
        opts = options.pop("opts", None) or _make_jax_finufft_opts(upsampfac)
        return jax_finufft.nufft1(  # type: ignore
            shape,
            strengths,
            angles[:, 2],
            angles[:, 1],
            angles[:, 0],
            eps=eps,
            iflag=-1,
            opts=opts,
            **options,
        )
    upsampfac = options.pop("upsampfac", upsampfac)
    return nufftax.nufft3d1(
        n_modes=shape[::-1],  # type: ignore
        c=strengths,
        x=angles[:, 0],
        y=angles[:, 1],
        z=angles[:, 2],
        eps=eps,
        isign=-1,
        upsampfac=upsampfac,
        **options,
    )


def _check_jax_finufft():
    if jax_finufft is None:
        raise RuntimeError(
            "Tried to use the `jax-finufft` non-uniform FFT backend "
            "(set via the `CRYOJAX_FINUFFT_BACKEND` environment "
            "variable), but `jax-finufft` is not installed. "
            "See https://github.com/flatironinstitute/jax-finufft "
            "for installation instructions."
        ) from JAX_FINUFFT_IMPORT_ERROR


def _make_jax_finufft_opts(upsampfac: float):
    assert NestedOpts is not None
    assert Opts is not None
    return NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )
