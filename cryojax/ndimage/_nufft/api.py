"""Non-uniform FFTs of point sources, in cryoJAX conventions."""

import math
from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from ..._config import CRYOJAX_FINUFFT_BACKEND
from ...jax_util import FloatLike
from .._pallas_helpers import resolve_enable_pallas
from .kernel import (
    PSWFKernel,
    fine_grid_size,
    kernel_fourier_series,
    kernel_width,
    make_pswf_kernel,
)
from .spread import interp, interp_bwd, spread, spread_bwd


try:
    import jax_finufft
    from jax_finufft.options import Opts

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jax_finufft, Opts = None, None
    JAX_FINUFFT_IMPORT_ERROR = err


def dispatch_nufft1(
    positions: Float[Array, "M d"],
    strengths: Float[Array, " M"] | Float[Array, "R M"],
    shape: tuple[int, ...],
    *,
    pixel_size: FloatLike = 1.0,
    eps: float = 1e-6,
    upsample_factor: float = 1.25,
    enable_pallas: bool | Mapping[str, bool] | None = None,
    options: dict[str, Any] | None = None,
) -> Complex[Array, "..."]:
    """Compute the `rfftn` of point sources with a type-1 non-uniform FFT,

    $$F[k] = \\sum_j c_j \\exp(-2 \\pi i \\sum_a k_a t_{j,a} / n_a),$$

    where $t_a = n_a // 2 + r_a / \\textrm{pixel_size}$ is a position in grid indices.
    The modes are those of `jax.numpy.fft.rfftn`: `fftfreq` order on the full axes and
    `0 .. n_x // 2` on the last, so a point on the grid gives the `rfftn` of a unit delta
    at that pixel. The backend is set by the `CRYOJAX_FINUFFT_BACKEND` environment
    variable: `"native"` (default) or `"jax-finufft"`. Differentiable in reverse mode,
    including forward-over-reverse; `jax.jvp` of this function is not supported.

    **Arguments:**

    - `positions`:
        The positions of the points, as `(x, y)` or `(x, y, z)`, in the units of
        `pixel_size`, measured from grid index `n // 2`.
    - `strengths`:
        The real strength of each point, of shape `(M,)`, or `(R, M)` for `R` sets of
        strengths sharing the positions.
    - `shape`:
        The shape of the real-space grid, as `(y, x)` or `(z, y, x)`.
    - `pixel_size`:
        The grid spacing.
    - `eps`:
        The requested relative accuracy.
    - `upsample_factor`:
        The ratio of the fine grid to `shape` (finufft's `upsampfac`), greater than 1.
        The kernel widens as it approaches 1, up to 16 fine-grid pixels.
    - `enable_pallas`:
        Whether the native backend spreads with Pallas kernels, for the forward and
        backward pass, as in [`cryojax.ndimage.spread_gaussians_2d`][].
    - `options`:
        Options for the `"jax-finufft"` backend only. See
        [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html).

    **Returns:**

    The rfft half grid, of shape `(*shape[:-1], shape[-1] // 2 + 1)`, with a leading `R`
    axis for batched `strengths`.
    """
    name = "dispatch_nufft1(positions, strengths, shape)"
    ndim = _check_positions(name, positions, shape)
    strengths = jnp.asarray(strengths)
    if jnp.iscomplexobj(strengths) or strengths.ndim not in (1, 2):
        raise ValueError(
            f"Found invalid value for `{name}`. The `strengths` must be real, with shape "
            f"`(M,)` or `(R, M)`, but got `strengths.dtype = {strengths.dtype}` and "
            f"`strengths.shape = {strengths.shape}`."
        )
    if jnp.shape(positions)[0] != strengths.shape[-1]:
        raise ValueError(
            f"Found invalid value for `{name}`. The `strengths` must have one entry per "
            f"point, but got `positions.shape = {jnp.shape(positions)}` and "
            f"`strengths.shape = {strengths.shape}`."
        )
    _check_kernel(
        "dispatch_nufft1(..., eps, upsample_factor)", eps, upsample_factor, ndim
    )
    dtype = jnp.result_type(positions, strengths, pixel_size, 0.0)
    positions, c = jnp.asarray(positions, dtype), strengths.astype(dtype)
    c = c if c.ndim == 2 else c[None]
    if CRYOJAX_FINUFFT_BACKEND == "jax-finufft":
        spectrum = _jax_finufft_nufft1(
            positions, c, shape, pixel_size, eps, upsample_factor, options
        )
    else:
        kernel = make_pswf_kernel(eps, upsample_factor, ndim)
        fine_shape = tuple(
            fine_grid_size(n, upsample_factor, kernel.width) for n in shape
        )
        u = _fine_coordinates(positions, shape, fine_shape, pixel_size)
        use_pallas_fwd, use_pallas_bwd = resolve_enable_pallas(enable_pallas)
        grid = _spread_dispatch(u, c, fine_shape, kernel, use_pallas_fwd, use_pallas_bwd)
        fine_spectrum = jnp.fft.rfftn(grid, axes=tuple(range(1, ndim + 1)))
        spectrum = _deconvolve(_crop(fine_spectrum, shape), kernel, shape, fine_shape)
    return spectrum if strengths.ndim == 2 else spectrum[0]


def dispatch_nufft2(
    positions: Float[Array, "M d"],
    rfftn_array: Complex[Array, "..."],
    shape: tuple[int, ...],
    *,
    pixel_size: FloatLike = 1.0,
    eps: float = 1e-6,
    upsample_factor: float = 1.25,
    enable_pallas: bool | Mapping[str, bool] | None = None,
    options: dict[str, Any] | None = None,
) -> Float[Array, " M"] | Float[Array, "R M"]:
    """Evaluate an rfft half grid at points with a type-2 non-uniform FFT,

    $$f_j = \\frac{1}{\\prod_a n_a} \\textrm{Re} \\sum_k w_k F[k]
    \\exp(+2 \\pi i \\sum_a k_a t_{j,a} / n_a),$$

    on the modes of [`cryojax.ndimage.dispatch_nufft1`][], with $w_k = 1$ on the
    self-conjugate last-axis columns (0, and $n_x/2$ when even) and $2$ elsewhere. At
    grid points this is `jax.numpy.fft.irfftn(rfftn_array, s=shape)`; it is
    $1 / \\prod_a n_a$ times the $w$-weighted adjoint of `dispatch_nufft1`.

    **Arguments:**

    - `positions`:
        The positions of the points, as for `dispatch_nufft1`.
    - `rfftn_array`:
        The rfft half grid, of shape `(*shape[:-1], shape[-1] // 2 + 1)`, or with a
        leading `R` axis for `R` grids sharing the positions.
    - `shape`:
        The shape of the real-space grid, as `(y, x)` or `(z, y, x)`.
    - `pixel_size`, `eps`, `upsample_factor`, `enable_pallas`, `options`:
        As for `dispatch_nufft1`.

    **Returns:**

    The real values at the points, of shape `(M,)`, or `(R, M)` for a batched
    `rfftn_array`.
    """
    name = "dispatch_nufft2(positions, rfftn_array, shape)"
    ndim = _check_positions(name, positions, shape)
    rfftn_array = jnp.asarray(rfftn_array)
    half_shape = (*shape[:-1], shape[-1] // 2 + 1)
    if rfftn_array.shape[-ndim:] != half_shape or rfftn_array.ndim not in (
        ndim,
        ndim + 1,
    ):
        raise ValueError(
            f"Found invalid value for `{name}`. The `rfftn_array` must have shape "
            f"`{half_shape}`, with an optional leading batch axis, but got "
            f"`rfftn_array.shape = {rfftn_array.shape}`."
        )
    _check_kernel(
        "dispatch_nufft2(..., eps, upsample_factor)", eps, upsample_factor, ndim
    )
    real_dtype = jnp.finfo(jnp.result_type(rfftn_array, 0j)).dtype
    dtype = jnp.result_type(positions, real_dtype, pixel_size, 0.0)
    positions = jnp.asarray(positions, dtype)
    spectrum = rfftn_array.astype(jnp.result_type(dtype, 1j))
    spectrum = spectrum if rfftn_array.ndim == ndim + 1 else spectrum[None]
    if CRYOJAX_FINUFFT_BACKEND == "jax-finufft":
        values = _jax_finufft_nufft2(
            positions, spectrum, shape, pixel_size, eps, upsample_factor, options
        )
    else:
        kernel = make_pswf_kernel(eps, upsample_factor, ndim)
        fine_shape = tuple(
            fine_grid_size(n, upsample_factor, kernel.width) for n in shape
        )
        u = _fine_coordinates(positions, shape, fine_shape, pixel_size)
        # Halving an even last axis's Nyquist column gives it w = 1: the fine grid's
        # `irfftn` doubles every column it holds as interior
        spectrum = _deconvolve(spectrum, kernel, shape, fine_shape, adjoint=True)
        grid = jnp.fft.irfftn(
            _pad(spectrum, shape, fine_shape),
            s=fine_shape,
            axes=tuple(range(1, ndim + 1)),
        )
        use_pallas_fwd, use_pallas_bwd = resolve_enable_pallas(enable_pallas)
        values = _interp_dispatch(u, grid, kernel, use_pallas_fwd, use_pallas_bwd)
    return values if rfftn_array.ndim == ndim + 1 else values[0]


#
# Checks
#
def _check_positions(name, positions, shape) -> int:
    ndim = len(shape)
    if ndim not in (2, 3) or jnp.ndim(positions) != 2 or jnp.shape(positions)[-1] != ndim:
        raise ValueError(
            f"Found invalid value for `{name}`. The `shape` must be 2D or 3D, and "
            "`positions` must have shape `(M, d)`, one coordinate per dimension, but got "
            f"`shape = {shape}` and `positions.shape = {jnp.shape(positions)}`."
        )
    return ndim


def _check_kernel(name, eps, upsample_factor, ndim):
    if not upsample_factor > 1:
        raise ValueError(
            f"Found invalid value for `{name}`. The `upsample_factor` must be greater "
            f"than 1, but got `upsample_factor = {upsample_factor}`."
        )
    if not eps > 0:
        raise ValueError(
            f"Found invalid value for `{name}`. The `eps` must be positive, but got "
            f"`eps = {eps}`."
        )
    try:
        kernel_width(eps, upsample_factor, ndim)
    except ValueError as err:
        raise ValueError(
            f"Found invalid value for `{name}`. The `eps = {eps}` needs a kernel wider "
            f"than 16 fine-grid pixels at `upsample_factor = {upsample_factor}`; "
            "increase `eps` or `upsample_factor`."
        ) from err


#
# Native backend
#
def _fine_coordinates(positions, shape, fine_shape, pixel_size):
    """`u_a = (m_a / n_a) t_a`, outside the `custom_vjp`s so that autodiff gives the
    `positions` and `pixel_size` derivatives."""
    n, m = np.array(shape[::-1]), np.array(fine_shape[::-1])
    dtype = positions.dtype
    return jnp.asarray(m / n, dtype) * (
        jnp.asarray(n // 2, dtype) + positions / jnp.asarray(pixel_size, dtype)
    )


def _crop(fine_spectrum, shape):
    """Keep `shape`'s rfft half grid: rows `0 .. (n+1)//2 - 1` and the last `n//2` on
    each full axis, columns `0 .. n_x//2`."""
    for axis, n in enumerate(shape[:-1], start=1):
        m = fine_spectrum.shape[axis]
        fine_spectrum = jnp.concatenate(
            [
                jax.lax.slice_in_dim(fine_spectrum, 0, (n + 1) // 2, axis=axis),
                jax.lax.slice_in_dim(fine_spectrum, m - n // 2, m, axis=axis),
            ],
            axis=axis,
        )
    return fine_spectrum[..., : shape[-1] // 2 + 1]


def _pad(spectrum, shape, fine_shape):
    """Zero-pad `shape`'s rfft half grid into `fine_shape`'s; the inverse of `_crop`."""
    spectrum = jnp.pad(
        spectrum,
        [(0, 0)] * (spectrum.ndim - 1) + [(0, fine_shape[-1] // 2 - shape[-1] // 2)],
    )
    for axis, (n, m) in enumerate(zip(shape[:-1], fine_shape[:-1]), start=1):
        head = jax.lax.slice_in_dim(spectrum, 0, (n + 1) // 2, axis=axis)
        tail = jax.lax.slice_in_dim(spectrum, (n + 1) // 2, n, axis=axis)
        zeros_shape = list(spectrum.shape)
        zeros_shape[axis] = m - n
        zeros = jnp.zeros(zeros_shape, spectrum.dtype)
        spectrum = jnp.concatenate([head, zeros, tail], axis=axis)
    return spectrum


def _deconvolve(spectrum, kernel, shape, fine_shape, *, adjoint=False):
    """Divide by the kernel's Fourier series, one axis at a time. For type 2
    (`adjoint`), also scale by `prod(m) / prod(n)` and halve an even Nyquist column."""
    real_dtype = jnp.finfo(spectrum.dtype).dtype
    ndim = len(shape)
    for axis, (n, m) in enumerate(zip(shape, fine_shape)):
        is_last = axis == ndim - 1
        factor = 1 / kernel_fourier_series(kernel, n, m, rfft=is_last)
        if adjoint:
            factor = factor * (m / n)
            if is_last and n % 2 == 0:
                factor[-1] /= 2
        factor = jnp.asarray(factor, real_dtype).reshape((-1,) + (1,) * (ndim - 1 - axis))
        spectrum = spectrum * factor
    return spectrum


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _spread_dispatch(u, c, fine_shape, kernel, use_pallas_fwd, use_pallas_bwd):
    if use_pallas_fwd:
        return _pallas_spread(u, c, fine_shape, kernel)
    return spread(u, c, fine_shape, kernel)


def _spread_dispatch_fwd(u, c, fine_shape, kernel, use_pallas_fwd, use_pallas_bwd):
    if use_pallas_fwd:
        grid = _pallas_spread(u, c, fine_shape, kernel)
    else:
        grid = spread(u, c, fine_shape, kernel)
    return grid, (u, c)


def _spread_dispatch_bwd(fine_shape, kernel, use_pallas_fwd, use_pallas_bwd, res, g_bar):
    del use_pallas_fwd
    u, c = res
    if use_pallas_bwd:
        c_bar, u_bar = _pallas_interp(
            u, g_bar, c, fine_shape, kernel, values=True, positions=True
        )
        return u_bar, c_bar
    return spread_bwd(u, c, g_bar, kernel)


_spread_dispatch.defvjp(_spread_dispatch_fwd, _spread_dispatch_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _interp_dispatch(u, g, kernel, use_pallas_fwd, use_pallas_bwd):
    if use_pallas_fwd:
        values, _ = _pallas_interp(
            u, g, None, g.shape[1:], kernel, values=True, positions=False
        )
        return values
    return interp(u, g, kernel)


def _interp_dispatch_fwd(u, g, kernel, use_pallas_fwd, use_pallas_bwd):
    if use_pallas_fwd:
        values, _ = _pallas_interp(
            u, g, None, g.shape[1:], kernel, values=True, positions=False
        )
    else:
        values = interp(u, g, kernel)
    return values, (u, g)


def _interp_dispatch_bwd(kernel, use_pallas_fwd, use_pallas_bwd, res, f_bar):
    del use_pallas_fwd
    u, g = res
    if use_pallas_bwd:
        g_bar = _pallas_spread(u, f_bar, g.shape[1:], kernel)
        _, u_bar = _pallas_interp(
            u, g, f_bar, g.shape[1:], kernel, values=False, positions=True
        )
        return u_bar, g_bar
    return interp_bwd(u, g, f_bar, kernel)


_interp_dispatch.defvjp(_interp_dispatch_fwd, _interp_dispatch_bwd)


def _pallas_spread(
    u: Float[Array, "M d"],
    c: Float[Array, "R M"],
    fine_shape: tuple[int, ...],
    kernel: PSWFKernel,
) -> Float[Array, "R ..."]:
    raise NotImplementedError(
        "The Pallas backend of the native NUFFT is not implemented yet. Pass "
        "`enable_pallas=False`."
    )


def _pallas_interp(
    u: Float[Array, "M d"],
    g: Float[Array, "R ..."],
    weights: Float[Array, "R M"] | None,
    fine_shape: tuple[int, ...],
    kernel: PSWFKernel,
    *,
    values: bool,
    positions: bool,
) -> tuple[Float[Array, "R M"] | None, Float[Array, "M d"] | None]:
    """Values `(R, M)` and/or `sum_r weights_r grad_u interp(u, g_r)`, `(M, d)`."""
    raise NotImplementedError(
        "The Pallas backend of the native NUFFT is not implemented yet. Pass "
        "`enable_pallas=False`."
    )


#
# jax-finufft backend
#
def _jax_finufft_nufft1(positions, c, shape, pixel_size, eps, upsample_factor, options):
    _check_jax_finufft()
    if positions.shape[0] == 0:
        half_shape = (c.shape[0], *shape[:-1], shape[-1] // 2 + 1)
        return jnp.zeros(half_shape, jnp.result_type(c, 1j))
    options = {} if options is None else dict(options)
    opts = options.pop("opts", None) or _make_jax_finufft_opts(upsample_factor)
    # An odd last axis holds 0 .. n_x // 2 first: `convert_fftn_to_rfftn` would give
    # `+n/2` where an even full axis meets an even last axis's Nyquist column
    modes = (*shape[:-1], shape[-1] + 1 - shape[-1] % 2)
    spectrum = jax_finufft.nufft1(  # type: ignore
        modes,
        c.astype(jnp.result_type(c, 1j)),
        *_angles(positions, shape, pixel_size),
        iflag=-1,
        eps=eps,
        opts=opts,
        **options,
    )
    if not _is_fft_ordered(opts, finufft_type=1):
        spectrum = jnp.fft.ifftshift(spectrum, axes=tuple(range(1, len(shape) + 1)))
    return spectrum[..., : shape[-1] // 2 + 1]


def _jax_finufft_nufft2(
    positions, spectrum, shape, pixel_size, eps, upsample_factor, options
):
    _check_jax_finufft()
    real_dtype = jnp.finfo(spectrum.dtype).dtype
    if positions.shape[0] == 0:
        return jnp.zeros((spectrum.shape[0], 0), real_dtype)
    options = {} if options is None else dict(options)
    opts = options.pop("opts", None) or _make_jax_finufft_opts(upsample_factor)
    grid = _hermitian_grid(spectrum, shape)
    if not _is_fft_ordered(opts, finufft_type=2):
        grid = jnp.fft.fftshift(grid, axes=tuple(range(1, len(shape) + 1)))
    values = jax_finufft.nufft2(  # type: ignore
        grid,
        *_angles(positions, shape, pixel_size),
        iflag=1,
        eps=eps,
        opts=opts,
        **options,
    )
    return (values.real / math.prod(shape)).astype(real_dtype)


def _angles(positions, shape, pixel_size):
    """finufft's angles `2 pi t_a / n_a` in `[-pi, pi)`, one array per axis of `shape`."""
    n = np.array(shape[::-1])
    dtype = positions.dtype
    t = jnp.asarray(n // 2, dtype) + positions / jnp.asarray(pixel_size, dtype)
    angles = jnp.mod(jnp.asarray(2 * np.pi / n, dtype) * t + np.pi, 2 * np.pi) - np.pi
    return [angles[:, a] for a in reversed(range(len(shape)))]


def _hermitian_grid(spectrum, shape):
    """`G[k] = w_k F[k] / 2`, `G[-k] = conj(G[k])` in FFT order, padded to `n + 1` on
    even axes so that both `+-n/2` exist."""
    n_x = shape[-1]
    column_weights = np.ones(n_x // 2 + 1)
    column_weights[0] = 0.5
    if n_x % 2 == 0:
        column_weights[-1] = 0.5
    grid = spectrum * jnp.asarray(column_weights, jnp.finfo(spectrum.dtype).dtype)
    # Columns 0 .. n_x//2, then the negative ones, n_x//2 of them for odd and even n_x
    grid = jnp.pad(grid, [(0, 0)] * len(shape) + [(0, n_x // 2)])
    for axis, n in enumerate(shape[:-1], start=1):
        if n % 2 == 0:
            grid = jnp.insert(grid, n // 2, 0, axis=axis)
    axes = tuple(range(1, len(shape) + 1))
    negated = jnp.roll(jnp.flip(grid, axes), 1, axes)
    return grid + jnp.conj(negated)


def _is_fft_ordered(opts, finufft_type):
    """Whether jax-finufft's forward options, `Opts` or `NestedOpts`, set `modeord`."""
    forward = (
        getattr(opts, "forward", None)
        or getattr(opts, f"type{finufft_type}", None)
        or opts
    )
    return bool(getattr(forward, "modeord", False))


def _make_jax_finufft_opts(upsample_factor):
    assert Opts is not None
    return Opts(modeord=True, upsampfac=upsample_factor, gpu_upsampfac=upsample_factor)


def _check_jax_finufft():
    if jax_finufft is None:
        raise RuntimeError(
            "Tried to use the `jax-finufft` non-uniform FFT backend "
            "(set via the `CRYOJAX_FINUFFT_BACKEND` environment "
            "variable), but `jax-finufft` is not installed. "
            "See https://github.com/flatironinstitute/jax-finufft "
            "for installation instructions."
        ) from JAX_FINUFFT_IMPORT_ERROR
