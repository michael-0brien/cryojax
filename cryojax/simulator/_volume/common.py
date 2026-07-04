"""Shared helpers for volume rendering backends.

This includes a pure-JAX Gaussian spreading backend (scatter atom/gaussian
strengths onto a uniform grid), adapted from `nufftax`'s NUFFT type-1
spreading, but specialized to the isotropic Gaussian (and pixel-averaged
Gaussian) kernels used by `IndependentAtomVolume` and `GaussianMixtureVolume`.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

from ...ndimage import make_1d_frequency_grid


def make_frequencies_1d(
    shape_u: tuple[int, ...],
    pixel_size_u: Float[Array, ""],
    modeord: int = 0,
):
    return tuple(
        make_1d_frequency_grid(
            s,
            pixel_size_u,
            outputs_rfftfreqs=False,
            fftshifted=(True if modeord == 0 else False),
        )
        for s in shape_u[::-1]
    )


# ============================================================================
# Gaussian spreading backend
# ============================================================================
#
# Scatters point strengths `c` onto a uniform grid, weighted by a compactly
# supported (`nspread` grid points wide) isotropic Gaussian centered at each
# point, or by the exact average of that Gaussian over a pixel/voxel (the
# "erf" kernel, used to sample the average value within a pixel rather than
# its value at a point).
#
# The custom VJP rule mirrors `nufftax`'s spreading (NUFFT type 1): the
# gradient w.r.t. `c` is the *adjoint* operation, i.e. interpolation (NUFFT
# type 2) at the same positions/kernel (`_interp_*d_impl`), and the gradient
# w.r.t. positions uses the analytic kernel derivative (`_spread_*d_grad_*`),
# rather than differentiating through `jnp.ceil`/`segment_sum` directly.
#
# `c` carries the *entire* per-point scattering weight (e.g. an atom's
# occupancy times its Gaussian amplitude): it multiplies the (normalized)
# kernel exactly once, regardless of dimensionality. `variance` only shapes
# the kernel and is not otherwise special-cased.
#
# Positions are expected in *grid-index units*, i.e. `x = 0` corresponds to
# grid index `shape[-1] // 2` (the RELION real-space center convention used
# throughout cryojax; see `normalize_positions_to_grid`). This differs from
# `nufftax`, which works on the NUFFT domain `[-pi, pi)`; here there is no
# NUFFT involved, so that intermediate representation is unnecessary.


def eps_to_nspread(eps: float) -> int:
    """FINUFFT-style heuristic mapping a precision `eps` to a kernel width."""
    log_tol = -math.log10(max(eps, 1e-16))
    return max(2, int(math.ceil(log_tol + 1)))


def normalize_positions_to_grid(
    positions: Float[Array, "... ndim"],
    shape: tuple[int, ...],
    pixel_size: Float[Array, ""],
) -> Float[Array, "... ndim"]:
    """Convert physical-space positions to grid-index units.

    The real-space center is always at integer index `shape[i] // 2` (RELION
    convention), for both even and odd `shape[i]`.
    """
    ndim = positions.shape[-1]
    # `shape` is ordered `(y, x)` or `(z, y, x)`; reverse so the first entry
    # corresponds to the x-coordinate.
    shape_spatial = shape[::-1][:ndim]
    centers = jnp.asarray([s // 2 for s in shape_spatial], dtype=float)
    return positions / pixel_size + centers


def _gaussian_shape_and_grad(r: Array, variance: Array) -> tuple[Array, Array]:
    """Normalized isotropic Gaussian marginal at physical offset `r`, and its
    derivative w.r.t. `r`."""
    shape = jnp.exp(-0.5 * r**2 / variance) / jnp.sqrt(2 * jnp.pi * variance)
    dshape_dr = -(r / variance) * shape
    return shape, dshape_dr


def _erf_shape_and_grad(
    r: Array, variance: Array, width: Float[Array, ""]
) -> tuple[Array, Array]:
    """Average of the Gaussian marginal over a pixel/voxel of `width` centered
    at physical offset `r`, and its derivative w.r.t. `r`."""
    scaling = 1.0 / jnp.sqrt(2 * variance)
    left, right = scaling * (r - width / 2), scaling * (r + width / 2)
    shape = (jsp.special.erf(right) - jsp.special.erf(left)) / (2 * width)
    two_over_sqrt_pi = 2.0 / jnp.sqrt(jnp.pi)
    dshape_dr = (
        scaling * two_over_sqrt_pi * (jnp.exp(-(right**2)) - jnp.exp(-(left**2)))
    ) / (2 * width)
    return shape, dshape_dr


def _kernel_shape_and_grad(
    z: Array, variance: Array, width: Float[Array, ""], *, use_erf: bool
) -> tuple[Array, Array]:
    """Evaluate the kernel and its derivative w.r.t. the grid-index offset
    `z`, converting to the physical offset `r = z * width` first (`width` is
    the pixel/voxel size).
    """
    r = z * width
    shape, dshape_dr = (
        _erf_shape_and_grad(r, variance, width)
        if use_erf
        else _gaussian_shape_and_grad(r, variance)
    )
    return shape, dshape_dr * width


def _kernel_shape(z: Array, variance: Array, width: Float[Array, ""], *, use_erf: bool):
    shape, _ = _kernel_shape_and_grad(z, variance, width, use_erf=use_erf)
    return shape


def _broadcast_per_point(value: Array) -> Array:
    """Add a trailing axis to a per-point array of shape `(M,)` so that it
    broadcasts against a `(M, nspread)` array of per-point kernel offsets.
    Scalars (shape `()`) are left unchanged.
    """
    return value[:, None] if jnp.ndim(value) == 1 else value


def _support_indices_and_offsets(
    coord: Float[Array, " M"], n: int, nspread: int
) -> tuple[Array, Array]:
    """Per-axis spread indices (wrapped to `[0, n)`) and kernel offsets `z`."""
    i0 = jnp.ceil(coord - nspread / 2.0).astype(jnp.int32)
    offsets = jnp.arange(nspread)
    indices = i0[:, None] + offsets[None, :]  # (M, nspread)
    z = indices.astype(coord.dtype) - coord[:, None]
    return indices % n, z


def _axis_weights_and_grad(coord, n, nspread, variance, width, *, use_erf):
    """Per-axis spread indices, kernel weights, and their `d/d(coord)`."""
    indices, z = _support_indices_and_offsets(coord, n, nspread)
    variance = _broadcast_per_point(variance)
    shape, dshape_dz = _kernel_shape_and_grad(z, variance, width, use_erf=use_erf)
    # z = index - coord, so d(shape)/d(coord) = -dshape_dz.
    return indices, shape, -dshape_dz


# ── 2-D ──────────────────────────────────────────────────────────────────────


def _kernel_weights_2d(z_x, z_y, variance, width, *, use_erf):
    variance = _broadcast_per_point(variance)
    weights_x = _kernel_shape(z_x, variance, width, use_erf=use_erf)
    weights_y = _kernel_shape(z_y, variance, width, use_erf=use_erf)
    return weights_y[:, :, None] * weights_x[:, None, :]


def _spread_2d_impl(x, y, c, ny, nx, *, variance, pixel_size, nspread, use_erf):
    indices_x, z_x = _support_indices_and_offsets(x, nx, nspread)
    indices_y, z_y = _support_indices_and_offsets(y, ny, nspread)
    weights_2d = _kernel_weights_2d(z_x, z_y, variance, pixel_size, use_erf=use_erf)
    weighted_c = c[:, None, None] * weights_2d
    indices_2d = indices_y[:, :, None] * nx + indices_x[:, None, :]
    fw = jax.ops.segment_sum(weighted_c.ravel(), indices_2d.ravel(), num_segments=ny * nx)
    return fw.reshape(ny, nx)


def _interp_2d_impl(x, y, fw, ny, nx, *, variance, pixel_size, nspread, use_erf):
    indices_x, z_x = _support_indices_and_offsets(x, nx, nspread)
    indices_y, z_y = _support_indices_and_offsets(y, ny, nspread)
    weights_2d = _kernel_weights_2d(z_x, z_y, variance, pixel_size, use_erf=use_erf)
    indices_2d = indices_y[:, :, None] * nx + indices_x[:, None, :]
    fw_gathered = fw.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)
    return jnp.sum(fw_gathered * weights_2d, axis=(-2, -1))


def _spread_2d_grad_xy(x, y, c, g, ny, nx, *, variance, pixel_size, nspread, use_erf):
    """Analytic gradient of `_spread_2d_impl` w.r.t. `x` and `y`."""
    indices_x, weights_x, dweights_x = _axis_weights_and_grad(
        x, nx, nspread, variance, pixel_size, use_erf=use_erf
    )
    indices_y, weights_y, dweights_y = _axis_weights_and_grad(
        y, ny, nspread, variance, pixel_size, use_erf=use_erf
    )
    indices_2d = indices_y[:, :, None] * nx + indices_x[:, None, :]
    g_gathered = g.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)

    weights_2d_dx = weights_y[:, :, None] * dweights_x[:, None, :]
    dx = c * jnp.sum(g_gathered * weights_2d_dx, axis=(-2, -1))

    weights_2d_dy = dweights_y[:, :, None] * weights_x[:, None, :]
    dy = c * jnp.sum(g_gathered * weights_2d_dy, axis=(-2, -1))

    return dx, dy


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _spread_2d(x, y, c, variance, pixel_size, ny, nx, nspread, use_erf):
    return _spread_2d_impl(
        x,
        y,
        c,
        ny,
        nx,
        variance=variance,
        pixel_size=pixel_size,
        nspread=nspread,
        use_erf=use_erf,
    )


def _spread_2d_fwd(x, y, c, variance, pixel_size, ny, nx, nspread, use_erf):
    out = _spread_2d_impl(
        x,
        y,
        c,
        ny,
        nx,
        variance=variance,
        pixel_size=pixel_size,
        nspread=nspread,
        use_erf=use_erf,
    )
    return out, (x, y, variance, pixel_size, c)


def _spread_2d_bwd(ny, nx, nspread, use_erf, res, g):
    x, y, variance, pixel_size, c = res
    kwargs = dict(
        variance=variance, pixel_size=pixel_size, nspread=nspread, use_erf=use_erf
    )

    # The adjoint of spreading is interpolation, at the same positions/kernel.
    dc = _interp_2d_impl(x, y, g, ny, nx, **kwargs)
    # Position gradients use the analytic kernel derivative.
    dx, dy = _spread_2d_grad_xy(x, y, c, g, ny, nx, **kwargs)

    # `variance`/`pixel_size` also shape the kernel; differentiate through the
    # (cheap) weight evaluation directly for these.
    indices_x, z_x = _support_indices_and_offsets(x, nx, nspread)
    indices_y, z_y = _support_indices_and_offsets(y, ny, nspread)
    indices_2d = indices_y[:, :, None] * nx + indices_x[:, None, :]
    g_gathered = g.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)

    def weighted(variance, pixel_size):
        weights_2d = _kernel_weights_2d(z_x, z_y, variance, pixel_size, use_erf=use_erf)
        return c[:, None, None] * weights_2d

    _, vjp_fn = jax.vjp(weighted, variance, pixel_size)
    dvariance, dpixel_size = vjp_fn(g_gathered)

    return dx, dy, dc, dvariance, dpixel_size


_spread_2d.defvjp(_spread_2d_fwd, _spread_2d_bwd)


def spread_2d(
    x: Float[Array, " M"],
    y: Float[Array, " M"],
    c: Float[Array, " M"],
    shape: tuple[int, int],
    *,
    variance: Float[Array, ""] | Float[Array, " M"],
    pixel_size: Float[Array, ""],
    nspread: int,
    use_erf: bool,
) -> Float[Array, "{shape[0]} {shape[1]}"]:
    """Spread point strengths `c` onto a 2D grid of the given `shape = (ny, nx)`
    with an isotropic Gaussian (or pixel-averaged Gaussian) kernel of the
    given `variance`.

    `x`, `y` are in grid-index units (see `normalize_positions_to_grid`). `c`
    carries the entire per-point scattering weight (e.g. amplitude times
    atom occupancy); `variance` may be a scalar or a per-point array of shape
    `(M,)`.
    """
    ny, nx = shape
    return _spread_2d(x, y, c, variance, pixel_size, ny, nx, nspread, use_erf)


# ── 3-D ──────────────────────────────────────────────────────────────────────


def _kernel_weights_3d(z_x, z_y, z_z, variance, width, *, use_erf):
    variance = _broadcast_per_point(variance)
    weights_x = _kernel_shape(z_x, variance, width, use_erf=use_erf)
    weights_y = _kernel_shape(z_y, variance, width, use_erf=use_erf)
    weights_z = _kernel_shape(z_z, variance, width, use_erf=use_erf)
    return (
        weights_z[:, :, None, None]
        * weights_y[:, None, :, None]
        * weights_x[:, None, None, :]
    )


def _spread_3d_impl(x, y, z, c, nz, ny, nx, *, variance, voxel_size, nspread, use_erf):
    indices_x, z_x = _support_indices_and_offsets(x, nx, nspread)
    indices_y, z_y = _support_indices_and_offsets(y, ny, nspread)
    indices_z, z_z = _support_indices_and_offsets(z, nz, nspread)
    weights_3d = _kernel_weights_3d(z_x, z_y, z_z, variance, voxel_size, use_erf=use_erf)
    weighted_c = c[:, None, None, None] * weights_3d
    indices_3d = (
        indices_z[:, :, None, None] * (nx * ny)
        + indices_y[:, None, :, None] * nx
        + indices_x[:, None, None, :]
    )
    fw = jax.ops.segment_sum(
        weighted_c.ravel(), indices_3d.ravel(), num_segments=nz * ny * nx
    )
    return fw.reshape(nz, ny, nx)


def _interp_3d_impl(x, y, z, fw, nz, ny, nx, *, variance, voxel_size, nspread, use_erf):
    indices_x, z_x = _support_indices_and_offsets(x, nx, nspread)
    indices_y, z_y = _support_indices_and_offsets(y, ny, nspread)
    indices_z, z_z = _support_indices_and_offsets(z, nz, nspread)
    weights_3d = _kernel_weights_3d(z_x, z_y, z_z, variance, voxel_size, use_erf=use_erf)
    indices_3d = (
        indices_z[:, :, None, None] * (nx * ny)
        + indices_y[:, None, :, None] * nx
        + indices_x[:, None, None, :]
    )
    fw_gathered = fw.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)
    return jnp.sum(fw_gathered * weights_3d, axis=(-3, -2, -1))


def _spread_3d_grad_xyz(
    x, y, z, c, g, nz, ny, nx, *, variance, voxel_size, nspread, use_erf
):
    """Analytic gradient of `_spread_3d_impl` w.r.t. `x`, `y`, and `z`."""
    indices_x, weights_x, dweights_x = _axis_weights_and_grad(
        x, nx, nspread, variance, voxel_size, use_erf=use_erf
    )
    indices_y, weights_y, dweights_y = _axis_weights_and_grad(
        y, ny, nspread, variance, voxel_size, use_erf=use_erf
    )
    indices_z, weights_z, dweights_z = _axis_weights_and_grad(
        z, nz, nspread, variance, voxel_size, use_erf=use_erf
    )
    indices_3d = (
        indices_z[:, :, None, None] * (nx * ny)
        + indices_y[:, None, :, None] * nx
        + indices_x[:, None, None, :]
    )
    g_gathered = g.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)

    weights_3d_dx = (
        weights_z[:, :, None, None]
        * weights_y[:, None, :, None]
        * dweights_x[:, None, None, :]
    )
    dx = c * jnp.sum(g_gathered * weights_3d_dx, axis=(-3, -2, -1))

    weights_3d_dy = (
        weights_z[:, :, None, None]
        * dweights_y[:, None, :, None]
        * weights_x[:, None, None, :]
    )
    dy = c * jnp.sum(g_gathered * weights_3d_dy, axis=(-3, -2, -1))

    weights_3d_dz = (
        dweights_z[:, :, None, None]
        * weights_y[:, None, :, None]
        * weights_x[:, None, None, :]
    )
    dz = c * jnp.sum(g_gathered * weights_3d_dz, axis=(-3, -2, -1))

    return dx, dy, dz


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _spread_3d(x, y, z, c, variance, voxel_size, nz, ny, nx, nspread, use_erf):
    return _spread_3d_impl(
        x,
        y,
        z,
        c,
        nz,
        ny,
        nx,
        variance=variance,
        voxel_size=voxel_size,
        nspread=nspread,
        use_erf=use_erf,
    )


def _spread_3d_fwd(x, y, z, c, variance, voxel_size, nz, ny, nx, nspread, use_erf):
    out = _spread_3d_impl(
        x,
        y,
        z,
        c,
        nz,
        ny,
        nx,
        variance=variance,
        voxel_size=voxel_size,
        nspread=nspread,
        use_erf=use_erf,
    )
    return out, (x, y, z, variance, voxel_size, c)


def _spread_3d_bwd(nz, ny, nx, nspread, use_erf, res, g):
    x, y, z, variance, voxel_size, c = res
    kwargs = dict(
        variance=variance, voxel_size=voxel_size, nspread=nspread, use_erf=use_erf
    )

    dc = _interp_3d_impl(x, y, z, g, nz, ny, nx, **kwargs)
    dx, dy, dz = _spread_3d_grad_xyz(x, y, z, c, g, nz, ny, nx, **kwargs)

    indices_x, z_x = _support_indices_and_offsets(x, nx, nspread)
    indices_y, z_y = _support_indices_and_offsets(y, ny, nspread)
    indices_z, z_z = _support_indices_and_offsets(z, nz, nspread)
    indices_3d = (
        indices_z[:, :, None, None] * (nx * ny)
        + indices_y[:, None, :, None] * nx
        + indices_x[:, None, None, :]
    )
    g_gathered = g.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)

    def weighted(variance, voxel_size):
        weights_3d = _kernel_weights_3d(
            z_x, z_y, z_z, variance, voxel_size, use_erf=use_erf
        )
        return c[:, None, None, None] * weights_3d

    _, vjp_fn = jax.vjp(weighted, variance, voxel_size)
    dvariance, dvoxel_size = vjp_fn(g_gathered)

    return dx, dy, dz, dc, dvariance, dvoxel_size


_spread_3d.defvjp(_spread_3d_fwd, _spread_3d_bwd)


def spread_3d(
    x: Float[Array, " M"],
    y: Float[Array, " M"],
    z: Float[Array, " M"],
    c: Float[Array, " M"],
    shape: tuple[int, int, int],
    *,
    variance: Float[Array, ""] | Float[Array, " M"],
    voxel_size: Float[Array, ""],
    nspread: int,
    use_erf: bool,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    """Spread point strengths `c` onto a 3D grid of the given
    `shape = (nz, ny, nx)` with an isotropic Gaussian (or voxel-averaged
    Gaussian) kernel of the given `variance`.

    `x`, `y`, `z` are in grid-index units (see `normalize_positions_to_grid`).
    `c` carries the entire per-point scattering weight (e.g. amplitude times
    atom occupancy); `variance` may be a scalar or a per-point array of shape
    `(M,)`.
    """
    nz, ny, nx = shape
    return _spread_3d(x, y, z, c, variance, voxel_size, nz, ny, nx, nspread, use_erf)
