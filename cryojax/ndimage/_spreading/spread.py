"""The pure-JAX Gaussian spreading backend (scatter point strengths onto a
uniform grid), adapted from `nufftax`'s NUFFT type-1 spreading, specialized
to the isotropic Gaussian (and pixel-averaged Gaussian) kernels used by
`cryojax.simulator.IndependentAtomVolume` and
`cryojax.simulator.GaussianMixtureVolume`.

This is the pallas-agnostic reference implementation: it works standalone
(e.g. on a machine with no GPU) and is the trusted reference the Pallas
kernels in `pallas_spread.py` are validated against.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float


# ============================================================================
# Gaussian spreading backend
# ============================================================================
#
# Scatters point strengths `amplitude` onto a uniform grid, weighted by a
# compactly supported (`n_spread` grid points wide) isotropic Gaussian
# centered at each point, or by the exact average of that Gaussian over a
# pixel/voxel (the "erf" kernel, used to sample the average value within a
# pixel rather than its value at a point).
#
# The custom VJP rule mirrors `nufftax`'s spreading (NUFFT type 1): the
# gradient w.r.t. `amplitude` is the *adjoint* operation, i.e. interpolation
# (NUFFT type 2) at the same positions/kernel, and the gradients w.r.t.
# positions/`variance`/`pixel_size` use the analytic kernel derivatives
# (`_spread_*d_bwd`), rather than differentiating through
# `jnp.ceil`/`segment_sum`/`jax.vjp` directly.
#
# `amplitude` carries the *entire* per-point scattering weight (e.g. an
# atom's occupancy times its Gaussian amplitude): it multiplies the
# (normalized) kernel exactly once, regardless of dimensionality. `variance`
# only shapes the kernel and is not otherwise special-cased.
#
# Positions `i`, `j`, `k` here are already in grid-index units (the physical
# -> grid-index conversion, which depends on `pixel_size`, happens in
# `api.py`, outside this custom-VJP boundary, so it is differentiated
# automatically/correctly by ordinary JAX autodiff composing with the
# analytic rules below).


def _gaussian_weight(r: Array, variance: Array) -> Array:
    """Normalized isotropic Gaussian marginal at physical offset `r`."""
    return jnp.exp(-0.5 * r**2 / variance) / jnp.sqrt(2 * jnp.pi * variance)


def _gaussian_weight_and_grads(
    r: Array, variance: Array, pixel_size: Float[Array, ""]
) -> tuple[Array, Array, Array, Array]:
    """`_gaussian_weight` and its derivatives w.r.t. `r`, `variance`, and
    `pixel_size` (the latter holding the grid-index offset `z = r /
    pixel_size` fixed, matching the `z`-parameterization used throughout the
    backward pass). Only called from `_kernel_weight_and_grads`, i.e. where
    derivatives are actually used (the analytic-gradient custom VJP path);
    elsewhere, call `_gaussian_weight` directly."""
    weight = _gaussian_weight(r, variance)
    dweight_dr = -(r / variance) * weight
    dweight_dvariance = weight * (r**2 / variance - 1.0) / (2.0 * variance)
    dweight_dpixel_size = dweight_dr * (r / pixel_size)
    return weight, dweight_dr, dweight_dvariance, dweight_dpixel_size


def _erf_weight(r: Array, variance: Array, pixel_size: Float[Array, ""]) -> Array:
    """Average of `_gaussian_weight` over a pixel/voxel of size `pixel_size`
    centered at physical offset `r`."""
    scaling = 1.0 / jnp.sqrt(2 * variance)
    left, right = scaling * (r - pixel_size / 2), scaling * (r + pixel_size / 2)
    return (jsp.special.erf(right) - jsp.special.erf(left)) / (2 * pixel_size)


def _erf_weight_and_grads(
    r: Array, variance: Array, pixel_size: Float[Array, ""]
) -> tuple[Array, Array, Array, Array]:
    """`_erf_weight` and its derivatives w.r.t. `r`, `variance`, and
    `pixel_size` (the latter holding `z = r / pixel_size` fixed). Only
    called from `_kernel_weight_and_grads`, i.e. where derivatives are
    actually used (the analytic-gradient custom VJP path); elsewhere, call
    `_erf_weight` directly rather than computing and discarding derivatives.
    """
    scaling = 1.0 / jnp.sqrt(2 * variance)
    left, right = scaling * (r - pixel_size / 2), scaling * (r + pixel_size / 2)
    exp_left, exp_right = jnp.exp(-(left**2)), jnp.exp(-(right**2))
    two_over_sqrt_pi = 2.0 / jnp.sqrt(jnp.pi)
    weight = (jsp.special.erf(right) - jsp.special.erf(left)) / (2 * pixel_size)
    # Both derivatives below reuse the same two exponentials computed above
    # rather than re-invoking `erf`/`exp`; `diff_term` is (proportional to)
    # `_erf_weight`'s own d/dr, `lever_term` is the extra combination needed
    # for the `variance`/`pixel_size` derivatives.
    diff_term = two_over_sqrt_pi * (exp_right - exp_left)
    lever_term = two_over_sqrt_pi * (exp_right * right - exp_left * left)
    dweight_dr = scaling * diff_term / (2 * pixel_size)
    dweight_dvariance = -lever_term / (4 * pixel_size * variance)
    dweight_dpixel_size = lever_term / (2 * pixel_size**2) - weight / pixel_size
    return weight, dweight_dr, dweight_dvariance, dweight_dpixel_size


def _kernel_weight_and_grads(
    z: Array, variance: Array, pixel_size: Float[Array, ""], *, use_erf: bool
) -> tuple[Array, Array, Array, Array]:
    """Evaluate the kernel and its derivatives w.r.t. the grid-index offset
    `z`, `variance`, and `pixel_size`, converting to the physical offset `r
    = z * pixel_size` first. Only called from `_axis_weights_and_grad`, i.e.
    where derivatives are actually used (the analytic-gradient custom VJP
    path); see `_kernel_weight` for the value-only counterpart used
    everywhere else.
    """
    r = z * pixel_size
    weight, dweight_dr, dweight_dvariance, dweight_dpixel_size = (
        _erf_weight_and_grads(r, variance, pixel_size)
        if use_erf
        else _gaussian_weight_and_grads(r, variance, pixel_size)
    )
    return weight, dweight_dr * pixel_size, dweight_dvariance, dweight_dpixel_size


def _kernel_weight(
    z: Array, variance: Array, pixel_size: Float[Array, ""], *, use_erf: bool
):
    """Evaluate the kernel value only (no derivative) at the grid-index
    offset `z`. Used wherever only the forward value is needed (the spread
    forward pass and the interpolation/adjoint value)."""
    r = z * pixel_size
    if use_erf:
        return _erf_weight(r, variance, pixel_size)
    return _gaussian_weight(r, variance)


def _broadcast_per_point(value: Array) -> Array:
    """Add a trailing axis to a per-point array of shape `(M,)` so that it
    broadcasts against a `(M, n_spread)` array of per-point kernel offsets.
    Scalars (shape `()`) are left unchanged.
    """
    return value[:, None] if jnp.ndim(value) == 1 else value


def _support_indices_and_offsets(
    coord: Float[Array, " M"], n: int, n_spread: int
) -> tuple[Array, Array]:
    """Per-axis spread indices (wrapped to `[0, n)`) and kernel offsets `z`."""
    i0 = jnp.ceil(coord - n_spread / 2.0).astype(jnp.int32)
    offsets = jnp.arange(n_spread)
    indices = i0[:, None] + offsets[None, :]  # (M, n_spread)
    z = indices.astype(coord.dtype) - coord[:, None]
    return indices % n, z


def _axis_weights_and_grad(coord, n, variance, n_spread, pixel_size, *, use_erf):
    """Per-axis spread indices, kernel weights, and their derivatives w.r.t.
    `coord`, `variance`, and `pixel_size`."""
    indices, z = _support_indices_and_offsets(coord, n, n_spread)
    variance = _broadcast_per_point(variance)
    weight, dweight_dz, dweight_dvariance, dweight_dpixel_size = _kernel_weight_and_grads(
        z, variance, pixel_size, use_erf=use_erf
    )
    # z = index - coord, so d(weight)/d(coord) = -dweight_dz.
    return indices, weight, -dweight_dz, dweight_dvariance, dweight_dpixel_size


# ── 2-D ──────────────────────────────────────────────────────────────────────


def spread_2d_impl(i, j, amplitude, variance, ny, nx, *, pixel_size, n_spread, use_erf):
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    variance = _broadcast_per_point(variance)
    weights_i = _kernel_weight(z_i, variance, pixel_size, use_erf=use_erf)
    weights_j = _kernel_weight(z_j, variance, pixel_size, use_erf=use_erf)
    # Fold `amplitude` into one 1-D factor before forming the (M, n_spread,
    # n_spread) outer product, rather than materializing the un-weighted
    # outer product and a separately amplitude-scaled copy of it.
    weighted_amplitude = (amplitude[:, None] * weights_j)[:, :, None] * weights_i[
        :, None, :
    ]
    indices_2d = indices_j[:, :, None] * nx + indices_i[:, None, :]
    fw = jax.ops.segment_sum(
        weighted_amplitude.ravel(), indices_2d.ravel(), num_segments=ny * nx
    )
    return fw.reshape(ny, nx)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _spread_2d(i, j, amplitude, variance, pixel_size, ny, nx, n_spread, use_erf):
    return spread_2d_impl(
        i,
        j,
        amplitude,
        variance,
        ny,
        nx,
        pixel_size=pixel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )


def _spread_2d_fwd(i, j, amplitude, variance, pixel_size, ny, nx, n_spread, use_erf):
    out = spread_2d_impl(
        i,
        j,
        amplitude,
        variance,
        ny,
        nx,
        pixel_size=pixel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    return out, (i, j, amplitude, variance, pixel_size)


def spread_2d_bwd(ny, nx, n_spread, use_erf, res, g):
    """Analytic backward pass. The adjoint of spreading is interpolation at
    the same positions/kernel (`damplitude`); position gradients use the
    analytic kernel derivative w.r.t. the grid offset; `variance`/
    `pixel_size` gradients use the analytic kernel derivative w.r.t. those
    (computed alongside the others in `_axis_weights_and_grad`, rather than
    via a separate `jax.vjp` closure that would re-evaluate the kernel).

    Every quantity here is computed once and reused: the (M, n_spread)
    per-axis weights/derivatives, and the (M, n_spread, n_spread) gathered
    cotangent `g_gathered`. The cotangent is contracted down one axis at a
    time (`g_dot_j`, `g_dot_i`) so that no (M, n_spread, n_spread) *weight*
    tensor is ever materialized (unlike the forward pass, where that full
    outer product is unavoidable — it is exactly the data `segment_sum`
    scatters).
    """
    i, j, amplitude, variance, pixel_size = res

    indices_i, weights_i, dwi_coord, dwi_variance, dwi_pixel = _axis_weights_and_grad(
        i, nx, variance, n_spread, pixel_size, use_erf=use_erf
    )
    indices_j, weights_j, dwj_coord, dwj_variance, dwj_pixel = _axis_weights_and_grad(
        j, ny, variance, n_spread, pixel_size, use_erf=use_erf
    )
    indices_2d = indices_j[:, :, None] * nx + indices_i[:, None, :]
    g_gathered = g.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)

    g_dot_j = jnp.sum(g_gathered * weights_j[:, :, None], axis=-2)  # (M, n_spread_i)
    g_dot_i = jnp.sum(g_gathered * weights_i[:, None, :], axis=-1)  # (M, n_spread_j)

    damplitude = jnp.sum(g_dot_j * weights_i, axis=-1)

    di = amplitude * jnp.sum(g_dot_j * dwi_coord, axis=-1)
    dj = amplitude * jnp.sum(g_dot_i * dwj_coord, axis=-1)

    dvariance_per_point = amplitude * (
        jnp.sum(g_dot_j * dwi_variance, axis=-1)
        + jnp.sum(g_dot_i * dwj_variance, axis=-1)
    )
    dpixel_size_per_point = amplitude * (
        jnp.sum(g_dot_j * dwi_pixel, axis=-1) + jnp.sum(g_dot_i * dwj_pixel, axis=-1)
    )
    dvariance = (
        jnp.sum(dvariance_per_point) if jnp.ndim(variance) == 0 else dvariance_per_point
    )
    dpixel_size = jnp.sum(dpixel_size_per_point)

    return di, dj, damplitude, dvariance, dpixel_size


_spread_2d.defvjp(_spread_2d_fwd, spread_2d_bwd)


# ── 3-D ──────────────────────────────────────────────────────────────────────


def spread_3d_impl(
    i, j, k, amplitude, variance, nz, ny, nx, *, voxel_size, n_spread, use_erf
):
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    indices_k, z_k = _support_indices_and_offsets(k, nz, n_spread)
    variance = _broadcast_per_point(variance)
    weights_i = _kernel_weight(z_i, variance, voxel_size, use_erf=use_erf)
    weights_j = _kernel_weight(z_j, variance, voxel_size, use_erf=use_erf)
    weights_k = _kernel_weight(z_k, variance, voxel_size, use_erf=use_erf)
    # Fold `amplitude` into one 1-D factor before forming the (M, n_spread,
    # n_spread, n_spread) outer product (see `spread_2d_impl`).
    weighted_amplitude = (
        (amplitude[:, None] * weights_k)[:, :, None, None]
        * weights_j[:, None, :, None]
        * weights_i[:, None, None, :]
    )
    indices_3d = (
        indices_k[:, :, None, None] * (nx * ny)
        + indices_j[:, None, :, None] * nx
        + indices_i[:, None, None, :]
    )
    fw = jax.ops.segment_sum(
        weighted_amplitude.ravel(), indices_3d.ravel(), num_segments=nz * ny * nx
    )
    return fw.reshape(nz, ny, nx)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _spread_3d(i, j, k, amplitude, variance, voxel_size, nz, ny, nx, n_spread, use_erf):
    return spread_3d_impl(
        i,
        j,
        k,
        amplitude,
        variance,
        nz,
        ny,
        nx,
        voxel_size=voxel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )


def _spread_3d_fwd(
    i, j, k, amplitude, variance, voxel_size, nz, ny, nx, n_spread, use_erf
):
    out = spread_3d_impl(
        i,
        j,
        k,
        amplitude,
        variance,
        nz,
        ny,
        nx,
        voxel_size=voxel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    return out, (i, j, k, amplitude, variance, voxel_size)


def spread_3d_bwd(nz, ny, nx, n_spread, use_erf, res, g):
    """Analytic backward pass (see `spread_2d_bwd`). No (M, n_spread,
    n_spread, n_spread) *weight* tensor is ever materialized: the (M,
    n_spread, n_spread, n_spread) gathered cotangent is contracted down one
    axis at a time, and each of the two "drop one axis" contraction chains
    (`g_dot_k`, `g_dot_i`) is reused for both quantities that need it (e.g.
    `g_dot_k` feeds both the i-axis and j-axis derivatives).
    """
    i, j, k, amplitude, variance, voxel_size = res

    indices_i, weights_i, dwi_coord, dwi_variance, dwi_pixel = _axis_weights_and_grad(
        i, nx, variance, n_spread, voxel_size, use_erf=use_erf
    )
    indices_j, weights_j, dwj_coord, dwj_variance, dwj_pixel = _axis_weights_and_grad(
        j, ny, variance, n_spread, voxel_size, use_erf=use_erf
    )
    indices_k, weights_k, dwk_coord, dwk_variance, dwk_pixel = _axis_weights_and_grad(
        k, nz, variance, n_spread, voxel_size, use_erf=use_erf
    )
    indices_3d = (
        indices_k[:, :, None, None] * (nx * ny)
        + indices_j[:, None, :, None] * nx
        + indices_i[:, None, None, :]
    )
    g_gathered = g.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)  # (M, K, J, I)

    g_dot_k = jnp.sum(g_gathered * weights_k[:, :, None, None], axis=-3)  # (M, J, I)
    g_dot_i = jnp.sum(g_gathered * weights_i[:, None, None, :], axis=-1)  # (M, K, J)

    g_dot_kj = jnp.sum(g_dot_k * weights_j[:, :, None], axis=-2)  # (M, I), leaves i
    g_dot_ki = jnp.sum(g_dot_k * weights_i[:, None, :], axis=-1)  # (M, J), leaves j
    g_dot_ij = jnp.sum(g_dot_i * weights_j[:, None, :], axis=-1)  # (M, K), leaves k

    damplitude = jnp.sum(g_dot_kj * weights_i, axis=-1)

    di = amplitude * jnp.sum(g_dot_kj * dwi_coord, axis=-1)
    dj = amplitude * jnp.sum(g_dot_ki * dwj_coord, axis=-1)
    dk = amplitude * jnp.sum(g_dot_ij * dwk_coord, axis=-1)

    dvariance_per_point = amplitude * (
        jnp.sum(g_dot_kj * dwi_variance, axis=-1)
        + jnp.sum(g_dot_ki * dwj_variance, axis=-1)
        + jnp.sum(g_dot_ij * dwk_variance, axis=-1)
    )
    dvoxel_size_per_point = amplitude * (
        jnp.sum(g_dot_kj * dwi_pixel, axis=-1)
        + jnp.sum(g_dot_ki * dwj_pixel, axis=-1)
        + jnp.sum(g_dot_ij * dwk_pixel, axis=-1)
    )
    dvariance = (
        jnp.sum(dvariance_per_point) if jnp.ndim(variance) == 0 else dvariance_per_point
    )
    dvoxel_size = jnp.sum(dvoxel_size_per_point)

    return di, dj, dk, damplitude, dvariance, dvoxel_size


_spread_3d.defvjp(_spread_3d_fwd, spread_3d_bwd)
