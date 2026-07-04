"""A pure-JAX Gaussian spreading backend (scatter point strengths onto a
uniform grid), adapted from `nufftax`'s NUFFT type-1 spreading, specialized
to the isotropic Gaussian (and pixel-averaged Gaussian) kernels used by
`cryojax.simulator.IndependentAtomVolume` and
`cryojax.simulator.GaussianMixtureVolume`.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, Float

from ..jax_util import FloatLike, NDArrayLike


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
# (NUFFT type 2) at the same positions/kernel (`_interp_*d_impl`), and the
# gradient w.r.t. positions uses the analytic kernel derivative
# (`_spread_*d_grad_*`), rather than differentiating through
# `jnp.ceil`/`segment_sum` directly.
#
# `amplitude` carries the *entire* per-point scattering weight (e.g. an
# atom's occupancy times its Gaussian amplitude): it multiplies the
# (normalized) kernel exactly once, regardless of dimensionality. `variance`
# only shapes the kernel and is not otherwise special-cased.
#
# The public `spread_2d`/`spread_3d` take physical-unit positions `x`, `y`,
# `z`: `x = 0` corresponds to grid index `n // 2` (the RELION real-space
# center convention used throughout cryojax), for both even and odd `n`.
# Internally, positions are converted to grid-index units
# (`_normalize_coord_to_grid`) and named `i`, `j`, `k` from that point on —
# this differs from `nufftax`, which instead works on the NUFFT domain
# `[-pi, pi)`; there is no NUFFT involved here, so that intermediate
# representation is unnecessary. The conversion is plain, cheap, and
# differentiable, so composing it with the custom-VJP'd core (which works
# entirely in grid-index units) is automatically correct — no changes to the
# custom VJP rule itself are needed.


def spread_2d(
    x: Float[Array, " M"],
    y: Float[Array, " M"],
    amplitude: Float[Array, " M"],
    variance: Float[Array, ""] | Float[Array, " M"],
    shape: tuple[int, int],
    *,
    pixel_size: Float[Array, ""],
    n_spread: int = 7,
    use_erf: bool = True,
) -> Float[Array, "{shape[0]} {shape[1]}"]:
    """Scatter point strengths onto a 2D grid with an isotropic Gaussian
    (or pixel-averaged Gaussian) kernel.

    This scatters each point's strength onto the `n_spread` nearest grid
    points along each axis, weighted by a compactly-supported Gaussian
    kernel. Differentiable with a custom VJP rule w.r.t. all array
    arguments.

    **Arguments:**

    - `x`, `y`:
        Physical-unit positions of shape `(M,)`, where `0` corresponds to
        the real-space center at grid index `n // 2` (for both even and
        odd `n`).
    - `amplitude`:
        The per-point scattering weight, of shape `(M,)` (e.g. an
        amplitude times an atom occupancy). Multiplies the (normalized)
        kernel exactly once, regardless of dimensionality.
    - `variance`:
        The variance of the isotropic Gaussian kernel. May be a scalar or
        a per-point array of shape `(M,)`.
    - `shape`:
        The shape `(ny, nx)` of the output grid.
    - `pixel_size`:
        The pixel size of the output grid, in the same units as `x`, `y`.
    - `n_spread`:
        The width (number of grid points, per axis) of the kernel used to
        spread each point. Controls speed / accuracy tradeoff: larger
        `n_spread` is more accurate but slower. Must be chosen relative to
        `variance` and the pixel/voxel size — too small truncates the
        Gaussian and silently biases the result. See
        [`cryojax.ndimage.variance_to_nspread`][] to pick a value for a given
        `variance`. Must not exceed the smallest dimension of `shape`
        (otherwise a single point's kernel support would wrap around the
        grid more than once, aliasing the result).
    - `use_erf`:
        If `True` (default), spread the exact average of the Gaussian over
        a pixel (used to sample the average value within a pixel, rather
        than its value at a point). If `False`, spread a point-sampled
        Gaussian instead.

    **Returns:**

    The grid of shape `(ny, nx)` with gaussians scattered onto it.
    """
    _check_n_spread(n_spread, shape)
    ny, nx = shape
    i = _normalize_coord_to_grid(x, nx, pixel_size)
    j = _normalize_coord_to_grid(y, ny, pixel_size)
    return _spread_2d(i, j, amplitude, variance, pixel_size, ny, nx, n_spread, use_erf)


def spread_3d(
    x: Float[Array, " M"],
    y: Float[Array, " M"],
    z: Float[Array, " M"],
    amplitude: Float[Array, " M"],
    variance: Float[Array, ""] | Float[Array, " M"],
    shape: tuple[int, int, int],
    *,
    voxel_size: Float[Array, ""],
    n_spread: int = 7,
    use_erf: bool = True,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    """Scatter point strengths onto a 3D grid with an isotropic Gaussian
    (or voxel-averaged Gaussian) kernel.

    This scatters each point's strength onto the `n_spread` nearest grid
    points along each axis, weighted by a compactly-supported Gaussian
    kernel. Differentiable with a custom VJP rule w.r.t. all array
    arguments.

    **Arguments:**

    - `x`, `y`, `z`:
        Physical-unit positions of shape `(M,)`, where `0` corresponds to
        the real-space center at grid index `n // 2` (for both even and
        odd `n`).
    - `amplitude`:
        The per-point scattering weight, of shape `(M,)` (e.g. an
        amplitude times an atom occupancy). Multiplies the (normalized)
        kernel exactly once, regardless of dimensionality.
    - `variance`:
        The variance of the isotropic Gaussian kernel. May be a scalar or
        a per-point array of shape `(M,)`.
    - `shape`:
        The shape `(nz, ny, nx)` of the output grid.
    - `voxel_size`:
        The voxel size of the output grid, in the same units as `x`, `y`,
        `z`.
    - `n_spread`:
        The width (number of grid points, per axis) of the kernel used to
        spread each point. Controls speed / accuracy tradeoff: larger
        `n_spread` is more accurate but slower. Must be chosen relative to
        `variance` and the pixel/voxel size — too small truncates the
        Gaussian and silently biases the result. See
        [`cryojax.ndimage.variance_to_nspread`][] to pick a value for a given
        `variance`. Must not exceed the smallest dimension of `shape`
        (otherwise a single point's kernel support would wrap around the
        grid more than once, aliasing the result).
    - `use_erf`:
        If `True` (default), spread the exact average of the Gaussian over
        a voxel (used to sample the average value within a voxel, rather
        than its value at a point). If `False`, spread a point-sampled
        Gaussian instead.

    **Returns:**

    The grid of shape `(nz, ny, nx)` with gaussians scattered onto it.
    """
    _check_n_spread(n_spread, shape)
    nz, ny, nx = shape
    i = _normalize_coord_to_grid(x, nx, voxel_size)
    j = _normalize_coord_to_grid(y, ny, voxel_size)
    k = _normalize_coord_to_grid(z, nz, voxel_size)
    return _spread_3d(
        i, j, k, amplitude, variance, voxel_size, nz, ny, nx, n_spread, use_erf
    )


def variance_to_nspread(
    variance: FloatLike | Float[NDArrayLike, " M"],
    pixel_size: FloatLike,
    n_sigma: float = 4.0,
) -> int:
    """Choose an `n_spread` sufficient to truncate the Gaussian kernel used by
    [`cryojax.ndimage.spread_2d`][]/[`cryojax.ndimage.spread_3d`][] at
    `n_sigma` standard deviations.

    `n_spread` sets array shapes, so it must be a static value rather than
    depending on `variance` through tracing; call this ahead of time with
    concrete values instead of guessing. Too small an `n_spread` silently
    truncates the Gaussian and biases the result.

    !!! warning
        Not JIT-compatible, and never invokes JAX — `variance` and
        `pixel_size` are handled with plain `numpy`/`math`, so passing
        numpy arrays or python floats never triggers a JAX dispatch or
        device transfer. Call this once outside of any `jax.jit`-compiled
        function (e.g. when choosing `n_spread` up front from known/expected
        variances), not from within a jitted call graph.

    **Arguments:**

    - `variance`:
        The variance (or per-point array of variances — the largest is
        used) that will be passed to `spread_2d`/`spread_3d`.
    - `pixel_size`:
        The pixel/voxel size of the grid `variance` will be spread onto.
    - `n_sigma`:
        The number of standard deviations of the Gaussian to truncate at.

    **Returns:**

    An integer `n_spread`, at least `2`.
    """
    max_variance = float(np.max(np.asarray(variance)))
    n_spread = 2.0 * n_sigma * math.sqrt(max_variance) / float(pixel_size)
    return max(2, math.ceil(n_spread))


# ============================================================================
# Private implementation
# ============================================================================


def _check_n_spread(n_spread: int, shape: tuple[int, ...]) -> None:
    """Guard against `n_spread` exceeding a grid dimension.

    Grid indices wrap modulo the grid size (`_support_indices_and_offsets`),
    so if `n_spread` exceeds a dimension, a single point's kernel support
    wraps around that axis more than once. This does not raise on its own —
    `segment_sum` silently adds the aliased contributions — so without this
    check the grid would be silently corrupted by periodic wraparound
    instead of erroring.
    """
    if n_spread > min(shape):
        raise ValueError(
            f"`n_spread` ({n_spread}) must not exceed the smallest grid "
            f"dimension in `shape` ({shape})."
        )


def _normalize_coord_to_grid(
    coord: Float[Array, " M"], n: int, pixel_size: Float[Array, ""]
) -> Float[Array, " M"]:
    """Convert a physical-unit coordinate to grid-index units, where the
    real-space center is always at integer index `n // 2` (RELION
    convention), for both even and odd `n`."""
    return coord / pixel_size + n // 2


def _gaussian_weight(r: Array, variance: Array) -> Array:
    """Normalized isotropic Gaussian marginal at physical offset `r`."""
    return jnp.exp(-0.5 * r**2 / variance) / jnp.sqrt(2 * jnp.pi * variance)


def _gaussian_weight_and_grad(r: Array, variance: Array) -> tuple[Array, Array]:
    """`_gaussian_weight` and its derivative w.r.t. `r`. Only called where the
    derivative is actually used (the analytic-gradient custom VJP path);
    elsewhere, call `_gaussian_weight` directly rather than computing and
    discarding the derivative."""
    weight = _gaussian_weight(r, variance)
    dweight_dr = -(r / variance) * weight
    return weight, dweight_dr


def _erf_weight(r: Array, variance: Array, pixel_size: Float[Array, ""]) -> Array:
    """Average of `_gaussian_weight` over a pixel/voxel of size `pixel_size`
    centered at physical offset `r`."""
    scaling = 1.0 / jnp.sqrt(2 * variance)
    left, right = scaling * (r - pixel_size / 2), scaling * (r + pixel_size / 2)
    return (jsp.special.erf(right) - jsp.special.erf(left)) / (2 * pixel_size)


def _erf_weight_and_grad(
    r: Array, variance: Array, pixel_size: Float[Array, ""]
) -> tuple[Array, Array]:
    """`_erf_weight` and its derivative w.r.t. `r`. Only called where the
    derivative is actually used (the analytic-gradient custom VJP path);
    elsewhere, call `_erf_weight` directly rather than computing and
    discarding the derivative."""
    scaling = 1.0 / jnp.sqrt(2 * variance)
    left, right = scaling * (r - pixel_size / 2), scaling * (r + pixel_size / 2)
    weight = _erf_weight(r, variance, pixel_size)
    two_over_sqrt_pi = 2.0 / jnp.sqrt(jnp.pi)
    dweight_dr = (
        scaling * two_over_sqrt_pi * (jnp.exp(-(right**2)) - jnp.exp(-(left**2)))
    ) / (2 * pixel_size)
    return weight, dweight_dr


def _kernel_weight_and_grad(
    z: Array, variance: Array, pixel_size: Float[Array, ""], *, use_erf: bool
) -> tuple[Array, Array]:
    """Evaluate the kernel and its derivative w.r.t. the grid-index offset
    `z`, converting to the physical offset `r = z * pixel_size` first
    (`pixel_size` is the pixel/voxel size). Only called from
    `_axis_weights_and_grad`, i.e. where the derivative is actually used (the
    analytic-gradient custom VJP path); see `_kernel_weight` for the
    value-only counterpart used everywhere else.
    """
    r = z * pixel_size
    weight, dweight_dr = (
        _erf_weight_and_grad(r, variance, pixel_size)
        if use_erf
        else _gaussian_weight_and_grad(r, variance)
    )
    return weight, dweight_dr * pixel_size


def _kernel_weight(
    z: Array, variance: Array, pixel_size: Float[Array, ""], *, use_erf: bool
):
    """Evaluate the kernel value only (no derivative) at the grid-index
    offset `z`. Used wherever only the forward value is needed (the spread
    forward pass, and the `variance`/`pixel_size` autodiff closures in the
    custom VJP backward passes)."""
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
    """Per-axis spread indices, kernel weights, and their `d/d(coord)`."""
    indices, z = _support_indices_and_offsets(coord, n, n_spread)
    variance = _broadcast_per_point(variance)
    weight, dweight_dz = _kernel_weight_and_grad(z, variance, pixel_size, use_erf=use_erf)
    # z = index - coord, so d(weight)/d(coord) = -dweight_dz.
    return indices, weight, -dweight_dz


# ── 2-D ──────────────────────────────────────────────────────────────────────


def _kernel_weights_2d(z_i, z_j, variance, pixel_size, *, use_erf):
    variance = _broadcast_per_point(variance)
    weights_i = _kernel_weight(z_i, variance, pixel_size, use_erf=use_erf)
    weights_j = _kernel_weight(z_j, variance, pixel_size, use_erf=use_erf)
    return weights_j[:, :, None] * weights_i[:, None, :]


def _spread_2d_impl(i, j, amplitude, variance, ny, nx, *, pixel_size, n_spread, use_erf):
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    weights_2d = _kernel_weights_2d(z_i, z_j, variance, pixel_size, use_erf=use_erf)
    weighted_amplitude = amplitude[:, None, None] * weights_2d
    indices_2d = indices_j[:, :, None] * nx + indices_i[:, None, :]
    fw = jax.ops.segment_sum(
        weighted_amplitude.ravel(), indices_2d.ravel(), num_segments=ny * nx
    )
    return fw.reshape(ny, nx)


def _interp_2d_impl(i, j, fw, variance, ny, nx, *, pixel_size, n_spread, use_erf):
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    weights_2d = _kernel_weights_2d(z_i, z_j, variance, pixel_size, use_erf=use_erf)
    indices_2d = indices_j[:, :, None] * nx + indices_i[:, None, :]
    fw_gathered = fw.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)
    return jnp.sum(fw_gathered * weights_2d, axis=(-2, -1))


def _spread_2d_grad_ij(
    i, j, amplitude, variance, g, ny, nx, *, pixel_size, n_spread, use_erf
):
    """Analytic gradient of `_spread_2d_impl` w.r.t. `i` and `j`."""
    indices_i, weights_i, dweights_i = _axis_weights_and_grad(
        i, nx, variance, n_spread, pixel_size, use_erf=use_erf
    )
    indices_j, weights_j, dweights_j = _axis_weights_and_grad(
        j, ny, variance, n_spread, pixel_size, use_erf=use_erf
    )
    indices_2d = indices_j[:, :, None] * nx + indices_i[:, None, :]
    g_gathered = g.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)

    weights_2d_di = weights_j[:, :, None] * dweights_i[:, None, :]
    di = amplitude * jnp.sum(g_gathered * weights_2d_di, axis=(-2, -1))

    weights_2d_dj = dweights_j[:, :, None] * weights_i[:, None, :]
    dj = amplitude * jnp.sum(g_gathered * weights_2d_dj, axis=(-2, -1))

    return di, dj


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _spread_2d(i, j, amplitude, variance, pixel_size, ny, nx, n_spread, use_erf):
    return _spread_2d_impl(
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
    out = _spread_2d_impl(
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


def _spread_2d_bwd(ny, nx, n_spread, use_erf, res, g):
    i, j, amplitude, variance, pixel_size = res

    # The adjoint of spreading is interpolation, at the same positions/kernel.
    damplitude = _interp_2d_impl(
        i,
        j,
        g,
        variance,
        ny,
        nx,
        pixel_size=pixel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    # Position gradients use the analytic kernel derivative.
    di, dj = _spread_2d_grad_ij(
        i,
        j,
        amplitude,
        variance,
        g,
        ny,
        nx,
        pixel_size=pixel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )

    # `variance`/`pixel_size` also shape the kernel; differentiate through the
    # (cheap) weight evaluation directly for these.
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    indices_2d = indices_j[:, :, None] * nx + indices_i[:, None, :]
    g_gathered = g.ravel()[indices_2d.ravel()].reshape(indices_2d.shape)

    def weighted(variance, pixel_size):
        weights_2d = _kernel_weights_2d(z_i, z_j, variance, pixel_size, use_erf=use_erf)
        return amplitude[:, None, None] * weights_2d

    _, vjp_fn = jax.vjp(weighted, variance, pixel_size)
    dvariance, dpixel_size = vjp_fn(g_gathered)

    return di, dj, damplitude, dvariance, dpixel_size


_spread_2d.defvjp(_spread_2d_fwd, _spread_2d_bwd)


# ── 3-D ──────────────────────────────────────────────────────────────────────


def _kernel_weights_3d(z_i, z_j, z_k, variance, pixel_size, *, use_erf):
    variance = _broadcast_per_point(variance)
    weights_i = _kernel_weight(z_i, variance, pixel_size, use_erf=use_erf)
    weights_j = _kernel_weight(z_j, variance, pixel_size, use_erf=use_erf)
    weights_k = _kernel_weight(z_k, variance, pixel_size, use_erf=use_erf)
    return (
        weights_k[:, :, None, None]
        * weights_j[:, None, :, None]
        * weights_i[:, None, None, :]
    )


def _spread_3d_impl(
    i, j, k, amplitude, variance, nz, ny, nx, *, voxel_size, n_spread, use_erf
):
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    indices_k, z_k = _support_indices_and_offsets(k, nz, n_spread)
    weights_3d = _kernel_weights_3d(z_i, z_j, z_k, variance, voxel_size, use_erf=use_erf)
    weighted_amplitude = amplitude[:, None, None, None] * weights_3d
    indices_3d = (
        indices_k[:, :, None, None] * (nx * ny)
        + indices_j[:, None, :, None] * nx
        + indices_i[:, None, None, :]
    )
    fw = jax.ops.segment_sum(
        weighted_amplitude.ravel(), indices_3d.ravel(), num_segments=nz * ny * nx
    )
    return fw.reshape(nz, ny, nx)


def _interp_3d_impl(i, j, k, fw, variance, nz, ny, nx, *, voxel_size, n_spread, use_erf):
    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    indices_k, z_k = _support_indices_and_offsets(k, nz, n_spread)
    weights_3d = _kernel_weights_3d(z_i, z_j, z_k, variance, voxel_size, use_erf=use_erf)
    indices_3d = (
        indices_k[:, :, None, None] * (nx * ny)
        + indices_j[:, None, :, None] * nx
        + indices_i[:, None, None, :]
    )
    fw_gathered = fw.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)
    return jnp.sum(fw_gathered * weights_3d, axis=(-3, -2, -1))


def _spread_3d_grad_ijk(
    i, j, k, amplitude, variance, g, nz, ny, nx, *, voxel_size, n_spread, use_erf
):
    """Analytic gradient of `_spread_3d_impl` w.r.t. `i`, `j`, and `k`."""
    indices_i, weights_i, dweights_i = _axis_weights_and_grad(
        i, nx, variance, n_spread, voxel_size, use_erf=use_erf
    )
    indices_j, weights_j, dweights_j = _axis_weights_and_grad(
        j, ny, variance, n_spread, voxel_size, use_erf=use_erf
    )
    indices_k, weights_k, dweights_k = _axis_weights_and_grad(
        k, nz, variance, n_spread, voxel_size, use_erf=use_erf
    )
    indices_3d = (
        indices_k[:, :, None, None] * (nx * ny)
        + indices_j[:, None, :, None] * nx
        + indices_i[:, None, None, :]
    )
    g_gathered = g.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)

    weights_3d_di = (
        weights_k[:, :, None, None]
        * weights_j[:, None, :, None]
        * dweights_i[:, None, None, :]
    )
    di = amplitude * jnp.sum(g_gathered * weights_3d_di, axis=(-3, -2, -1))

    weights_3d_dj = (
        weights_k[:, :, None, None]
        * dweights_j[:, None, :, None]
        * weights_i[:, None, None, :]
    )
    dj = amplitude * jnp.sum(g_gathered * weights_3d_dj, axis=(-3, -2, -1))

    weights_3d_dk = (
        dweights_k[:, :, None, None]
        * weights_j[:, None, :, None]
        * weights_i[:, None, None, :]
    )
    dk = amplitude * jnp.sum(g_gathered * weights_3d_dk, axis=(-3, -2, -1))

    return di, dj, dk


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _spread_3d(i, j, k, amplitude, variance, voxel_size, nz, ny, nx, n_spread, use_erf):
    return _spread_3d_impl(
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
    out = _spread_3d_impl(
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


def _spread_3d_bwd(nz, ny, nx, n_spread, use_erf, res, g):
    i, j, k, amplitude, variance, voxel_size = res

    damplitude = _interp_3d_impl(
        i,
        j,
        k,
        g,
        variance,
        nz,
        ny,
        nx,
        voxel_size=voxel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )
    di, dj, dk = _spread_3d_grad_ijk(
        i,
        j,
        k,
        amplitude,
        variance,
        g,
        nz,
        ny,
        nx,
        voxel_size=voxel_size,
        n_spread=n_spread,
        use_erf=use_erf,
    )

    indices_i, z_i = _support_indices_and_offsets(i, nx, n_spread)
    indices_j, z_j = _support_indices_and_offsets(j, ny, n_spread)
    indices_k, z_k = _support_indices_and_offsets(k, nz, n_spread)
    indices_3d = (
        indices_k[:, :, None, None] * (nx * ny)
        + indices_j[:, None, :, None] * nx
        + indices_i[:, None, None, :]
    )
    g_gathered = g.ravel()[indices_3d.ravel()].reshape(indices_3d.shape)

    def weighted(variance, voxel_size):
        weights_3d = _kernel_weights_3d(
            z_i, z_j, z_k, variance, voxel_size, use_erf=use_erf
        )
        return amplitude[:, None, None, None] * weights_3d

    _, vjp_fn = jax.vjp(weighted, variance, voxel_size)
    dvariance, dvoxel_size = vjp_fn(g_gathered)

    return di, dj, dk, damplitude, dvariance, dvoxel_size


_spread_3d.defvjp(_spread_3d_fwd, _spread_3d_bwd)
