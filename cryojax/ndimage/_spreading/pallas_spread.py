"""Pallas/Triton GPU kernel backend for Gaussian spreading, modeled on
FINUFFT's CUDA spreader (`gpu_method=1`, "nupts-driven": a grid-stride loop
over points, atomically scattering each point's kernel footprint straight
into the global output array). Unlike the pure-JAX backend in `spread.py`
(which materializes an `(M, n_spread^d)` buffer of per-point kernel weights
via `segment_sum`), this backend never holds more than `O(block_size *
n_spread^d)` at once per kernel program, for a total footprint of `O(M)`
across the whole call.

The primal (`pallas_spread_{2,3}d`) scatters, so it needs atomics (only
available on the Triton backend). Its VJP (`pallas_spread_vjp_{2,3}d`) is
a pure gather (the adjoint of spreading is interpolation, exactly as in
`spread.py`), so no atomics are needed there at all: each point reads its
own fixed-size neighborhood of the output cotangent independently. The one
exception is the final reduction of `dvariance`/`dpixel_size` to a scalar
when `variance` is shared across points, which is left to plain `jnp.sum`
outside the kernel (cheap relative to the `O(M)` per-point outputs the
kernel itself produces), rather than an in-kernel atomic reduction.

Benchmarking (see project history/memory) found the *forward* kernel rarely
beats the pure-JAX backend outright (serial `atomic_add` contention), but
the *backward* kernel is a real, consistent win (no such contention -- it's
a gather) in both memory and speed. There is no single best
`enable_pallas` configuration across hardware/scale; see that parameter's
docstring in `api.py` for the actual recommendation. The `enable_pallas`
custom-VJP dispatch that chooses between this file's kernels and the
pure-JAX backend in `spread.py` lives in `api.py`, alongside the public
`spread_gaussians_2d`/`spread_gaussians_3d` functions that call it.

Implementation notes, each hard-won prototyping directly on GPU (see the
`pallas-triton-gotchas` note):
- `jax.scipy.special.erf` does not lower in Pallas-Triton (replaced by
  `_erf_approx` below, accurate to ~3.6e-7 vs. `jsp.special.erf`).
- `pl.pallas_call` output buffers are *not* zero-initialized, so the forward
  (atomic-add) kernel must explicitly alias a `jnp.zeros(...)` input into
  its output.
- The `n_spread x n_spread[ x n_spread]` neighborhood loop is implemented
  with nested `jax.lax.fori_loop`, not a Python-unrolled loop: unrolling
  makes compile time scale with `n_spread**ndim` (up to ~10s at
  `n_spread=17`), while `fori_loop` compiles one copy of the loop body and
  iterates at runtime instead (~130ms regardless of `n_spread`), with no
  correctness or runtime cost (verified byte-identical output, noise-level
  runtime difference).
"""

import math
from functools import cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
from jaxtyping import Array

from .._internal import (
    CompilerParams,
    add,
    batchable,
    choose_block_size,
    direction_specs,
    is_zero,
    jvp_over_nonzero,
    mul,
    neg,
    read_directions,
    tangent_operands,
    tprod2,
    tprod3,
    valid_lane_mask,
)


# ============================================================================
# Kernel-body math (re-implemented from `spread.py`'s reference math using
# only primitives that lower in Pallas-Triton -- see module docstring).
# ============================================================================


def _erf_approx(x: Array) -> Array:
    """Abramowitz & Stegun 7.1.26 rational approximation of `erf`, accurate
    to ~1.5e-7. `jax.scipy.special.erf` (and `jax.lax.erf`) do not lower in
    Pallas-Triton, so this is used instead of `spread._erf_weight`'s exact
    `erf` call whenever kernel weights are computed inside a Pallas kernel
    body."""
    sign = jnp.sign(x)
    ax = jnp.abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - poly * jnp.exp(-ax * ax))


def _kernel_weight(
    z: Array, variance: Array, pixel_size: Array, *, use_erf: bool
) -> Array:
    """Kernel-safe value-only counterpart of `spread._kernel_weight`."""
    r = z * pixel_size
    if use_erf:
        scaling = 1.0 / jnp.sqrt(2 * variance)
        left, right = scaling * (r - pixel_size / 2), scaling * (r + pixel_size / 2)
        return (_erf_approx(right) - _erf_approx(left)) / (2 * pixel_size)
    return jnp.exp(-0.5 * r**2 / variance) / jnp.sqrt(2 * jnp.pi * variance)


def _kernel_weight_and_grads(
    z: Array, variance: Array, pixel_size: Array, *, use_erf: bool
) -> tuple[Array, Array, Array, Array]:
    """Kernel-safe counterpart of `spread._kernel_weight_and_grads`. Returns
    `(weight, dweight_dz, dweight_dvariance, dweight_dpixel_size)` where
    `dweight_dz` already folds in the `dr/dz = pixel_size` chain factor,
    matching `spread.py`'s convention (callers use `-dweight_dz` for the
    coordinate gradient, since `z = index - coord`)."""
    r = z * pixel_size
    if use_erf:
        scaling = 1.0 / jnp.sqrt(2 * variance)
        left, right = scaling * (r - pixel_size / 2), scaling * (r + pixel_size / 2)
        exp_left, exp_right = jnp.exp(-(left**2)), jnp.exp(-(right**2))
        # A plain Python float (not `2.0 / jnp.sqrt(jnp.pi)`): that computes
        # `jnp.sqrt` on a bare constant with no traced array anywhere in the
        # expression to inherit a concrete dtype from, which Pallas-Triton
        # silently resolves to float32 regardless of the surrounding
        # computation's actual (e.g. float64) dtype -- the root cause of a
        # real ('f64', 'f32') lowering crash in the `use_erf=True` branch.
        two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
        weight = (_erf_approx(right) - _erf_approx(left)) / (2 * pixel_size)
        diff_term = two_over_sqrt_pi * (exp_right - exp_left)
        lever_term = two_over_sqrt_pi * (exp_right * right - exp_left * left)
        dweight_dr = scaling * diff_term / (2 * pixel_size)
        dweight_dvariance = -lever_term / (4 * pixel_size * variance)
        dweight_dpixel_size = lever_term / (2 * pixel_size**2) - weight / pixel_size
    else:
        weight = jnp.exp(-0.5 * r**2 / variance) / jnp.sqrt(2 * jnp.pi * variance)
        dweight_dr = -(r / variance) * weight
        dweight_dvariance = weight * (r**2 / variance - 1.0) / (2.0 * variance)
        dweight_dpixel_size = dweight_dr * (r / pixel_size)
    return weight, dweight_dr * pixel_size, dweight_dvariance, dweight_dpixel_size


# ============================================================================
# Primal: atomic-add scatter ("nupts-driven")
# ============================================================================
#
# TODO(pallas-output-driven): FINUFFT's own `gpu_method=0` ("auto") default
# picks `subprob` (`gpu_method=2`), not `nupts-driven`, for NUFFT type-1
# (spread) specifically. A bin-sorted, tile-local-accumulation follow-up
# ("Milestone B" / output-driven) was tried in this project's history and
# found conclusively worse than nupts-driven on every axis benchmarked --
# see project memory. Don't revisit without new information.


@cache
def _make_spread_kernel_2d(
    ny: int, nx: int, n_spread: int, use_erf: bool, block_size: int, m_total: int
):
    def kernel(i_ref, j_ref, amp_ref, var_ref, pixel_size_ref, zeros_ref, out_ref):
        del zeros_ref  # aliased into out_ref; only used to force zero-init
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        def outer_body(oy, carry):
            idx_y = (i0y + oy) % ny
            z_y = (i0y + oy).astype(i.dtype) - j
            wy = _kernel_weight(z_y, variance, pixel_size, use_erf=use_erf)

            def inner_body(ox, carry2):
                idx_x = (i0x + ox) % nx
                z_x = (i0x + ox).astype(i.dtype) - i
                wx = _kernel_weight(z_x, variance, pixel_size, use_erf=use_erf)
                val = amp * wy * wx
                flat = idx_y * nx + idx_x
                pltriton.atomic_add(out_ref, (flat,), val, mask=valid)
                return carry2

            jax.lax.fori_loop(0, n_spread, inner_body, 0)
            return carry

        jax.lax.fori_loop(0, n_spread, outer_body, 0)

    return kernel


@cache
def _make_spread_kernel_3d(
    nz: int, ny: int, nx: int, n_spread: int, use_erf: bool, block_size: int, m_total: int
):
    def kernel(i_ref, j_ref, k_ref, amp_ref, var_ref, voxel_size_ref, zeros_ref, out_ref):
        del zeros_ref
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        def oz_body(oz, carry):
            idx_z = (i0z + oz) % nz
            z_z = (i0z + oz).astype(i.dtype) - k
            wz = _kernel_weight(z_z, variance, voxel_size, use_erf=use_erf)

            def oy_body(oy, carry2):
                idx_y = (i0y + oy) % ny
                z_y = (i0y + oy).astype(i.dtype) - j
                wy = _kernel_weight(z_y, variance, voxel_size, use_erf=use_erf)
                wzy = wz * wy

                def ox_body(ox, carry3):
                    idx_x = (i0x + ox) % nx
                    z_x = (i0x + ox).astype(i.dtype) - i
                    wx = _kernel_weight(z_x, variance, voxel_size, use_erf=use_erf)
                    val = amp * wzy * wx
                    flat = idx_z * (nx * ny) + idx_y * nx + idx_x
                    pltriton.atomic_add(out_ref, (flat,), val, mask=valid)
                    return carry3

                jax.lax.fori_loop(0, n_spread, ox_body, 0)
                return carry2

            jax.lax.fori_loop(0, n_spread, oy_body, 0)
            return carry

        jax.lax.fori_loop(0, n_spread, oz_body, 0)

    return kernel


@partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7, 8))
def pallas_spread_2d(i, j, amplitude, variance, pixel_size, ny, nx, n_spread, use_erf):
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    pixel_size_b = jnp.reshape(pixel_size, (1,)).astype(dtype)
    block_size = choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_kernel_2d(ny, nx, n_spread, use_erf, block_size, m_total)
    zeros = jnp.zeros((ny * nx,), dtype=dtype)
    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((1,), lambda p: (0,)),
            pl.BlockSpec((ny * nx,), lambda p: (0,)),
        ],
        out_specs=pl.BlockSpec((ny * nx,), lambda p: (0,)),
        out_shape=jax.ShapeDtypeStruct((ny * nx,), dtype),
        input_output_aliases={5: 0},
        compiler_params=CompilerParams(),
    )(i, j, amplitude, variance_b, pixel_size_b, zeros)
    return out.reshape(ny, nx)


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10))
def pallas_spread_3d(
    i, j, k, amplitude, variance, voxel_size, nz, ny, nx, n_spread, use_erf
):
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    k = k.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    voxel_size_b = jnp.reshape(voxel_size, (1,)).astype(dtype)
    block_size = choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_kernel_3d(nz, ny, nx, n_spread, use_erf, block_size, m_total)
    zeros = jnp.zeros((nz * ny * nx,), dtype=dtype)
    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((1,), lambda p: (0,)),
            pl.BlockSpec((nz * ny * nx,), lambda p: (0,)),
        ],
        out_specs=pl.BlockSpec((nz * ny * nx,), lambda p: (0,)),
        out_shape=jax.ShapeDtypeStruct((nz * ny * nx,), dtype),
        input_output_aliases={6: 0},
        compiler_params=CompilerParams(),
    )(i, j, k, amplitude, variance_b, voxel_size_b, zeros)
    return out.reshape(nz, ny, nx)


# ============================================================================
# VJP: pure gather ("interpolation", the adjoint of spreading) -- no
# atomics needed; see module docstring.
# ============================================================================


@cache
def _make_spread_vjp_kernel_2d(
    ny: int, nx: int, n_spread: int, use_erf: bool, block_size: int, m_total: int
):
    def kernel(
        i_ref,
        j_ref,
        amp_ref,
        var_ref,
        pixel_size_ref,
        g_ref,
        di_ref,
        dj_ref,
        damp_ref,
        dvar_ref,
        dpix_ref,
    ):
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def outer_body(oy, carry):
            damp, di, dj, dvar, dpix = carry
            idx_y = (i0y + oy) % ny
            z_y = (i0y + oy).astype(i.dtype) - j
            wy, dwy_dz, dwy_dvar, dwy_dpix = _kernel_weight_and_grads(
                z_y, variance, pixel_size, use_erf=use_erf
            )

            def inner_body(ox, carry2):
                damp, di, dj, dvar, dpix = carry2
                idx_x = (i0x + ox) % nx
                z_x = (i0x + ox).astype(i.dtype) - i
                wx, dwx_dz, dwx_dvar, dwx_dpix = _kernel_weight_and_grads(
                    z_x, variance, pixel_size, use_erf=use_erf
                )
                flat = idx_y * nx + idx_x
                g_val = pltriton.load(g_ref.at[flat], mask=valid, other=0.0)

                damp = damp + g_val * wy * wx
                di = di + amp * g_val * wy * (-dwx_dz)
                dj = dj + amp * g_val * wx * (-dwy_dz)
                dvar = dvar + amp * g_val * (wy * dwx_dvar + wx * dwy_dvar)
                dpix = dpix + amp * g_val * (wy * dwx_dpix + wx * dwy_dpix)
                return damp, di, dj, dvar, dpix

            return jax.lax.fori_loop(0, n_spread, inner_body, (damp, di, dj, dvar, dpix))

        damp, di, dj, dvar, dpix = jax.lax.fori_loop(
            0, n_spread, outer_body, (zero, zero, zero, zero, zero)
        )

        pltriton.store(di_ref, di, mask=valid)
        pltriton.store(dj_ref, dj, mask=valid)
        pltriton.store(damp_ref, damp, mask=valid)
        pltriton.store(dvar_ref, dvar, mask=valid)
        pltriton.store(dpix_ref, dpix, mask=valid)

    return kernel


@cache
def _make_spread_vjp_kernel_3d(
    nz: int, ny: int, nx: int, n_spread: int, use_erf: bool, block_size: int, m_total: int
):
    def kernel(
        i_ref,
        j_ref,
        k_ref,
        amp_ref,
        var_ref,
        voxel_size_ref,
        g_ref,
        di_ref,
        dj_ref,
        dk_ref,
        damp_ref,
        dvar_ref,
        dpix_ref,
    ):
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def oz_body(oz, carry):
            damp, di, dj, dk, dvar, dpix = carry
            idx_z = (i0z + oz) % nz
            z_z = (i0z + oz).astype(i.dtype) - k
            wz, dwz_dz, dwz_dvar, dwz_dpix = _kernel_weight_and_grads(
                z_z, variance, voxel_size, use_erf=use_erf
            )

            def oy_body(oy, carry2):
                damp, di, dj, dk, dvar, dpix = carry2
                idx_y = (i0y + oy) % ny
                z_y = (i0y + oy).astype(i.dtype) - j
                wy, dwy_dz, dwy_dvar, dwy_dpix = _kernel_weight_and_grads(
                    z_y, variance, voxel_size, use_erf=use_erf
                )

                def ox_body(ox, carry3):
                    damp, di, dj, dk, dvar, dpix = carry3
                    idx_x = (i0x + ox) % nx
                    z_x = (i0x + ox).astype(i.dtype) - i
                    wx, dwx_dz, dwx_dvar, dwx_dpix = _kernel_weight_and_grads(
                        z_x, variance, voxel_size, use_erf=use_erf
                    )
                    flat = idx_z * (nx * ny) + idx_y * nx + idx_x
                    g_val = pltriton.load(g_ref.at[flat], mask=valid, other=0.0)

                    wzy, wzx, wyx = wz * wy, wz * wx, wy * wx
                    damp = damp + g_val * wzy * wx
                    di = di + amp * g_val * wzy * (-dwx_dz)
                    dj = dj + amp * g_val * wzx * (-dwy_dz)
                    dk = dk + amp * g_val * wyx * (-dwz_dz)
                    dvar = dvar + amp * g_val * (
                        wzy * dwx_dvar + wzx * dwy_dvar + wyx * dwz_dvar
                    )
                    dpix = dpix + amp * g_val * (
                        wzy * dwx_dpix + wzx * dwy_dpix + wyx * dwz_dpix
                    )
                    return damp, di, dj, dk, dvar, dpix

                return jax.lax.fori_loop(
                    0, n_spread, ox_body, (damp, di, dj, dk, dvar, dpix)
                )

            return jax.lax.fori_loop(0, n_spread, oy_body, (damp, di, dj, dk, dvar, dpix))

        damp, di, dj, dk, dvar, dpix = jax.lax.fori_loop(
            0, n_spread, oz_body, (zero, zero, zero, zero, zero, zero)
        )

        pltriton.store(di_ref, di, mask=valid)
        pltriton.store(dj_ref, dj, mask=valid)
        pltriton.store(dk_ref, dk, mask=valid)
        pltriton.store(damp_ref, damp, mask=valid)
        pltriton.store(dvar_ref, dvar, mask=valid)
        pltriton.store(dpix_ref, dpix, mask=valid)

    return kernel


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def pallas_spread_vjp_2d(ny, nx, n_spread, use_erf, res, g):
    i, j, amplitude, variance, pixel_size = res
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    pixel_size_b = jnp.reshape(pixel_size, (1,)).astype(dtype)
    g = g.astype(dtype)
    block_size = choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_vjp_kernel_2d(ny, nx, n_spread, use_erf, block_size, m_total)
    out_shapes = [jax.ShapeDtypeStruct((m_total,), dtype)] * 5
    di, dj, damplitude, dvariance_pp, dpixel_size_pp = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((1,), lambda p: (0,)),
            pl.BlockSpec((ny * nx,), lambda p: (0,)),
        ],
        out_specs=[pl.BlockSpec((block_size,), lambda p: (p,))] * 5,
        out_shape=out_shapes,
        compiler_params=CompilerParams(),
    )(i, j, amplitude, variance_b, pixel_size_b, g.reshape(-1))
    dvariance = jnp.sum(dvariance_pp) if jnp.ndim(variance) == 0 else dvariance_pp
    dpixel_size = jnp.sum(dpixel_size_pp)
    return di, dj, damplitude, dvariance, dpixel_size


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4))
def pallas_spread_vjp_3d(nz, ny, nx, n_spread, use_erf, res, g):
    i, j, k, amplitude, variance, voxel_size = res
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    k = k.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    voxel_size_b = jnp.reshape(voxel_size, (1,)).astype(dtype)
    g = g.astype(dtype)
    block_size = choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_vjp_kernel_3d(
        nz, ny, nx, n_spread, use_erf, block_size, m_total
    )
    out_shapes = [jax.ShapeDtypeStruct((m_total,), dtype)] * 6
    di, dj, dk, damplitude, dvariance_pp, dvoxel_size_pp = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((block_size,), lambda p: (p,)),
            pl.BlockSpec((1,), lambda p: (0,)),
            pl.BlockSpec((nz * ny * nx,), lambda p: (0,)),
        ],
        out_specs=[pl.BlockSpec((block_size,), lambda p: (p,))] * 6,
        out_shape=out_shapes,
        compiler_params=CompilerParams(),
    )(i, j, k, amplitude, variance_b, voxel_size_b, g.reshape(-1))
    dvariance = jnp.sum(dvariance_pp) if jnp.ndim(variance) == 0 else dvariance_pp
    dvoxel_size = jnp.sum(dvoxel_size_pp)
    return di, dj, dk, damplitude, dvariance, dvoxel_size


# ============================================================================
# Forward-mode (JVP) rules
# ============================================================================
#
# JAX cannot differentiate these kernels itself: the scatter aliases its zero-init
# buffer into the output, which `_pallas_call_jvp_rule` refuses outright, and the
# generic rule cannot trace a kernel whose block index maps read `program_id`. Each
# wrapper therefore carries a `jax.custom_jvp` whose tangent is a kernel of its own.
#
# A rule returns the primal kernel and a TANGENT-ONLY kernel as two separate calls,
# never one fused kernel producing both. That is what makes the rules linearizable:
# `jax.linearize` keeps the primal (known) and records the tangent kernel as the
# linear map, so an optimizer that already holds the gradient at a point can apply
# the tangent map there -- Hessian-vector products -- without recomputing the primal
# scatter or gather.
#
# The tangent kernels take any number of DIRECTIONS at one primal point: a `jax.vmap`
# over tangents (a batch of Hessian-vector products, `jax.jacfwd`) is routed by
# `custom_vmap` to one launch that evaluates the weights and their derivatives once
# per neighbour and accumulates every direction, instead of one program per direction.
# They are also SPECIALIZED on which tangents are symbolic zeros
# (`symbolic_zeros=True`): a term multiplied by a zero loaded from memory still costs
# its weight evaluation, so zero terms are dropped at trace time, the nonzero pattern
# being a static key of the cached kernel factory.


def _weight_tangent(dw_dz, dw_dvar, dw_dpix, tcoord, tvar, tpix):
    """Total differential of one separable weight. `z = index - coord`, so the
    coordinate tangent enters with `-dw/dz`."""
    return add(
        None if tcoord is None else -dw_dz * tcoord,
        mul(dw_dvar, tvar),
        mul(dw_dpix, tpix),
    )


# ── 2D ──────────────────────────────────────────────────────────────────────


@cache
def _make_spread_tangent_kernel_2d(
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    nonzero: tuple[bool, bool, bool, bool, bool],
    n_directions: int,
):
    """The tangent of `_make_spread_kernel_2d` in `n_directions` directions:
    `d(a wx wy) = da wx wy + a (twx wy + wx twy)`, weights evaluated once per
    neighbour, one atomic add per direction."""
    nonzero_i, nonzero_j, nonzero_amp, nonzero_var, nonzero_pix = nonzero
    nonzero_x = nonzero_i or nonzero_var or nonzero_pix
    nonzero_y = nonzero_j or nonzero_var or nonzero_pix
    n_nonzero, n_pixels = sum(nonzero), ny * nx
    names = ("i", "j", "amp", "var", "pix")

    def kernel(i_ref, j_ref, amp_ref, var_ref, pixel_size_ref, *refs):
        tangent_refs, tout_ref = refs[:n_nonzero], refs[-1]
        # refs[n_nonzero] is the zeros buffer aliased into the output
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        T = read_directions(tangent_refs, nonzero, names, n_directions)
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        def weight(z, coord_name, nonzero_axis):
            if not nonzero_axis:
                return _kernel_weight(z, variance, pixel_size, use_erf=use_erf), None
            w, dw_dz, dw_dvar, dw_dpix = _kernel_weight_and_grads(
                z, variance, pixel_size, use_erf=use_erf
            )
            return w, [
                _weight_tangent(
                    dw_dz, dw_dvar, dw_dpix, t[coord_name], t["var"], t["pix"]
                )
                for t in T
            ]

        def at(tangents, d):
            return None if tangents is None else tangents[d]

        def outer_body(oy, carry):
            idx_y = (i0y + oy) % ny
            wy, twy = weight((i0y + oy).astype(i.dtype) - j, "j", nonzero_y)

            def inner_body(ox, carry2):
                idx_x = (i0x + ox) % nx
                wx, twx = weight((i0x + ox).astype(i.dtype) - i, "i", nonzero_x)
                flat = idx_y * nx + idx_x
                for d in range(n_directions):
                    tval = add(
                        mul(T[d]["amp"], wx * wy),
                        mul(at(twx, d), amp * wy),
                        mul(at(twy, d), amp * wx),
                    )
                    pltriton.atomic_add(
                        tout_ref, (flat + d * n_pixels,), tval, mask=valid
                    )
                return carry2

            jax.lax.fori_loop(0, n_spread, inner_body, 0)
            return carry

        jax.lax.fori_loop(0, n_spread, outer_body, 0)

    return kernel


@cache
def _make_spread_vjp_tangent_kernel_2d(
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    nonzero: tuple[bool, bool, bool, bool, bool, bool],
    n_directions: int,
):
    """The tangent of `_make_spread_vjp_kernel_2d` in `n_directions` directions, by
    the product rule.

    The gather is linear in `g`, so `g`'s tangent is the gather of `tg` with the
    primal factors. Each residual tangent needs the total differential of every
    per-pixel factor -- second derivatives of the weight -- which `jax.jvp` of
    `_kernel_weight_and_grads` supplies inside the kernel body, forward mode over
    elementwise math, over only the nonzero inputs. The weights and their derivatives
    are evaluated once per neighbour and contracted with every direction; the primal
    gather itself is not accumulated."""
    nonzero_i, nonzero_j, nonzero_amp, nonzero_var, nonzero_pix, nonzero_g = nonzero
    nonzero_x = nonzero_i or nonzero_var or nonzero_pix
    nonzero_y = nonzero_j or nonzero_var or nonzero_pix
    n_nonzero, n_pixels = sum(nonzero), ny * nx
    names = ("i", "j", "amp", "var", "pix", "g")

    def weight_and_grads(z, variance, pixel_size):
        return _kernel_weight_and_grads(z, variance, pixel_size, use_erf=use_erf)

    def kernel(i_ref, j_ref, amp_ref, var_ref, pixel_size_ref, g_ref, *refs):
        tangent_refs, out_refs = refs[:n_nonzero], refs[n_nonzero:]
        odi_ref, odj_ref, odamp_ref, odvar_ref, odpix_ref = out_refs
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        nonzero_names = [n for n, nz_ in zip(names, nonzero) if nz_]
        T = read_directions(
            [r for r, name in zip(tangent_refs, nonzero_names) if name != "g"],
            nonzero[:5],
            names[:5],
            n_directions,
        )
        tg_ref = tangent_refs[-1] if nonzero_g else None
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def weights(z, coord_name, nonzero_axis):
            """Primal `(w, dw/dz, dw/dvar, dw/dpix)` and, per direction, their tangents
            (`None`s when this axis has no nonzero tangent)."""
            primal = weight_and_grads(z, variance, pixel_size)
            if not nonzero_axis:
                return primal, [(None,) * 4] * n_directions
            tangents = []
            for t in T:
                tz = None if t[coord_name] is None else -t[coord_name]
                tangents.append(
                    jvp_over_nonzero(
                        weight_and_grads,
                        (z, variance, pixel_size),
                        (tz, t["var"], t["pix"]),
                    )[1]
                )
            return primal, tangents

        def outer_body(oy, carry):
            idx_y = (i0y + oy) % ny
            (wy, dwy, vwy, hwy), Ty = weights(
                (i0y + oy).astype(i.dtype) - j, "j", nonzero_y
            )

            def inner_body(ox, acc):
                idx_x = (i0x + ox) % nx
                (wx, dwx, vwx, hwx), Tx = weights(
                    (i0x + ox).astype(i.dtype) - i, "i", nonzero_x
                )
                flat = idx_y * nx + idx_x
                g_val = pltriton.load(g_ref.at[flat], mask=valid, other=0.0)

                f_amp = wy * wx
                f_i = -dwx * wy
                f_j = -dwy * wx
                f_var = wy * vwx + wx * vwy
                f_pix = wy * hwx + wx * hwy

                new_acc = []
                for d in range(n_directions):
                    (Twx, Tdwx, Tvwx, Thwx), (Twy, Tdwy, Tvwy, Thwy) = Tx[d], Ty[d]
                    Tf_amp = tprod2(wy, wx, Twy, Twx)
                    Tf_i = neg(tprod2(dwx, wy, Tdwx, Twy))
                    Tf_j = neg(tprod2(dwy, wx, Tdwy, Twx))
                    Tf_var = add(tprod2(wy, vwx, Twy, Tvwx), tprod2(wx, vwy, Twx, Tvwy))
                    Tf_pix = add(tprod2(wy, hwx, Twy, Thwx), tprod2(wx, hwy, Twx, Thwy))
                    tg_val = (
                        pltriton.load(
                            tg_ref.at[flat + d * n_pixels], mask=valid, other=0.0
                        )
                        if nonzero_g
                        else None
                    )
                    tamp = T[d]["amp"]

                    def tangent(f, Tf):
                        return add(
                            mul(tg_val, amp * f),
                            mul(g_val, add(mul(tamp, f), mul(amp, Tf))),
                        )

                    odamp, odi, odj, odvar, odpix = acc[5 * d : 5 * d + 5]
                    new_acc += [
                        add(odamp, mul(tg_val, f_amp), mul(g_val, Tf_amp)),
                        add(odi, tangent(f_i, Tf_i)),
                        add(odj, tangent(f_j, Tf_j)),
                        add(odvar, tangent(f_var, Tf_var)),
                        add(odpix, tangent(f_pix, Tf_pix)),
                    ]
                return tuple(new_acc)

            return jax.lax.fori_loop(0, n_spread, inner_body, carry)

        acc = jax.lax.fori_loop(0, n_spread, outer_body, (zero,) * (5 * n_directions))
        for d in range(n_directions):
            odamp, odi, odj, odvar, odpix = acc[5 * d : 5 * d + 5]
            for ref, value in (
                (odi_ref, odi),
                (odj_ref, odj),
                (odamp_ref, odamp),
                (odvar_ref, odvar),
                (odpix_ref, odpix),
            ):
                pltriton.store(ref.at[d], value, mask=valid)

    return kernel


def _spread_tangent_2d_impl(
    primals, nonzero_tangents, nonzero, ny, nx, n_spread, use_erf
):
    """The scatter's tangent in every direction of `nonzero_tangents`, each of which
    carries a leading direction axis (aligned with the `True`s of `nonzero`):
    `(n_directions, ny, nx)`."""
    i, j, amplitude, variance, pixel_size = primals
    n_directions = nonzero_tangents[0].shape[0]
    m_total, dtype = i.shape[0], i.dtype
    primals_b = (
        i,
        j.astype(dtype),
        amplitude.astype(dtype),
        jnp.broadcast_to(variance, (m_total,)).astype(dtype),
        jnp.reshape(pixel_size, (1,)).astype(dtype),
    )
    names = ("i", "j", "amp", "var", "pix")
    tangents_b = tangent_operands(
        nonzero_tangents,
        [p for p, nz_ in zip(primals_b, nonzero) if nz_],
        [n for n, nz_ in zip(names, nonzero) if nz_],
        dtype,
    )
    block_size = choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_tangent_kernel_2d(
        ny, nx, n_spread, use_erf, block_size, m_total, nonzero, n_directions
    )
    n_pixels = ny * nx
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    timage = pl.BlockSpec((n_directions * n_pixels,), lambda p: (0,))
    tangent_specs = direction_specs(
        nonzero, names, block_size, n_directions, ("pix",), (), n_pixels
    )
    tout = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[per_point] * 4 + [scalar] + tangent_specs + [timage],
        out_specs=timage,
        out_shape=jax.ShapeDtypeStruct((n_directions * n_pixels,), dtype),
        input_output_aliases={5 + len(tangents_b): 0},
        compiler_params=CompilerParams(),
    )(*primals_b, *tangents_b, jnp.zeros((n_directions * n_pixels,), dtype=dtype))
    return tout.reshape(n_directions, ny, nx)


@cache
def _spread_tangent_2d_batchable(ny, nx, n_spread, use_erf, nonzero):
    def impl(primals, tangents):
        return _spread_tangent_2d_impl(
            primals, tangents, nonzero, ny, nx, n_spread, use_erf
        )

    def single(*args):
        primals, tangents = args[:5], args[5:]
        return impl(primals, tuple(t[None] for t in tangents))[0]

    return batchable(single, 5, impl)


def pallas_spread_tangent_2d(
    i, j, amplitude, variance, pixel_size, tangents, ny, nx, n_spread, use_erf
):
    """The tangent of the 2D scatter; `tangents` aligned with the five primals, symbolic
    zeros allowed and dropped at trace time. Under `jax.vmap` over the tangents, every
    direction runs in one launch."""
    nonzero = tuple(not is_zero(t) for t in tangents)
    fn = _spread_tangent_2d_batchable(ny, nx, n_spread, use_erf, nonzero)
    return fn(
        i, j, amplitude, variance, pixel_size, *(t for t in tangents if not is_zero(t))
    )


def _spread_2d_jvp_rule(ny, nx, n_spread, use_erf, primals, tangents):
    # Two calls, never one: the primal stays known under `jax.linearize`.
    out = pallas_spread_2d(*primals, ny, nx, n_spread, use_erf)
    if all(is_zero(t) for t in tangents):
        return out, jnp.zeros_like(out)
    return out, pallas_spread_tangent_2d(*primals, tangents, ny, nx, n_spread, use_erf)


pallas_spread_2d.defjvp(_spread_2d_jvp_rule, symbolic_zeros=True)


def _spread_vjp_tangent_2d_impl(
    res, g, nonzero_tangents, nonzero, ny, nx, n_spread, use_erf
):
    """The gather's tangent -- its five outputs' -- in every direction of
    `nonzero_tangents` (leading direction axis, aligned with the `True`s of
    `nonzero`)."""
    i, j, amplitude, variance, pixel_size = res
    n_directions = nonzero_tangents[0].shape[0]
    m_total, dtype = i.shape[0], i.dtype
    n_pixels = ny * nx
    primals_b = (
        i,
        j.astype(dtype),
        amplitude.astype(dtype),
        jnp.broadcast_to(variance, (m_total,)).astype(dtype),
        jnp.reshape(pixel_size, (1,)).astype(dtype),
        g.astype(dtype).reshape(-1),
    )
    names = ("i", "j", "amp", "var", "pix", "g")
    tangents_b = tangent_operands(
        nonzero_tangents,
        [p for p, nz_ in zip(primals_b, nonzero) if nz_],
        [n for n, nz_ in zip(names, nonzero) if nz_],
        dtype,
    )
    block_size = choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_vjp_tangent_kernel_2d(
        ny, nx, n_spread, use_erf, block_size, m_total, nonzero, n_directions
    )
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    image = pl.BlockSpec((n_pixels,), lambda p: (0,))
    per_direction = pl.BlockSpec((n_directions, block_size), lambda p: (0, p))
    tangent_specs = direction_specs(
        nonzero, names, block_size, n_directions, ("pix",), ("g",), n_pixels
    )
    odi, odj, odamplitude, odvariance_pp, odpixel_size_pp = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[per_point] * 4 + [scalar, image] + tangent_specs,
        out_specs=[per_direction] * 5,
        out_shape=[jax.ShapeDtypeStruct((n_directions, m_total), dtype)] * 5,
        compiler_params=CompilerParams(),
    )(*primals_b, *tangents_b)
    return (
        odi,
        odj,
        odamplitude,
        jnp.sum(odvariance_pp, axis=-1) if jnp.ndim(variance) == 0 else odvariance_pp,
        jnp.sum(odpixel_size_pp, axis=-1),
    )


@cache
def _spread_vjp_tangent_2d_batchable(ny, nx, n_spread, use_erf, nonzero):
    def impl(primals, tangents):
        res, g = primals[:5], primals[5]
        return _spread_vjp_tangent_2d_impl(
            res, g, tangents, nonzero, ny, nx, n_spread, use_erf
        )

    def single(*args):
        primals, tangents = args[:6], args[6:]
        return jax.tree.map(
            lambda t: t[0], impl(primals, tuple(t[None] for t in tangents))
        )

    return batchable(single, 6, impl)


def pallas_spread_vjp_tangent_2d(ny, nx, n_spread, use_erf, res, g, tres, tg):
    """The tangent of the 2D gather's five outputs; `tres`/`tg` may hold symbolic zeros,
    dropped at trace time. Under `jax.vmap` over the tangents, every direction runs in
    one launch."""
    tangents = (*tres, tg)
    nonzero = tuple(not is_zero(t) for t in tangents)
    fn = _spread_vjp_tangent_2d_batchable(ny, nx, n_spread, use_erf, nonzero)
    return fn(*res, g, *(t for t in tangents if not is_zero(t)))


def _spread_vjp_2d_jvp_rule(ny, nx, n_spread, use_erf, primals, tangents):
    res, g = primals
    tres, tg = tangents
    out = pallas_spread_vjp_2d(ny, nx, n_spread, use_erf, res, g)
    if is_zero(tg) and all(is_zero(t) for t in tres):
        return out, jax.tree.map(jnp.zeros_like, out)
    return out, pallas_spread_vjp_tangent_2d(ny, nx, n_spread, use_erf, res, g, tres, tg)


pallas_spread_vjp_2d.defjvp(_spread_vjp_2d_jvp_rule, symbolic_zeros=True)


# ── 3D ──────────────────────────────────────────────────────────────────────


@cache
def _make_spread_tangent_kernel_3d(
    nz: int,
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    nonzero: tuple[bool, bool, bool, bool, bool, bool],
    n_directions: int,
):
    """3D twin of `_make_spread_tangent_kernel_2d`."""
    nonzero_i, nonzero_j, nonzero_k, nonzero_amp, nonzero_var, nonzero_pix = nonzero
    nonzero_x = nonzero_i or nonzero_var or nonzero_pix
    nonzero_y = nonzero_j or nonzero_var or nonzero_pix
    nonzero_z = nonzero_k or nonzero_var or nonzero_pix
    n_nonzero, n_voxels = sum(nonzero), nz * ny * nx
    names = ("i", "j", "k", "amp", "var", "pix")

    def kernel(i_ref, j_ref, k_ref, amp_ref, var_ref, voxel_size_ref, *refs):
        tangent_refs, tout_ref = refs[:n_nonzero], refs[-1]
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        T = read_directions(tangent_refs, nonzero, names, n_directions)
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        def weight(z, coord_name, nonzero_axis):
            if not nonzero_axis:
                return _kernel_weight(z, variance, voxel_size, use_erf=use_erf), None
            w, dw_dz, dw_dvar, dw_dpix = _kernel_weight_and_grads(
                z, variance, voxel_size, use_erf=use_erf
            )
            return w, [
                _weight_tangent(
                    dw_dz, dw_dvar, dw_dpix, t[coord_name], t["var"], t["pix"]
                )
                for t in T
            ]

        def at(tangents, d):
            return None if tangents is None else tangents[d]

        def oz_body(oz, carry):
            idx_z = (i0z + oz) % nz
            wz, twz = weight((i0z + oz).astype(i.dtype) - k, "k", nonzero_z)

            def oy_body(oy, carry2):
                idx_y = (i0y + oy) % ny
                wy, twy = weight((i0y + oy).astype(i.dtype) - j, "j", nonzero_y)

                def ox_body(ox, carry3):
                    idx_x = (i0x + ox) % nx
                    wx, twx = weight((i0x + ox).astype(i.dtype) - i, "i", nonzero_x)
                    flat = idx_z * (nx * ny) + idx_y * nx + idx_x
                    for d in range(n_directions):
                        tval = add(
                            mul(T[d]["amp"], wz * wy * wx),
                            mul(
                                amp,
                                tprod3(wz, wy, wx, at(twz, d), at(twy, d), at(twx, d)),
                            ),
                        )
                        pltriton.atomic_add(
                            tout_ref, (flat + d * n_voxels,), tval, mask=valid
                        )
                    return carry3

                jax.lax.fori_loop(0, n_spread, ox_body, 0)
                return carry2

            jax.lax.fori_loop(0, n_spread, oy_body, 0)
            return carry

        jax.lax.fori_loop(0, n_spread, oz_body, 0)

    return kernel


@cache
def _make_spread_vjp_tangent_kernel_3d(
    nz: int,
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    nonzero: tuple[bool, bool, bool, bool, bool, bool, bool],
    n_directions: int,
):
    """3D twin of `_make_spread_vjp_tangent_kernel_2d`."""
    nonzero_i, nonzero_j, nonzero_k, nonzero_amp, nonzero_var, nonzero_pix, nonzero_g = (
        nonzero
    )
    nonzero_x = nonzero_i or nonzero_var or nonzero_pix
    nonzero_y = nonzero_j or nonzero_var or nonzero_pix
    nonzero_z = nonzero_k or nonzero_var or nonzero_pix
    n_nonzero, n_voxels = sum(nonzero), nz * ny * nx
    names = ("i", "j", "k", "amp", "var", "pix", "g")

    def weight_and_grads(z, variance, voxel_size):
        return _kernel_weight_and_grads(z, variance, voxel_size, use_erf=use_erf)

    def kernel(i_ref, j_ref, k_ref, amp_ref, var_ref, voxel_size_ref, g_ref, *refs):
        tangent_refs, out_refs = refs[:n_nonzero], refs[n_nonzero:]
        odi_ref, odj_ref, odk_ref, odamp_ref, odvar_ref, odpix_ref = out_refs
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        nonzero_names = [n for n, nz_ in zip(names, nonzero) if nz_]
        T = read_directions(
            [r for r, name in zip(tangent_refs, nonzero_names) if name != "g"],
            nonzero[:6],
            names[:6],
            n_directions,
        )
        tg_ref = tangent_refs[-1] if nonzero_g else None
        valid = valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def weights(z, coord_name, nonzero_axis):
            primal = weight_and_grads(z, variance, voxel_size)
            if not nonzero_axis:
                return primal, [(None,) * 4] * n_directions
            tangents = []
            for t in T:
                tz = None if t[coord_name] is None else -t[coord_name]
                tangents.append(
                    jvp_over_nonzero(
                        weight_and_grads,
                        (z, variance, voxel_size),
                        (tz, t["var"], t["pix"]),
                    )[1]
                )
            return primal, tangents

        def oz_body(oz, carry):
            idx_z = (i0z + oz) % nz
            (wz, dwz, vwz, hwz), Tz = weights(
                (i0z + oz).astype(i.dtype) - k, "k", nonzero_z
            )

            def oy_body(oy, carry2):
                idx_y = (i0y + oy) % ny
                (wy, dwy, vwy, hwy), Ty = weights(
                    (i0y + oy).astype(i.dtype) - j, "j", nonzero_y
                )

                def ox_body(ox, acc):
                    idx_x = (i0x + ox) % nx
                    (wx, dwx, vwx, hwx), Tx = weights(
                        (i0x + ox).astype(i.dtype) - i, "i", nonzero_x
                    )
                    flat = idx_z * (nx * ny) + idx_y * nx + idx_x
                    g_val = pltriton.load(g_ref.at[flat], mask=valid, other=0.0)

                    wzy, wzx, wyx = wz * wy, wz * wx, wy * wx
                    f_amp = wzy * wx
                    f_i, f_j, f_k = -dwx * wzy, -dwy * wzx, -dwz * wyx
                    f_var = wzy * vwx + wzx * vwy + wyx * vwz
                    f_pix = wzy * hwx + wzx * hwy + wyx * hwz

                    new_acc = []
                    for d in range(n_directions):
                        (Twx, Tdwx, Tvwx, Thwx) = Tx[d]
                        (Twy, Tdwy, Tvwy, Thwy) = Ty[d]
                        (Twz, Tdwz, Tvwz, Thwz) = Tz[d]
                        Tf_amp = tprod3(wz, wy, wx, Twz, Twy, Twx)
                        Tf_i = neg(tprod3(dwx, wz, wy, Tdwx, Twz, Twy))
                        Tf_j = neg(tprod3(dwy, wz, wx, Tdwy, Twz, Twx))
                        Tf_k = neg(tprod3(dwz, wy, wx, Tdwz, Twy, Twx))
                        Tf_var = add(
                            tprod3(wz, wy, vwx, Twz, Twy, Tvwx),
                            tprod3(wz, wx, vwy, Twz, Twx, Tvwy),
                            tprod3(wy, wx, vwz, Twy, Twx, Tvwz),
                        )
                        Tf_pix = add(
                            tprod3(wz, wy, hwx, Twz, Twy, Thwx),
                            tprod3(wz, wx, hwy, Twz, Twx, Thwy),
                            tprod3(wy, wx, hwz, Twy, Twx, Thwz),
                        )
                        tg_val = (
                            pltriton.load(
                                tg_ref.at[flat + d * n_voxels], mask=valid, other=0.0
                            )
                            if nonzero_g
                            else None
                        )
                        tamp = T[d]["amp"]

                        def tangent(f, Tf):
                            return add(
                                mul(tg_val, amp * f),
                                mul(g_val, add(mul(tamp, f), mul(amp, Tf))),
                            )

                        odamp, odi, odj, odk, odvar, odpix = acc[6 * d : 6 * d + 6]
                        new_acc += [
                            add(odamp, mul(tg_val, f_amp), mul(g_val, Tf_amp)),
                            add(odi, tangent(f_i, Tf_i)),
                            add(odj, tangent(f_j, Tf_j)),
                            add(odk, tangent(f_k, Tf_k)),
                            add(odvar, tangent(f_var, Tf_var)),
                            add(odpix, tangent(f_pix, Tf_pix)),
                        ]
                    return tuple(new_acc)

                return jax.lax.fori_loop(0, n_spread, ox_body, carry2)

            return jax.lax.fori_loop(0, n_spread, oy_body, carry)

        acc = jax.lax.fori_loop(0, n_spread, oz_body, (zero,) * (6 * n_directions))
        for d in range(n_directions):
            odamp, odi, odj, odk, odvar, odpix = acc[6 * d : 6 * d + 6]
            for ref, value in (
                (odi_ref, odi),
                (odj_ref, odj),
                (odk_ref, odk),
                (odamp_ref, odamp),
                (odvar_ref, odvar),
                (odpix_ref, odpix),
            ):
                pltriton.store(ref.at[d], value, mask=valid)

    return kernel


def _spread_tangent_3d_impl(
    primals, nonzero_tangents, nonzero, nz, ny, nx, n_spread, use_erf
):
    """3D twin of `_spread_tangent_2d_impl`: `(n_directions, nz, ny, nx)`."""
    i, j, k, amplitude, variance, voxel_size = primals
    n_directions = nonzero_tangents[0].shape[0]
    m_total, dtype = i.shape[0], i.dtype
    primals_b = (
        i,
        j.astype(dtype),
        k.astype(dtype),
        amplitude.astype(dtype),
        jnp.broadcast_to(variance, (m_total,)).astype(dtype),
        jnp.reshape(voxel_size, (1,)).astype(dtype),
    )
    names = ("i", "j", "k", "amp", "var", "pix")
    tangents_b = tangent_operands(
        nonzero_tangents,
        [p for p, nz_ in zip(primals_b, nonzero) if nz_],
        [n for n, nz_ in zip(names, nonzero) if nz_],
        dtype,
    )
    block_size = choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_tangent_kernel_3d(
        nz, ny, nx, n_spread, use_erf, block_size, m_total, nonzero, n_directions
    )
    n_voxels = nz * ny * nx
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    tvolume = pl.BlockSpec((n_directions * n_voxels,), lambda p: (0,))
    tangent_specs = direction_specs(
        nonzero, names, block_size, n_directions, ("pix",), (), n_voxels
    )
    tout = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[per_point] * 5 + [scalar] + tangent_specs + [tvolume],
        out_specs=tvolume,
        out_shape=jax.ShapeDtypeStruct((n_directions * n_voxels,), dtype),
        input_output_aliases={6 + len(tangents_b): 0},
        compiler_params=CompilerParams(),
    )(*primals_b, *tangents_b, jnp.zeros((n_directions * n_voxels,), dtype=dtype))
    return tout.reshape(n_directions, nz, ny, nx)


@cache
def _spread_tangent_3d_batchable(nz, ny, nx, n_spread, use_erf, nonzero):
    def impl(primals, tangents):
        return _spread_tangent_3d_impl(
            primals, tangents, nonzero, nz, ny, nx, n_spread, use_erf
        )

    def single(*args):
        primals, tangents = args[:6], args[6:]
        return impl(primals, tuple(t[None] for t in tangents))[0]

    return batchable(single, 6, impl)


def pallas_spread_tangent_3d(
    i, j, k, amplitude, variance, voxel_size, tangents, nz, ny, nx, n_spread, use_erf
):
    """3D twin of `pallas_spread_tangent_2d`."""
    nonzero = tuple(not is_zero(t) for t in tangents)
    fn = _spread_tangent_3d_batchable(nz, ny, nx, n_spread, use_erf, nonzero)
    return fn(
        i,
        j,
        k,
        amplitude,
        variance,
        voxel_size,
        *(t for t in tangents if not is_zero(t)),
    )


def _spread_3d_jvp_rule(nz, ny, nx, n_spread, use_erf, primals, tangents):
    out = pallas_spread_3d(*primals, nz, ny, nx, n_spread, use_erf)
    if all(is_zero(t) for t in tangents):
        return out, jnp.zeros_like(out)
    return out, pallas_spread_tangent_3d(
        *primals, tangents, nz, ny, nx, n_spread, use_erf
    )


pallas_spread_3d.defjvp(_spread_3d_jvp_rule, symbolic_zeros=True)


def _spread_vjp_tangent_3d_impl(
    res, g, nonzero_tangents, nonzero, nz, ny, nx, n_spread, use_erf
):
    """3D twin of `_spread_vjp_tangent_2d_impl`."""
    i, j, k, amplitude, variance, voxel_size = res
    n_directions = nonzero_tangents[0].shape[0]
    m_total, dtype = i.shape[0], i.dtype
    n_voxels = nz * ny * nx
    primals_b = (
        i,
        j.astype(dtype),
        k.astype(dtype),
        amplitude.astype(dtype),
        jnp.broadcast_to(variance, (m_total,)).astype(dtype),
        jnp.reshape(voxel_size, (1,)).astype(dtype),
        g.astype(dtype).reshape(-1),
    )
    names = ("i", "j", "k", "amp", "var", "pix", "g")
    tangents_b = tangent_operands(
        nonzero_tangents,
        [p for p, nz_ in zip(primals_b, nonzero) if nz_],
        [n for n, nz_ in zip(names, nonzero) if nz_],
        dtype,
    )
    block_size = choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_spread_vjp_tangent_kernel_3d(
        nz, ny, nx, n_spread, use_erf, block_size, m_total, nonzero, n_directions
    )
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    volume = pl.BlockSpec((n_voxels,), lambda p: (0,))
    per_direction = pl.BlockSpec((n_directions, block_size), lambda p: (0, p))
    tangent_specs = direction_specs(
        nonzero, names, block_size, n_directions, ("pix",), ("g",), n_voxels
    )
    odi, odj, odk, odamplitude, odvariance_pp, odvoxel_size_pp = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[per_point] * 5 + [scalar, volume] + tangent_specs,
        out_specs=[per_direction] * 6,
        out_shape=[jax.ShapeDtypeStruct((n_directions, m_total), dtype)] * 6,
        compiler_params=CompilerParams(),
    )(*primals_b, *tangents_b)
    return (
        odi,
        odj,
        odk,
        odamplitude,
        jnp.sum(odvariance_pp, axis=-1) if jnp.ndim(variance) == 0 else odvariance_pp,
        jnp.sum(odvoxel_size_pp, axis=-1),
    )


@cache
def _spread_vjp_tangent_3d_batchable(nz, ny, nx, n_spread, use_erf, nonzero):
    def impl(primals, tangents):
        res, g = primals[:6], primals[6]
        return _spread_vjp_tangent_3d_impl(
            res, g, tangents, nonzero, nz, ny, nx, n_spread, use_erf
        )

    def single(*args):
        primals, tangents = args[:7], args[7:]
        return jax.tree.map(
            lambda t: t[0], impl(primals, tuple(t[None] for t in tangents))
        )

    return batchable(single, 7, impl)


def pallas_spread_vjp_tangent_3d(nz, ny, nx, n_spread, use_erf, res, g, tres, tg):
    """3D twin of `pallas_spread_vjp_tangent_2d`."""
    tangents = (*tres, tg)
    nonzero = tuple(not is_zero(t) for t in tangents)
    fn = _spread_vjp_tangent_3d_batchable(nz, ny, nx, n_spread, use_erf, nonzero)
    return fn(*res, g, *(t for t in tangents if not is_zero(t)))


def _spread_vjp_3d_jvp_rule(nz, ny, nx, n_spread, use_erf, primals, tangents):
    res, g = primals
    tres, tg = tangents
    out = pallas_spread_vjp_3d(nz, ny, nx, n_spread, use_erf, res, g)
    if is_zero(tg) and all(is_zero(t) for t in tres):
        return out, jax.tree.map(jnp.zeros_like, out)
    return out, pallas_spread_vjp_tangent_3d(
        nz, ny, nx, n_spread, use_erf, res, g, tres, tg
    )


pallas_spread_vjp_3d.defjvp(_spread_vjp_3d_jvp_rule, symbolic_zeros=True)
