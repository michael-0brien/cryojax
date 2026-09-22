"""Pallas/Triton GPU kernel backend for Gaussian spreading, modeled on
FINUFFT's CUDA spreader (`gpu_method=1`, "nupts-driven": a grid-stride loop
over points, atomically scattering each point's kernel footprint straight
into the global output array). Unlike the pure-JAX backend in `spread.py`
(which materializes an `(M, n_spread^d)` buffer of per-point kernel weights
via `segment_sum`), this backend never holds more than `O(block_size *
n_spread^d)` at once per kernel program, for a total footprint of `O(M)`
across the whole call.

Forward (`pallas_spread_fwd_{2,3}d`) scatters, so it needs atomics (only
available on the Triton backend). Backward (`pallas_interp_bwd_{2,3}d`) is
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
import operator
from collections.abc import Mapping
from functools import cache, partial, reduce

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array

from ..._config import CRYOJAX_ENABLE_PALLAS, CRYOJAX_PALLAS_BLOCK_SIZE


# The Triton compiler-params class has been renamed across jax versions
# (`TritonCompilerParams` up to jax 0.5.3, `CompilerParams` from 0.9.1+) --
# isolate the version-compat shim to this one place.
_CompilerParams = getattr(pltriton, "CompilerParams", None) or getattr(
    pltriton, "TritonCompilerParams"
)


def _choose_block_size(n_spread: int, ndim: int) -> int:
    """Number of points each Pallas program handles.

    Empirically tuned across three GPU generations (Ampere RTX 3090, Hopper
    H100 PCIe, Blackwell RTX PRO 6000; both 2D and 3D -- see project memory
    for the full sweep) -- unlike the original per-instruction-budget
    formula this replaces, whose rationale (bounding Python-unrolled
    instruction count) no longer applies now that the kernel uses
    `jax.lax.fori_loop`. A flat `block_size=128` is the best overall choice
    across all three architectures and both dimensionalities. Override via
    the `CRYOJAX_PALLAS_BLOCK_SIZE` environment variable if you've
    benchmarked a better value for your own (GPU, M, n_spread) --
    deliberately not a function parameter, to keep the number of ways to
    configure this down to one.
    """
    del n_spread, ndim  # kept for API stability; the flat default doesn't use them
    if CRYOJAX_PALLAS_BLOCK_SIZE is not None:
        return CRYOJAX_PALLAS_BLOCK_SIZE
    return 128


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


def _kernel_weight_and_grad(
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


def _valid_lane_mask(block_size: int, m_total: int) -> Array:
    """Boolean mask marking which lanes in the current grid program
    correspond to real points vs. padding past `m_total` (when `m_total`
    isn't a multiple of `block_size`)."""
    prog = pl.program_id(0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (block_size,), 0)
    return (prog * block_size + lane) < m_total


# ============================================================================
# Forward: atomic-add scatter ("nupts-driven")
# ============================================================================
#
# TODO(pallas-output-driven): FINUFFT's own `gpu_method=0` ("auto") default
# picks `subprob` (`gpu_method=2`), not `nupts-driven`, for NUFFT type-1
# (spread) specifically. A bin-sorted, tile-local-accumulation follow-up
# ("Milestone B" / output-driven) was tried in this project's history and
# found conclusively worse than nupts-driven on every axis benchmarked --
# see project memory. Don't revisit without new information.


@cache
def _make_fwd_kernel_2d(
    ny: int, nx: int, n_spread: int, use_erf: bool, block_size: int, m_total: int
):
    def kernel(i_ref, j_ref, amp_ref, var_ref, pixel_size_ref, zeros_ref, out_ref):
        del zeros_ref  # aliased into out_ref; only used to force zero-init
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        valid = _valid_lane_mask(block_size, m_total)

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
def _make_fwd_kernel_3d(
    nz: int, ny: int, nx: int, n_spread: int, use_erf: bool, block_size: int, m_total: int
):
    def kernel(i_ref, j_ref, k_ref, amp_ref, var_ref, voxel_size_ref, zeros_ref, out_ref):
        del zeros_ref
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        valid = _valid_lane_mask(block_size, m_total)

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
def pallas_spread_fwd_2d(
    i, j, amplitude, variance, pixel_size, ny, nx, n_spread, use_erf
):
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    pixel_size_b = jnp.reshape(pixel_size, (1,)).astype(dtype)
    block_size = _choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_fwd_kernel_2d(ny, nx, n_spread, use_erf, block_size, m_total)
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
        compiler_params=_CompilerParams(),
    )(i, j, amplitude, variance_b, pixel_size_b, zeros)
    return out.reshape(ny, nx)


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10))
def pallas_spread_fwd_3d(
    i, j, k, amplitude, variance, voxel_size, nz, ny, nx, n_spread, use_erf
):
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    k = k.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    voxel_size_b = jnp.reshape(voxel_size, (1,)).astype(dtype)
    block_size = _choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_fwd_kernel_3d(nz, ny, nx, n_spread, use_erf, block_size, m_total)
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
        compiler_params=_CompilerParams(),
    )(i, j, k, amplitude, variance_b, voxel_size_b, zeros)
    return out.reshape(nz, ny, nx)


# ============================================================================
# Backward: pure gather ("interpolation", the adjoint of spreading) -- no
# atomics needed; see module docstring.
# ============================================================================


@cache
def _make_bwd_kernel_2d(
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
        valid = _valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def outer_body(oy, carry):
            damp, di, dj, dvar, dpix = carry
            idx_y = (i0y + oy) % ny
            z_y = (i0y + oy).astype(i.dtype) - j
            wy, dwy_dz, dwy_dvar, dwy_dpix = _kernel_weight_and_grad(
                z_y, variance, pixel_size, use_erf=use_erf
            )

            def inner_body(ox, carry2):
                damp, di, dj, dvar, dpix = carry2
                idx_x = (i0x + ox) % nx
                z_x = (i0x + ox).astype(i.dtype) - i
                wx, dwx_dz, dwx_dvar, dwx_dpix = _kernel_weight_and_grad(
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
def _make_bwd_kernel_3d(
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
        valid = _valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def oz_body(oz, carry):
            damp, di, dj, dk, dvar, dpix = carry
            idx_z = (i0z + oz) % nz
            z_z = (i0z + oz).astype(i.dtype) - k
            wz, dwz_dz, dwz_dvar, dwz_dpix = _kernel_weight_and_grad(
                z_z, variance, voxel_size, use_erf=use_erf
            )

            def oy_body(oy, carry2):
                damp, di, dj, dk, dvar, dpix = carry2
                idx_y = (i0y + oy) % ny
                z_y = (i0y + oy).astype(i.dtype) - j
                wy, dwy_dz, dwy_dvar, dwy_dpix = _kernel_weight_and_grad(
                    z_y, variance, voxel_size, use_erf=use_erf
                )

                def ox_body(ox, carry3):
                    damp, di, dj, dk, dvar, dpix = carry3
                    idx_x = (i0x + ox) % nx
                    z_x = (i0x + ox).astype(i.dtype) - i
                    wx, dwx_dz, dwx_dvar, dwx_dpix = _kernel_weight_and_grad(
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
def pallas_interp_bwd_2d(ny, nx, n_spread, use_erf, res, g):
    i, j, amplitude, variance, pixel_size = res
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    pixel_size_b = jnp.reshape(pixel_size, (1,)).astype(dtype)
    g = g.astype(dtype)
    block_size = _choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_bwd_kernel_2d(ny, nx, n_spread, use_erf, block_size, m_total)
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
        compiler_params=_CompilerParams(),
    )(i, j, amplitude, variance_b, pixel_size_b, g.reshape(-1))
    dvariance = jnp.sum(dvariance_pp) if jnp.ndim(variance) == 0 else dvariance_pp
    dpixel_size = jnp.sum(dpixel_size_pp)
    return di, dj, damplitude, dvariance, dpixel_size


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4))
def pallas_interp_bwd_3d(nz, ny, nx, n_spread, use_erf, res, g):
    i, j, k, amplitude, variance, voxel_size = res
    m_total, dtype = i.shape[0], i.dtype
    j = j.astype(dtype)
    k = k.astype(dtype)
    amplitude = amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    voxel_size_b = jnp.reshape(voxel_size, (1,)).astype(dtype)
    g = g.astype(dtype)
    block_size = _choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_bwd_kernel_3d(nz, ny, nx, n_spread, use_erf, block_size, m_total)
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
        compiler_params=_CompilerParams(),
    )(i, j, k, amplitude, variance_b, voxel_size_b, g.reshape(-1))
    dvariance = jnp.sum(dvariance_pp) if jnp.ndim(variance) == 0 else dvariance_pp
    dvoxel_size = jnp.sum(dvoxel_size_pp)
    return di, dj, dk, damplitude, dvariance, dvoxel_size


# ============================================================================
# `enable_pallas` resolution (the custom-VJP dispatch that uses this lives in
# `api.py`)
# ============================================================================


def resolve_enable_pallas(
    enable_pallas: bool | Mapping[str, bool] | None,
) -> tuple[bool, bool]:
    """Resolve `enable_pallas` (see `api.py`) to `(use_pallas_fwd,
    use_pallas_bwd)`, against the `CRYOJAX_ENABLE_PALLAS` env var default,
    and fail fast if Pallas is requested without a GPU available (rather
    than deferring to whatever error `pallas_call` itself would raise)."""
    if enable_pallas is None:
        use_fwd = use_bwd = CRYOJAX_ENABLE_PALLAS
    elif isinstance(enable_pallas, bool):
        use_fwd = use_bwd = enable_pallas
    elif isinstance(enable_pallas, Mapping):
        use_fwd = enable_pallas.get("fwd", CRYOJAX_ENABLE_PALLAS)
        use_bwd = enable_pallas.get("bwd", CRYOJAX_ENABLE_PALLAS)
    else:
        raise TypeError(
            "`enable_pallas` must be `None`, a `bool`, or a mapping with "
            f"'fwd'/'bwd' keys, got {type(enable_pallas)}."
        )
    if (use_fwd or use_bwd) and jax.default_backend() != "gpu":
        raise RuntimeError(
            "`enable_pallas` requires a CUDA GPU (the Pallas/Triton backend), "
            f"but the current JAX default backend is {jax.default_backend()!r}. "
            "Pass `enable_pallas=False` (or leave it unset, along with "
            "`CRYOJAX_ENABLE_PALLAS`) to use the pure-JAX backend instead."
        )
    return use_fwd, use_bwd


# ============================================================================
# Forward-mode (JVP) rules
# ============================================================================
#
# JAX cannot differentiate these kernels itself: the scatter aliases its zero-init
# buffer into the output, which `_pallas_call_jvp_rule` refuses outright, and the
# generic rule cannot trace a kernel whose block index maps read `program_id`. Each
# wrapper therefore carries a `jax.custom_jvp` whose tangent is a kernel of its own.
#
# Two things keep the rules cheap. They are FUSED: one launch writes the primal and
# the tangent, evaluating each separable weight (and its derivatives) once per
# neighbour instead of once per output. And they are SPECIALIZED on which tangents
# are symbolically zero (`symbolic_zeros=True`): a Hessian-vector product in an
# optimizer's shared-parameter direction is zero on almost every input, and a term
# multiplied by a zero loaded from memory still costs its weight evaluation, so the
# dead terms are dropped at trace time. The live pattern is a static key of the
# cached kernel factory. An operand whose tangent is dead is never read in the kernel,
# so the primal is passed in its slot rather than allocating a zeros array.


def _is_zero(tangent) -> bool:
    return isinstance(tangent, SymbolicZero)


def _add(*terms):
    """Sum of the non-`None` terms; `None` when there are none."""
    terms = [t for t in terms if t is not None]
    return reduce(operator.add, terms) if terms else None


def _mul(a, b):
    return None if a is None or b is None else a * b


def _jvp_live(fn, args, tangents):
    """`jax.jvp(fn, args, tangents)` over only the arguments whose tangent is not
    `None`; the rest are closed over as constants, so no dead term is traced."""
    live = [k for k, t in enumerate(tangents) if t is not None]

    def fn_live(*live_vals):
        full = list(args)
        for k, v in zip(live, live_vals):
            full[k] = v
        return fn(*full)

    return jax.jvp(
        fn_live, tuple(args[k] for k in live), tuple(tangents[k] for k in live)
    )


def _like_operand(tangent, operand, dtype):
    """A live tangent laid out like the kernel operand it pairs with: flattened if the
    operand is, broadcast if the operand is (a scalar broadcast per point)."""
    tangent = jnp.asarray(tangent, dtype=dtype)
    if tangent.ndim > operand.ndim:
        tangent = tangent.reshape(-1)
    return jnp.broadcast_to(tangent, operand.shape)


def _weight_tangent(dw_dz, dw_dvar, dw_dpix, tcoord, tvar, tpix):
    """Total differential of one separable weight. `z = index - coord`, so the
    coordinate tangent enters with `-dw/dz`."""
    return _add(
        None if tcoord is None else -dw_dz * tcoord,
        _mul(dw_dvar, tvar),
        _mul(dw_dpix, tpix),
    )


@cache
def _make_fwd_and_jvp_kernel_2d(
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    live: tuple[bool, bool, bool, bool, bool],
):
    """`_make_fwd_kernel_2d` and its tangent in one pass:
    `d(a wx wy) = da wx wy + a (twx wy + wx twy)`."""
    live_i, live_j, live_amp, live_var, live_pix = live
    live_x = live_i or live_var or live_pix
    live_y = live_j or live_var or live_pix

    def kernel(
        i_ref,
        j_ref,
        amp_ref,
        var_ref,
        pixel_size_ref,
        ti_ref,
        tj_ref,
        tamp_ref,
        tvar_ref,
        tpixel_size_ref,
        zeros_ref,
        tzeros_ref,
        out_ref,
        tout_ref,
    ):
        del zeros_ref, tzeros_ref  # aliased into the outputs; only force zero-init
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        ti = ti_ref[...] if live_i else None
        tj = tj_ref[...] if live_j else None
        tamp = tamp_ref[...] if live_amp else None
        tvar = tvar_ref[...] if live_var else None
        tpix = tpixel_size_ref[0] if live_pix else None
        valid = _valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        def weight(z, tcoord, is_live):
            if not is_live:
                return _kernel_weight(z, variance, pixel_size, use_erf=use_erf), None
            w, dw_dz, dw_dvar, dw_dpix = _kernel_weight_and_grad(
                z, variance, pixel_size, use_erf=use_erf
            )
            return w, _weight_tangent(dw_dz, dw_dvar, dw_dpix, tcoord, tvar, tpix)

        def outer_body(oy, carry):
            idx_y = (i0y + oy) % ny
            wy, twy = weight((i0y + oy).astype(i.dtype) - j, tj, live_y)

            def inner_body(ox, carry2):
                idx_x = (i0x + ox) % nx
                wx, twx = weight((i0x + ox).astype(i.dtype) - i, ti, live_x)
                flat = idx_y * nx + idx_x
                pltriton.atomic_add(out_ref, (flat,), amp * wx * wy, mask=valid)
                tval = _add(_mul(tamp, wx * wy), _mul(twx, amp * wy), _mul(twy, amp * wx))
                pltriton.atomic_add(tout_ref, (flat,), tval, mask=valid)
                return carry2

            jax.lax.fori_loop(0, n_spread, inner_body, 0)
            return carry

        jax.lax.fori_loop(0, n_spread, outer_body, 0)

    return kernel


@cache
def _make_bwd_and_jvp_kernel_2d(
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    live: tuple[bool, bool, bool, bool, bool, bool],
):
    """`_make_bwd_kernel_2d` and its tangent in one pass.

    The gather is linear in `g`, so `g`'s tangent is the gather of `tg` with the
    primal factors. Each residual tangent needs the total differential of every
    per-pixel factor -- second derivatives of the weight -- which `jax.jvp` of
    `_kernel_weight_and_grad` supplies inside the kernel body, forward mode over
    elementwise math, over only the live inputs."""
    live_i, live_j, live_amp, live_var, live_pix, live_g = live
    live_x = live_i or live_var or live_pix
    live_y = live_j or live_var or live_pix

    def weight_and_grad(z, variance, pixel_size):
        return _kernel_weight_and_grad(z, variance, pixel_size, use_erf=use_erf)

    def kernel(
        i_ref,
        j_ref,
        amp_ref,
        var_ref,
        pixel_size_ref,
        g_ref,
        ti_ref,
        tj_ref,
        tamp_ref,
        tvar_ref,
        tpixel_size_ref,
        tg_ref,
        di_ref,
        dj_ref,
        damp_ref,
        dvar_ref,
        dpix_ref,
        odi_ref,
        odj_ref,
        odamp_ref,
        odvar_ref,
        odpix_ref,
    ):
        i, j, amp, variance = i_ref[...], j_ref[...], amp_ref[...], var_ref[...]
        pixel_size = pixel_size_ref[0]
        ti = ti_ref[...] if live_i else None
        tj = tj_ref[...] if live_j else None
        tamp = tamp_ref[...] if live_amp else None
        tvar = tvar_ref[...] if live_var else None
        tpix = tpixel_size_ref[0] if live_pix else None
        valid = _valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def weights(z, tcoord, is_live):
            """`(w, dw/dz, dw/dvar, dw/dpix)` and their tangents (`None`s if dead)."""
            if not is_live:
                return weight_and_grad(z, variance, pixel_size), (None,) * 4
            tz = None if tcoord is None else -tcoord
            return _jvp_live(weight_and_grad, (z, variance, pixel_size), (tz, tvar, tpix))

        def outer_body(oy, carry):
            damp, di, dj, dvar, dpix, odamp, odi, odj, odvar, odpix = carry
            idx_y = (i0y + oy) % ny
            (wy, dwy, vwy, hwy), (Twy, Tdwy, Tvwy, Thwy) = weights(
                (i0y + oy).astype(i.dtype) - j, tj, live_y
            )

            def inner_body(ox, carry2):
                damp, di, dj, dvar, dpix, odamp, odi, odj, odvar, odpix = carry2
                idx_x = (i0x + ox) % nx
                (wx, dwx, vwx, hwx), (Twx, Tdwx, Tvwx, Thwx) = weights(
                    (i0x + ox).astype(i.dtype) - i, ti, live_x
                )
                flat = idx_y * nx + idx_x
                g_val = pltriton.load(g_ref.at[flat], mask=valid, other=0.0)

                # The gather's per-pixel factors...
                f_amp = wy * wx
                f_i = -dwx * wy
                f_j = -dwy * wx
                f_var = wy * vwx + wx * vwy
                f_pix = wy * hwx + wx * hwy
                damp = damp + g_val * f_amp
                di = di + amp * g_val * f_i
                dj = dj + amp * g_val * f_j
                dvar = dvar + amp * g_val * f_var
                dpix = dpix + amp * g_val * f_pix

                # ...and their tangents, by the product rule.
                Tf_amp = _add(_mul(Twy, wx), _mul(wy, Twx))
                Tf_i = _add(_mul(Tdwx, -wy), _mul(-dwx, Twy))
                Tf_j = _add(_mul(Tdwy, -wx), _mul(-dwy, Twx))
                Tf_var = _add(
                    _mul(Twy, vwx), _mul(wy, Tvwx), _mul(Twx, vwy), _mul(wx, Tvwy)
                )
                Tf_pix = _add(
                    _mul(Twy, hwx), _mul(wy, Thwx), _mul(Twx, hwy), _mul(wx, Thwy)
                )

                tg_val = (
                    pltriton.load(tg_ref.at[flat], mask=valid, other=0.0)
                    if live_g
                    else None
                )

                def tangent(f, Tf):
                    return _add(
                        _mul(tg_val, amp * f),
                        _mul(g_val, _add(_mul(tamp, f), _mul(amp, Tf))),
                    )

                odamp = _add(odamp, _mul(tg_val, f_amp), _mul(g_val, Tf_amp))
                odi = _add(odi, tangent(f_i, Tf_i))
                odj = _add(odj, tangent(f_j, Tf_j))
                odvar = _add(odvar, tangent(f_var, Tf_var))
                odpix = _add(odpix, tangent(f_pix, Tf_pix))
                return damp, di, dj, dvar, dpix, odamp, odi, odj, odvar, odpix

            return jax.lax.fori_loop(
                0,
                n_spread,
                inner_body,
                (damp, di, dj, dvar, dpix, odamp, odi, odj, odvar, odpix),
            )

        damp, di, dj, dvar, dpix, odamp, odi, odj, odvar, odpix = jax.lax.fori_loop(
            0, n_spread, outer_body, (zero,) * 10
        )

        pltriton.store(di_ref, di, mask=valid)
        pltriton.store(dj_ref, dj, mask=valid)
        pltriton.store(damp_ref, damp, mask=valid)
        pltriton.store(dvar_ref, dvar, mask=valid)
        pltriton.store(dpix_ref, dpix, mask=valid)
        pltriton.store(odi_ref, odi, mask=valid)
        pltriton.store(odj_ref, odj, mask=valid)
        pltriton.store(odamp_ref, odamp, mask=valid)
        pltriton.store(odvar_ref, odvar, mask=valid)
        pltriton.store(odpix_ref, odpix, mask=valid)

    return kernel


def pallas_spread_fwd_and_jvp_2d(
    i, j, amplitude, variance, pixel_size, tangents, ny, nx, n_spread, use_erf
):
    """`(out, tangent_out)`; `tangents` aligned with the five primals, symbolic zeros
    allowed and dropped at trace time."""
    live = tuple(not _is_zero(t) for t in tangents)
    m_total, dtype = i.shape[0], i.dtype
    j, amplitude = j.astype(dtype), amplitude.astype(dtype)
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    pixel_size_b = jnp.reshape(pixel_size, (1,)).astype(dtype)
    primals_b = (i, j, amplitude, variance_b, pixel_size_b)
    # A dead tangent's slot holds its primal: same shape, never read.
    tangents_b = tuple(
        (_like_operand(t, p, dtype) if is_live else p)
        for t, p, is_live in zip(tangents, primals_b, live)
    )
    block_size = _choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_fwd_and_jvp_kernel_2d(
        ny, nx, n_spread, use_erf, block_size, m_total, live
    )
    zeros = jnp.zeros((ny * nx,), dtype=dtype)
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    image = pl.BlockSpec((ny * nx,), lambda p: (0,))
    out, tout = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[per_point] * 4 + [scalar] + [per_point] * 4 + [scalar, image, image],
        out_specs=[image, image],
        out_shape=[jax.ShapeDtypeStruct((ny * nx,), dtype)] * 2,
        input_output_aliases={10: 0, 11: 1},
        compiler_params=_CompilerParams(),
    )(*primals_b, *tangents_b, zeros, zeros)
    return out.reshape(ny, nx), tout.reshape(ny, nx)


def _pallas_spread_fwd_2d_jvp(ny, nx, n_spread, use_erf, primals, tangents):
    if all(_is_zero(t) for t in tangents):
        out = pallas_spread_fwd_2d(*primals, ny, nx, n_spread, use_erf)
        return out, jnp.zeros_like(out)
    return pallas_spread_fwd_and_jvp_2d(*primals, tangents, ny, nx, n_spread, use_erf)


pallas_spread_fwd_2d.defjvp(_pallas_spread_fwd_2d_jvp, symbolic_zeros=True)


def pallas_interp_bwd_and_jvp_2d(ny, nx, n_spread, use_erf, res, g, tres, tg):
    """`(out, tangent_out)`, each the gather's five outputs; `tres`/`tg` may hold
    symbolic zeros, dropped at trace time."""
    live = tuple(not _is_zero(t) for t in (*tres, tg))
    i, j, amplitude, variance, pixel_size = res
    m_total, dtype = i.shape[0], i.dtype
    variance_b = jnp.broadcast_to(variance, (m_total,)).astype(dtype)
    primals_b = (
        i,
        j.astype(dtype),
        amplitude.astype(dtype),
        variance_b,
        jnp.reshape(pixel_size, (1,)).astype(dtype),
        g.astype(dtype).reshape(-1),
    )
    tangents_b = tuple(
        (_like_operand(t, p, dtype) if is_live else p)
        for t, p, is_live in zip((*tres, tg), primals_b, live)
    )
    block_size = _choose_block_size(n_spread, ndim=2)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_bwd_and_jvp_kernel_2d(
        ny, nx, n_spread, use_erf, block_size, m_total, live
    )
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    image = pl.BlockSpec((ny * nx,), lambda p: (0,))
    outs = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=([per_point] * 4 + [scalar, image]) * 2,
        out_specs=[per_point] * 10,
        out_shape=[jax.ShapeDtypeStruct((m_total,), dtype)] * 10,
        compiler_params=_CompilerParams(),
    )(*primals_b, *tangents_b)
    di, dj, damplitude, dvariance_pp, dpixel_size_pp = outs[:5]
    odi, odj, odamplitude, odvariance_pp, odpixel_size_pp = outs[5:]
    scalar_variance = jnp.ndim(variance) == 0
    out = (
        di,
        dj,
        damplitude,
        jnp.sum(dvariance_pp) if scalar_variance else dvariance_pp,
        jnp.sum(dpixel_size_pp),
    )
    tangent_out = (
        odi,
        odj,
        odamplitude,
        jnp.sum(odvariance_pp) if scalar_variance else odvariance_pp,
        jnp.sum(odpixel_size_pp),
    )
    return out, tangent_out


def _pallas_interp_bwd_2d_jvp(ny, nx, n_spread, use_erf, primals, tangents):
    res, g = primals
    tres, tg = tangents
    if _is_zero(tg) and all(_is_zero(t) for t in tres):
        out = pallas_interp_bwd_2d(ny, nx, n_spread, use_erf, res, g)
        return out, jax.tree.map(jnp.zeros_like, out)
    return pallas_interp_bwd_and_jvp_2d(ny, nx, n_spread, use_erf, res, g, tres, tg)


pallas_interp_bwd_2d.defjvp(_pallas_interp_bwd_2d_jvp, symbolic_zeros=True)


def _tprod3(a, b, c, ta, tb, tc):
    """Tangent of `a b c` by the product rule, `None` tangents dropped."""
    return _add(_mul(ta, b * c), _mul(tb, a * c), _mul(tc, a * b))


def _neg(x):
    return None if x is None else -x


@cache
def _make_fwd_and_jvp_kernel_3d(
    nz: int,
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    live: tuple[bool, bool, bool, bool, bool, bool],
):
    """`_make_fwd_kernel_3d` and its tangent in one pass."""
    live_i, live_j, live_k, live_amp, live_var, live_pix = live
    live_x = live_i or live_var or live_pix
    live_y = live_j or live_var or live_pix
    live_z = live_k or live_var or live_pix

    def kernel(
        i_ref,
        j_ref,
        k_ref,
        amp_ref,
        var_ref,
        voxel_size_ref,
        ti_ref,
        tj_ref,
        tk_ref,
        tamp_ref,
        tvar_ref,
        tvoxel_size_ref,
        zeros_ref,
        tzeros_ref,
        out_ref,
        tout_ref,
    ):
        del zeros_ref, tzeros_ref  # aliased into the outputs; only force zero-init
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        ti = ti_ref[...] if live_i else None
        tj = tj_ref[...] if live_j else None
        tk = tk_ref[...] if live_k else None
        tamp = tamp_ref[...] if live_amp else None
        tvar = tvar_ref[...] if live_var else None
        tpix = tvoxel_size_ref[0] if live_pix else None
        valid = _valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        def weight(z, tcoord, is_live):
            if not is_live:
                return _kernel_weight(z, variance, voxel_size, use_erf=use_erf), None
            w, dw_dz, dw_dvar, dw_dpix = _kernel_weight_and_grad(
                z, variance, voxel_size, use_erf=use_erf
            )
            return w, _weight_tangent(dw_dz, dw_dvar, dw_dpix, tcoord, tvar, tpix)

        def oz_body(oz, carry):
            idx_z = (i0z + oz) % nz
            wz, twz = weight((i0z + oz).astype(i.dtype) - k, tk, live_z)

            def oy_body(oy, carry2):
                idx_y = (i0y + oy) % ny
                wy, twy = weight((i0y + oy).astype(i.dtype) - j, tj, live_y)

                def ox_body(ox, carry3):
                    idx_x = (i0x + ox) % nx
                    wx, twx = weight((i0x + ox).astype(i.dtype) - i, ti, live_x)
                    flat = idx_z * (nx * ny) + idx_y * nx + idx_x
                    pltriton.atomic_add(out_ref, (flat,), amp * wz * wy * wx, mask=valid)
                    tval = _add(
                        _mul(tamp, wz * wy * wx),
                        _mul(amp, _tprod3(wz, wy, wx, twz, twy, twx)),
                    )
                    pltriton.atomic_add(tout_ref, (flat,), tval, mask=valid)
                    return carry3

                jax.lax.fori_loop(0, n_spread, ox_body, 0)
                return carry2

            jax.lax.fori_loop(0, n_spread, oy_body, 0)
            return carry

        jax.lax.fori_loop(0, n_spread, oz_body, 0)

    return kernel


@cache
def _make_bwd_and_jvp_kernel_3d(
    nz: int,
    ny: int,
    nx: int,
    n_spread: int,
    use_erf: bool,
    block_size: int,
    m_total: int,
    live: tuple[bool, bool, bool, bool, bool, bool, bool],
):
    """`_make_bwd_kernel_3d` and its tangent in one pass; see the 2D twin."""
    live_i, live_j, live_k, live_amp, live_var, live_pix, live_g = live
    live_x = live_i or live_var or live_pix
    live_y = live_j or live_var or live_pix
    live_z = live_k or live_var or live_pix

    def weight_and_grad(z, variance, voxel_size):
        return _kernel_weight_and_grad(z, variance, voxel_size, use_erf=use_erf)

    def kernel(
        i_ref,
        j_ref,
        k_ref,
        amp_ref,
        var_ref,
        voxel_size_ref,
        g_ref,
        ti_ref,
        tj_ref,
        tk_ref,
        tamp_ref,
        tvar_ref,
        tvoxel_size_ref,
        tg_ref,
        di_ref,
        dj_ref,
        dk_ref,
        damp_ref,
        dvar_ref,
        dpix_ref,
        odi_ref,
        odj_ref,
        odk_ref,
        odamp_ref,
        odvar_ref,
        odpix_ref,
    ):
        i, j, k = i_ref[...], j_ref[...], k_ref[...]
        amp, variance = amp_ref[...], var_ref[...]
        voxel_size = voxel_size_ref[0]
        ti = ti_ref[...] if live_i else None
        tj = tj_ref[...] if live_j else None
        tk = tk_ref[...] if live_k else None
        tamp = tamp_ref[...] if live_amp else None
        tvar = tvar_ref[...] if live_var else None
        tpix = tvoxel_size_ref[0] if live_pix else None
        valid = _valid_lane_mask(block_size, m_total)

        i0x = jnp.ceil(i - n_spread / 2.0).astype(jnp.int32)
        i0y = jnp.ceil(j - n_spread / 2.0).astype(jnp.int32)
        i0z = jnp.ceil(k - n_spread / 2.0).astype(jnp.int32)

        zero = jnp.zeros((block_size,), dtype=i.dtype)

        def weights(z, tcoord, is_live):
            if not is_live:
                return weight_and_grad(z, variance, voxel_size), (None,) * 4
            tz = None if tcoord is None else -tcoord
            return _jvp_live(weight_and_grad, (z, variance, voxel_size), (tz, tvar, tpix))

        def oz_body(oz, carry):
            idx_z = (i0z + oz) % nz
            (wz, dwz, vwz, hwz), (Twz, Tdwz, Tvwz, Thwz) = weights(
                (i0z + oz).astype(i.dtype) - k, tk, live_z
            )

            def oy_body(oy, carry2):
                idx_y = (i0y + oy) % ny
                (wy, dwy, vwy, hwy), (Twy, Tdwy, Tvwy, Thwy) = weights(
                    (i0y + oy).astype(i.dtype) - j, tj, live_y
                )

                def ox_body(ox, carry3):
                    damp, di, dj, dk, dvar, dpix, odamp, odi, odj, odk, odvar, odpix = (
                        carry3
                    )
                    idx_x = (i0x + ox) % nx
                    (wx, dwx, vwx, hwx), (Twx, Tdwx, Tvwx, Thwx) = weights(
                        (i0x + ox).astype(i.dtype) - i, ti, live_x
                    )
                    flat = idx_z * (nx * ny) + idx_y * nx + idx_x
                    g_val = pltriton.load(g_ref.at[flat], mask=valid, other=0.0)

                    wzy, wzx, wyx = wz * wy, wz * wx, wy * wx
                    f_amp = wzy * wx
                    f_i, f_j, f_k = -dwx * wzy, -dwy * wzx, -dwz * wyx
                    f_var = wzy * vwx + wzx * vwy + wyx * vwz
                    f_pix = wzy * hwx + wzx * hwy + wyx * hwz
                    damp = damp + g_val * f_amp
                    di = di + amp * g_val * f_i
                    dj = dj + amp * g_val * f_j
                    dk = dk + amp * g_val * f_k
                    dvar = dvar + amp * g_val * f_var
                    dpix = dpix + amp * g_val * f_pix

                    Tf_amp = _tprod3(wz, wy, wx, Twz, Twy, Twx)
                    Tf_i = _neg(_tprod3(dwx, wz, wy, Tdwx, Twz, Twy))
                    Tf_j = _neg(_tprod3(dwy, wz, wx, Tdwy, Twz, Twx))
                    Tf_k = _neg(_tprod3(dwz, wy, wx, Tdwz, Twy, Twx))
                    Tf_var = _add(
                        _tprod3(wz, wy, vwx, Twz, Twy, Tvwx),
                        _tprod3(wz, wx, vwy, Twz, Twx, Tvwy),
                        _tprod3(wy, wx, vwz, Twy, Twx, Tvwz),
                    )
                    Tf_pix = _add(
                        _tprod3(wz, wy, hwx, Twz, Twy, Thwx),
                        _tprod3(wz, wx, hwy, Twz, Twx, Thwy),
                        _tprod3(wy, wx, hwz, Twy, Twx, Thwz),
                    )

                    tg_val = (
                        pltriton.load(tg_ref.at[flat], mask=valid, other=0.0)
                        if live_g
                        else None
                    )

                    def tangent(f, Tf):
                        return _add(
                            _mul(tg_val, amp * f),
                            _mul(g_val, _add(_mul(tamp, f), _mul(amp, Tf))),
                        )

                    odamp = _add(odamp, _mul(tg_val, f_amp), _mul(g_val, Tf_amp))
                    odi = _add(odi, tangent(f_i, Tf_i))
                    odj = _add(odj, tangent(f_j, Tf_j))
                    odk = _add(odk, tangent(f_k, Tf_k))
                    odvar = _add(odvar, tangent(f_var, Tf_var))
                    odpix = _add(odpix, tangent(f_pix, Tf_pix))
                    return (
                        damp,
                        di,
                        dj,
                        dk,
                        dvar,
                        dpix,
                        odamp,
                        odi,
                        odj,
                        odk,
                        odvar,
                        odpix,
                    )

                return jax.lax.fori_loop(0, n_spread, ox_body, carry2)

            return jax.lax.fori_loop(0, n_spread, oy_body, carry)

        outs = jax.lax.fori_loop(0, n_spread, oz_body, (zero,) * 12)
        damp, di, dj, dk, dvar, dpix, odamp, odi, odj, odk, odvar, odpix = outs

        for ref, value in (
            (di_ref, di),
            (dj_ref, dj),
            (dk_ref, dk),
            (damp_ref, damp),
            (dvar_ref, dvar),
            (dpix_ref, dpix),
            (odi_ref, odi),
            (odj_ref, odj),
            (odk_ref, odk),
            (odamp_ref, odamp),
            (odvar_ref, odvar),
            (odpix_ref, odpix),
        ):
            pltriton.store(ref, value, mask=valid)

    return kernel


def pallas_spread_fwd_and_jvp_3d(
    i, j, k, amplitude, variance, voxel_size, tangents, nz, ny, nx, n_spread, use_erf
):
    live = tuple(not _is_zero(t) for t in tangents)
    m_total, dtype = i.shape[0], i.dtype
    primals_b = (
        i,
        j.astype(dtype),
        k.astype(dtype),
        amplitude.astype(dtype),
        jnp.broadcast_to(variance, (m_total,)).astype(dtype),
        jnp.reshape(voxel_size, (1,)).astype(dtype),
    )
    tangents_b = tuple(
        (_like_operand(t, p, dtype) if is_live else p)
        for t, p, is_live in zip(tangents, primals_b, live)
    )
    block_size = _choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_fwd_and_jvp_kernel_3d(
        nz, ny, nx, n_spread, use_erf, block_size, m_total, live
    )
    n_voxels = nz * ny * nx
    zeros = jnp.zeros((n_voxels,), dtype=dtype)
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    volume = pl.BlockSpec((n_voxels,), lambda p: (0,))
    out, tout = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=([per_point] * 5 + [scalar]) * 2 + [volume, volume],
        out_specs=[volume, volume],
        out_shape=[jax.ShapeDtypeStruct((n_voxels,), dtype)] * 2,
        input_output_aliases={12: 0, 13: 1},
        compiler_params=_CompilerParams(),
    )(*primals_b, *tangents_b, zeros, zeros)
    return out.reshape(nz, ny, nx), tout.reshape(nz, ny, nx)


def _pallas_spread_fwd_3d_jvp(nz, ny, nx, n_spread, use_erf, primals, tangents):
    if all(_is_zero(t) for t in tangents):
        out = pallas_spread_fwd_3d(*primals, nz, ny, nx, n_spread, use_erf)
        return out, jnp.zeros_like(out)
    return pallas_spread_fwd_and_jvp_3d(*primals, tangents, nz, ny, nx, n_spread, use_erf)


pallas_spread_fwd_3d.defjvp(_pallas_spread_fwd_3d_jvp, symbolic_zeros=True)


def pallas_interp_bwd_and_jvp_3d(nz, ny, nx, n_spread, use_erf, res, g, tres, tg):
    live = tuple(not _is_zero(t) for t in (*tres, tg))
    i, j, k, amplitude, variance, voxel_size = res
    m_total, dtype = i.shape[0], i.dtype
    primals_b = (
        i,
        j.astype(dtype),
        k.astype(dtype),
        amplitude.astype(dtype),
        jnp.broadcast_to(variance, (m_total,)).astype(dtype),
        jnp.reshape(voxel_size, (1,)).astype(dtype),
        g.astype(dtype).reshape(-1),
    )
    tangents_b = tuple(
        (_like_operand(t, p, dtype) if is_live else p)
        for t, p, is_live in zip((*tres, tg), primals_b, live)
    )
    block_size = _choose_block_size(n_spread, ndim=3)
    grid = (pl.cdiv(m_total, block_size),)
    kernel = _make_bwd_and_jvp_kernel_3d(
        nz, ny, nx, n_spread, use_erf, block_size, m_total, live
    )
    per_point = pl.BlockSpec((block_size,), lambda p: (p,))
    scalar = pl.BlockSpec((1,), lambda p: (0,))
    volume = pl.BlockSpec((nz * ny * nx,), lambda p: (0,))
    outs = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=([per_point] * 5 + [scalar, volume]) * 2,
        out_specs=[per_point] * 12,
        out_shape=[jax.ShapeDtypeStruct((m_total,), dtype)] * 12,
        compiler_params=_CompilerParams(),
    )(*primals_b, *tangents_b)
    scalar_variance = jnp.ndim(variance) == 0

    def assemble(di, dj, dk, damplitude, dvariance_pp, dvoxel_size_pp):
        return (
            di,
            dj,
            dk,
            damplitude,
            jnp.sum(dvariance_pp) if scalar_variance else dvariance_pp,
            jnp.sum(dvoxel_size_pp),
        )

    return assemble(*outs[:6]), assemble(*outs[6:])


def _pallas_interp_bwd_3d_jvp(nz, ny, nx, n_spread, use_erf, primals, tangents):
    res, g = primals
    tres, tg = tangents
    if _is_zero(tg) and all(_is_zero(t) for t in tres):
        out = pallas_interp_bwd_3d(nz, ny, nx, n_spread, use_erf, res, g)
        return out, jax.tree.map(jnp.zeros_like, out)
    return pallas_interp_bwd_and_jvp_3d(nz, ny, nx, n_spread, use_erf, res, g, tres, tg)


pallas_interp_bwd_3d.defjvp(_pallas_interp_bwd_3d_jvp, symbolic_zeros=True)
