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
from collections.abc import Mapping
from functools import cache

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
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
