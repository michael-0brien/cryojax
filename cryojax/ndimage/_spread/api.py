"""Public API for Gaussian spreading: scatters point strengths onto a
uniform grid, dispatching between the pure-JAX backend (`spread.py`) and the
Pallas/Triton GPU backend (`pallas_spread.py`) per-call via `enable_pallas`.
"""

import math
from collections.abc import Mapping

import numpy as np
from jaxtyping import Array, Float

from ...jax_util import FloatLike, NDArrayLike
from .pallas_spread import (
    _resolve_enable_pallas,
    _spread_2d_dispatch,
    _spread_3d_dispatch,
)


def spread_gaussians_2d(
    x: Float[Array, " M"],
    y: Float[Array, " M"],
    amplitude: Float[Array, " M"],
    variance: Float[Array, ""] | Float[Array, " M"],
    shape: tuple[int, int],
    *,
    pixel_size: Float[Array, ""],
    n_spread: int = 7,
    use_erf: bool = True,
    enable_pallas: bool | Mapping[str, bool] | None = None,
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
    - `enable_pallas`:
        Whether to use the Pallas/Triton GPU kernel backend instead of the
        pure-JAX (`segment_sum`-based) backend, for the forward and backward
        pass independently. Requires a CUDA GPU; raises if requested
        without one.

        There is no single best choice for every case -- extensive
        benchmarking found:

        - The pure-JAX **forward** pass usually wins outright (the Pallas
          forward kernel needs an atomic scatter, which contends under
          realistic atom density). Leave `"fwd"` `False` (the default) in
          most cases.
        - The Pallas **backward** pass is a real, consistent win in both
          memory (~10-37x less) and speed (1.25x-8.3x faster) than the
          pure-JAX analytic backward, because it's a pure gather with no
          atomic contention. `{"bwd": True}` -- pure-JAX forward, Pallas
          backward -- is the sensible starting point if you want to opt
          into anything.
        - `enable_pallas=True` (both directions) trades some speed for a
          flat, `M`-independent memory profile that also survives
          `jax.vmap`-ing this computation over a batch of particles
          (pure-JAX's memory scales with batch size; Pallas's doesn't) --
          worth it specifically when even `{"bwd": True}`'s memory isn't
          low enough, e.g. multi-particle refinement with many particles
          vmapped together.
        - The right choice is also hardware- and scale-dependent (e.g. on
          Hopper, plain pure-JAX can outperform `{"bwd": True}` at moderate
          `M`, with the crossover shifting by architecture) -- benchmark
          your own workload if this matters.

        `True`/`False` applies to both the forward and backward pass; a
        dict with `"fwd"`/`"bwd"` keys sets them independently (e.g.
        `{"fwd": True}` uses Pallas only for the forward pass). `None`
        (default) defers to the `CRYOJAX_ENABLE_PALLAS` environment
        variable (`False` if unset). The number of points each Pallas grid
        program handles is not configurable here; it defaults to a flat
        128 (empirically the best overall choice across Ampere, Hopper,
        and Blackwell, in both 2D and 3D), overridable only via the
        `CRYOJAX_PALLAS_BLOCK_SIZE` environment variable if you've
        benchmarked a better value for your own (GPU, `M`, `n_spread`).

    **Returns:**

    The grid of shape `(ny, nx)` with gaussians scattered onto it.
    """  # noqa: E501
    _check_n_spread(n_spread, shape)
    use_pallas_fwd, use_pallas_bwd = _resolve_enable_pallas(enable_pallas)
    ny, nx = shape
    i = _normalize_coord_to_grid(x, nx, pixel_size)
    j = _normalize_coord_to_grid(y, ny, pixel_size)
    return _spread_2d_dispatch(
        i,
        j,
        amplitude,
        variance,
        pixel_size,
        ny,
        nx,
        n_spread,
        use_erf,
        use_pallas_fwd,
        use_pallas_bwd,
    )


def spread_gaussians_3d(
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
    enable_pallas: bool | Mapping[str, bool] | None = None,
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
    - `enable_pallas`:
        Whether to use the Pallas/Triton GPU kernel backend instead of the
        pure-JAX (`segment_sum`-based) backend, for the forward and backward
        pass independently. Requires a CUDA GPU; raises if requested
        without one.

        There is no single best choice for every case -- extensive
        benchmarking found:

        - The pure-JAX **forward** pass usually wins outright (the Pallas
          forward kernel needs an atomic scatter, which contends under
          realistic atom density). Leave `"fwd"` `False` (the default) in
          most cases.
        - The Pallas **backward** pass is a real, consistent win in both
          memory (~10-37x less) and speed (1.25x-8.3x faster) than the
          pure-JAX analytic backward, because it's a pure gather with no
          atomic contention. `{"bwd": True}` -- pure-JAX forward, Pallas
          backward -- is the sensible starting point if you want to opt
          into anything.
        - `enable_pallas=True` (both directions) trades some speed for a
          flat, `M`-independent memory profile that also survives
          `jax.vmap`-ing this computation over a batch of particles
          (pure-JAX's memory scales with batch size; Pallas's doesn't) --
          worth it specifically when even `{"bwd": True}`'s memory isn't
          low enough, e.g. multi-particle refinement with many particles
          vmapped together.
        - The right choice is also hardware- and scale-dependent (e.g. on
          Hopper, plain pure-JAX can outperform `{"bwd": True}` at moderate
          `M`, with the crossover shifting by architecture) -- benchmark
          your own workload if this matters.

        `True`/`False` applies to both the forward and backward pass; a
        dict with `"fwd"`/`"bwd"` keys sets them independently (e.g.
        `{"fwd": True}` uses Pallas only for the forward pass). `None`
        (default) defers to the `CRYOJAX_ENABLE_PALLAS` environment
        variable (`False` if unset). The number of points each Pallas grid
        program handles is not configurable here; it defaults to a flat
        128 (empirically the best overall choice across Ampere, Hopper,
        and Blackwell, in both 2D and 3D), overridable only via the
        `CRYOJAX_PALLAS_BLOCK_SIZE` environment variable if you've
        benchmarked a better value for your own (GPU, `M`, `n_spread`).

    **Returns:**

    The grid of shape `(nz, ny, nx)` with gaussians scattered onto it.
    """  # noqa: E501
    _check_n_spread(n_spread, shape)
    use_pallas_fwd, use_pallas_bwd = _resolve_enable_pallas(enable_pallas)
    nz, ny, nx = shape
    i = _normalize_coord_to_grid(x, nx, voxel_size)
    j = _normalize_coord_to_grid(y, ny, voxel_size)
    k = _normalize_coord_to_grid(z, nz, voxel_size)
    return _spread_3d_dispatch(
        i,
        j,
        k,
        amplitude,
        variance,
        voxel_size,
        nz,
        ny,
        nx,
        n_spread,
        use_erf,
        use_pallas_fwd,
        use_pallas_bwd,
    )


def variance_to_nspread(
    variance: FloatLike | Float[NDArrayLike, " M"],
    pixel_size: FloatLike,
    n_sigma: float = 4.0,
) -> int:
    """Choose an `n_spread` sufficient to truncate the Gaussian kernel used by
    [`cryojax.ndimage.spread_gaussians_2d`][]/[`cryojax.ndimage.spread_gaussians_3d`][]
    at `n_sigma` standard deviations.

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
        used) that will be passed to `spread_gaussians_2d`/`spread_gaussians_3d`.
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
# Private helpers
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
