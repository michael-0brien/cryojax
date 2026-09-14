"""Benchmark `cryojax.ndimage.spread_gaussians_2d`'s `enable_pallas` combos
against the pure-JAX default, at realistic atom density.

Compares three configurations across a sweep of `M` (atom count) and
`n_spread`:

  - `enable_pallas=False`      -- pure-JAX forward + pure-JAX backward (default)
  - `enable_pallas={"bwd": True}` -- pure-JAX forward + Pallas backward ("mixed")
  - `enable_pallas=True`       -- Pallas forward + Pallas backward ("full")

For each cell, times three things: the forward pass alone, the backward pass
alone (isolated via `jax.vjp`'s pullback, not bundled with the forward-value
computation), and the full `value_and_grad` pipeline (what a real loss
evaluation actually runs). See project history/memory for the full
cross-hardware investigation this distills; briefly: pure-JAX forward
usually wins outright (Pallas forward needs an atomic scatter, which
contends under realistic density), Pallas backward is a real win in both
memory and speed (a pure gather, no contention), and "mixed" is generally
the best full-pipeline choice, though the right choice is hardware- and
scale-dependent -- there is no universal winner, which is exactly why
`enable_pallas` supports forward/backward independently rather than a
single on/off switch.

Positions are synthetic but density-calibrated (~2 atoms/px, matching real
PDB structures projected to 2D -- see project memory), not uniform-random
over the whole grid, which was found earlier in this project to overstate
Pallas's advantage.

Requires a real CUDA GPU (the Pallas/Triton backend); run with
`~/venvs/cryojax-env/bin/python benchmarks/benchmark_pallas_spread.py` (or
whatever environment has a CUDA-enabled `jaxlib` installed) -- the repo's
own `.venv` is CPU-only and cannot run this.
"""

import math
from datetime import UTC, datetime
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cryojax.ndimage import spread_gaussians_2d


assert jax.default_backend() == "gpu", (
    "This benchmark requires a real CUDA GPU (the Pallas/Triton backend); "
    f"the current JAX default backend is {jax.default_backend()!r}."
)

COMBOS = {
    "pure_jax": False,
    "mixed": {"bwd": True},
    "full_pallas": True,
}
DENSITY = 2.0  # atoms/px, real-PDB-calibrated (see project memory)
PIXEL_SIZE = 1.0


def _make_points(key, m, grid_dim, pixel_size):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    half_extent = (grid_dim - 20) / 2 * pixel_size
    x = jax.random.uniform(k1, (m,), minval=-half_extent, maxval=half_extent)
    y = jax.random.uniform(k2, (m,), minval=-half_extent, maxval=half_extent)
    amplitude = jax.random.uniform(k3, (m,), minval=0.5, maxval=1.5)
    variance = jax.random.uniform(k4, (m,), minval=0.3, maxval=0.8)
    return x, y, amplitude, variance


def _time_call(fn, *args, n_repeats=5):
    """Median wall-clock time over `n_repeats` calls, after one warmup call
    (to absorb JIT compilation)."""
    out = jax.block_until_ready(fn(*args))
    del out
    times = []
    for _ in range(n_repeats):
        start = perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(perf_counter() - start)
        del out
    return float(np.median(times))


def _time_cell(m, n_spread, grid_dim, enable_pallas, x, y, amplitude, variance):
    """Time forward-only, backward-only (isolated), and the full
    value_and_grad pipeline, for one (M, n_spread, enable_pallas) cell."""
    shape = (grid_dim, grid_dim)

    def forward(x, y, amplitude, variance):
        return spread_gaussians_2d(
            x,
            y,
            amplitude,
            variance,
            shape,
            pixel_size=PIXEL_SIZE,
            n_spread=n_spread,
            enable_pallas=enable_pallas,
        )

    def loss(x, y, amplitude, variance):
        return jnp.sum(forward(x, y, amplitude, variance) ** 2)

    fwd_jit = jax.jit(forward)
    t_fwd = _time_call(fwd_jit, x, y, amplitude, variance)

    # Isolate the backward pass alone via `jax.vjp`'s pullback, rather than
    # bundling it with the forward-value computation (which value_and_grad
    # does, and is timed separately below).
    _, vjp_fn = jax.vjp(jax.jit(loss), x, y, amplitude, variance)
    vjp_jit = jax.jit(vjp_fn)
    g = jnp.asarray(1.0)
    t_bwd = _time_call(vjp_jit, g)

    value_and_grad_jit = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
    t_full = _time_call(value_and_grad_jit, x, y, amplitude, variance)

    return t_fwd, t_bwd, t_full


def run_sweep(m_values, n_spread_values):
    rows = []
    key = jax.random.PRNGKey(0)
    for n_spread in n_spread_values:
        for m in m_values:
            grid_dim = int(round(math.sqrt(m / DENSITY)))
            print(f"--- n_spread={n_spread}, M={m:.0e}, grid={grid_dim}x{grid_dim} ---")
            x, y, amplitude, variance = _make_points(
                jax.random.fold_in(key, m * 100 + n_spread), m, grid_dim, PIXEL_SIZE
            )

            row = {"n_spread": n_spread, "m": m, "grid_dim": grid_dim}
            for label, enable_pallas in COMBOS.items():
                try:
                    t_fwd, t_bwd, t_full = _time_cell(
                        m, n_spread, grid_dim, enable_pallas, x, y, amplitude, variance
                    )
                    print(
                        f"  {label:12s}: fwd={t_fwd * 1e3:8.2f} ms "
                        f" bwd={t_bwd * 1e3:8.2f} ms  "
                        f"full={t_full * 1e3:8.2f} ms"
                    )
                    row[f"{label}_fwd_ms"] = t_fwd * 1e3
                    row[f"{label}_bwd_ms"] = t_bwd * 1e3
                    row[f"{label}_full_ms"] = t_full * 1e3
                except Exception as e:  # noqa: BLE001 -- record OOM/etc, keep sweeping
                    print(f"  {label:12s}: FAILED ({type(e).__name__}: {e})")
                    row[f"{label}_fwd_ms"] = float("nan")
                    row[f"{label}_bwd_ms"] = float("nan")
                    row[f"{label}_full_ms"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, out_path: str):
    n_spread_values = sorted(df["n_spread"].unique())
    passes = ["fwd", "bwd", "full"]
    pass_titles = {
        "fwd": "Forward only",
        "bwd": "Backward only",
        "full": "value_and_grad",
    }
    colors = {"pure_jax": "tab:blue", "mixed": "tab:orange", "full_pallas": "tab:green"}

    fig, axes = plt.subplots(
        len(n_spread_values),
        len(passes),
        figsize=(5 * len(passes), 4.5 * len(n_spread_values)),
        squeeze=False,
    )
    for i, n_spread in enumerate(n_spread_values):
        sub = df[df["n_spread"] == n_spread].sort_values("m")
        for j, pass_name in enumerate(passes):
            ax = axes[i][j]
            for label in COMBOS:
                ax.loglog(
                    sub["m"],
                    sub[f"{label}_{pass_name}_ms"],
                    "o-",
                    label=label,
                    color=colors[label],
                )
            ax.set_xlabel("Number of atoms $M$")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"n_spread={n_spread}, {pass_titles[pass_name]}")
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    m_values = [10**5, 10**6, 10**7]
    n_spread_values = [7, 17]

    df = run_sweep(m_values, n_spread_values)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = f"benchmarks/benchmark_pallas_spread_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    plot_results(df, f"benchmarks/benchmark_pallas_spread_{timestamp}.png")
