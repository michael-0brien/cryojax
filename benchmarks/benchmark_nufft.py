"""Benchmark the native NUFFT and `GaussianMixtureProjection` on chain-ordered atoms.

One configuration per process, so that peak memory is its own; results are appended to
`--out` as JSON lines. Points are cached as `.npy` in `--cache-dir`, so that the cufinufft
and CPU finufft references load the same ones; `--prepare` only writes them. The achieved
error is against `<cache-dir>/pts_*_ref.npy`, if present: the type-1 rfft half grid from
CPU finufft in double at eps 1e-12, from a generator outside this script.

```
python benchmark_nufft.py --backend pallas --ndim 2 --n 4096 --M 4000000 \
    --eps 1e-4 --sigma 1.25 --cache-dir scratch --out results.jsonl
```

Backends: `jax` and `pallas` (native, both passes), `pallas_bwd` (native, backward pass
only), `gmp` and `gmp_bwd` (`GaussianMixtureProjection` point mode, pure JAX or Pallas
backward pass, at `upsample_factor=2` with `tolerance=eps`).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cryojax.simulator as cxs  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cryojax.ndimage import dispatch_nufft1  # noqa: E402
from cryojax.ndimage._nufft.api import _fine_coordinates, _spread_dispatch  # noqa: E402
from cryojax.ndimage._nufft.kernel import fine_grid_size, make_pswf_kernel  # noqa: E402


PIXEL_SIZE = 0.83
ATOMS_PER_BLOB = 7000
BLOB_HALF_WIDTH = 20.0
STEP = 1.5
ATOM_VARIANCE = 0.5
PALLAS = {"jax": False, "pallas": True, "pallas_bwd": {"bwd": True}}
GMP_PALLAS = {"gmp": False, "gmp_bwd": {"bwd": True}}


def make_points(ndim, n, M, order, seed=0):
    """Chain-ordered atoms: random walks of 1.5 A steps folded into 40 A blobs of 7,000
    atoms, blobs tiled along three tilted 300 px bands (2D) or at random (3D); `order`
    `"random"` permutes the atoms. Returns float32 positions `(M, ndim)` in angstroms,
    as `(x, y[, z])`, and standard normal strengths."""
    rng = np.random.default_rng(seed)
    num_blobs = math.ceil(M / ATOMS_PER_BLOB)
    length = n * PIXEL_SIZE
    h = BLOB_HALF_WIDTH
    steps = rng.standard_normal((num_blobs, ATOMS_PER_BLOB, ndim))
    steps *= STEP / np.linalg.norm(steps, axis=-1, keepdims=True)
    walk = np.cumsum(steps, axis=1)
    walk = h - np.abs((walk + h) % (4 * h) - 2 * h)
    if ndim == 2:
        band = np.minimum(np.arange(num_blobs) * 3 // num_blobs, 2)
        along = np.concatenate(
            [np.sort(rng.uniform(-0.4, 0.4, (band == b).sum())) for b in range(3)]
        )
        across = (band - 1) * length / 4 + rng.uniform(
            -150 * PIXEL_SIZE, 150 * PIXEL_SIZE, num_blobs
        )
        a = 0.3
        centers = np.stack(
            [
                along * length * np.cos(a) - across * np.sin(a),
                along * length * np.sin(a) + across * np.cos(a),
            ],
            axis=-1,
        )
    else:
        centers = rng.uniform(-length / 2 + 2 * h, length / 2 - 2 * h, (num_blobs, 3))
    positions = (centers[:, None, :] + walk).reshape(-1, ndim)[:M]
    strengths = rng.standard_normal(M)
    if order == "random":
        positions = positions[rng.permutation(M)]
    return positions.astype(np.float32), strengths.astype(np.float32)


def points_path(cache_dir, ndim, n, M, order):
    return Path(cache_dir) / f"pts_{ndim}d_{n}_{M}_{order}"


def load_points(cache_dir, ndim, n, M, order):
    stem = points_path(cache_dir, ndim, n, M, order)
    pos_file, str_file = (
        stem.with_name(stem.name + "_pos.npy"),
        stem.with_name(stem.name + "_str.npy"),
    )
    if not pos_file.exists():
        positions, strengths = make_points(ndim, n, M, order)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(pos_file, positions)
        np.save(str_file, strengths)
    return np.load(pos_file), np.load(str_file)


def median_ms(fn, reps=10):
    jax.block_until_ready(fn())
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def peak_bytes():
    return jax.devices()[0].memory_stats()["peak_bytes_in_use"]


def rel_error(out, ref_file):
    ref = np.load(ref_file)
    out = np.asarray(out)
    return float(np.linalg.norm(out - ref) / np.linalg.norm(ref))


def run(args):
    ndim, n, M = args.ndim, args.n, args.M
    shape = (n,) * ndim
    half_shape = (*shape[:-1], n // 2 + 1)
    positions, strengths = load_points(args.cache_dir, ndim, n, M, args.order)
    positions, strengths = jnp.asarray(positions), jnp.asarray(strengths)
    keys = jax.random.split(jax.random.key(0), 4)
    target = jax.random.normal(keys[0], half_shape, jnp.complex64)
    dpos = jax.random.normal(keys[1], positions.shape, jnp.float32)
    dstr = jax.random.normal(keys[2], strengths.shape, jnp.float32)
    row = dict(
        ndim=ndim,
        n=n,
        M=M,
        eps=args.eps,
        sigma=args.sigma,
        backend=args.backend,
        order=args.order,
        device=str(jax.devices()[0]),
    )
    if args.backend in PALLAS:
        kernel = make_pswf_kernel(args.eps, args.sigma, ndim, np.float32)
        fine_shape = tuple(fine_grid_size(s, args.sigma, kernel.width) for s in shape)
        row.update(width=kernel.width, fine_shape=list(fine_shape))
        pallas = PALLAS[args.backend]

        def transform(pos, c):
            return dispatch_nufft1(
                pos,
                c,
                shape,
                pixel_size=PIXEL_SIZE,
                eps=args.eps,
                upsample_factor=args.sigma,
                enable_pallas=pallas,
            )
    else:
        pallas = GMP_PALLAS[args.backend]
        pos3 = jnp.concatenate([positions, jnp.zeros((M, 3 - ndim))], axis=1)
        volume = cxs.GaussianMixtureVolume(pos3, strengths, jnp.full((M,), ATOM_VARIANCE))
        config = cxs.BasicImageConfig(
            shape=shape, pixel_size=PIXEL_SIZE, voltage_in_kilovolts=300.0
        )
        integrator = cxs.GaussianMixtureProjection.with_calibration(
            volume,
            config,
            sampling_mode="point",
            upsample_factor=2,
            tolerance=args.eps,
            cutoff_sigma=math.sqrt(2 * math.log(10 / args.eps)),
            enable_pallas=pallas,
        )
        row.update(n_spread=[int(v) for v in np.atleast_1d(integrator.n_spread)])

        def transform(pos, c):
            vol = cxs.GaussianMixtureVolume(
                jnp.concatenate([pos, jnp.zeros((M, 3 - ndim))], axis=1),
                c,
                jnp.full((M,), ATOM_VARIANCE),
            )
            return integrator.integrate(vol, config)

    def loss(pos, c):
        return 0.5 * jnp.sum(jnp.abs(transform(pos, c) - target) ** 2)

    forward = jax.jit(transform)
    value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    hvp = jax.jit(
        lambda p, c, vp, vc: jax.jvp(jax.grad(loss, argnums=(0, 1)), (p, c), (vp, vc))[1]
    )
    row["peak_bytes_data"] = peak_bytes()
    try:
        row["forward_ms"] = median_ms(lambda: forward(positions, strengths))
        row["peak_bytes_forward"] = peak_bytes()
        row["value_grad_ms"] = median_ms(lambda: value_and_grad(positions, strengths))
        row["peak_bytes_value_grad"] = peak_bytes()
        if args.backend in PALLAS:
            row["hvp_ms"] = median_ms(lambda: hvp(positions, strengths, dpos, dstr))
            row["peak_bytes_hvp"] = peak_bytes()
    except Exception as err:  # noqa: BLE001
        row["error"] = repr(err)[:300]
    if "forward_ms" in row:
        ref_file = points_path(args.cache_dir, ndim, n, M, args.order)
        ref_file = ref_file.with_name(ref_file.name + "_ref.npy")
        if ref_file.exists():
            out = forward(positions, strengths)
            if args.backend in GMP_PALLAS:
                # Pixel-sampled gaussians of variance `ATOM_VARIANCE`, image pixels
                freqs = [np.fft.fftfreq(s, PIXEL_SIZE) for s in shape[:-1]]
                freqs.append(np.fft.rfftfreq(shape[-1], PIXEL_SIZE))
                k2 = sum(f**2 for f in np.meshgrid(*freqs, indexing="ij"))
                ref = np.load(ref_file) * np.exp(-2 * np.pi**2 * ATOM_VARIANCE * k2)
                ref /= PIXEL_SIZE**2
                row["rel_error"] = float(
                    np.linalg.norm(np.asarray(out) - ref) / np.linalg.norm(ref)
                )
            else:
                row["rel_error"] = rel_error(out, ref_file)
    if args.backend in PALLAS and "forward_ms" in row:
        row.update(split_ms(args, positions, strengths, shape, kernel, fine_shape, keys))
    return row


def split_ms(args, positions, strengths, shape, kernel, fine_shape, keys):
    """Forward-pass time of the spread, of the fine `rfftn`, and of the whole."""
    use_pallas = PALLAS[args.backend]
    fwd = use_pallas if isinstance(use_pallas, bool) else use_pallas.get("fwd", False)
    c = strengths[None]

    def spread_only(pos, c):
        u = _fine_coordinates(pos, shape, fine_shape, PIXEL_SIZE)
        return _spread_dispatch(u, c, fine_shape, kernel, fwd, False)

    grid = jax.random.normal(keys[3], (1, *fine_shape), jnp.float32)
    rfft = jax.jit(lambda g: jnp.fft.rfftn(g, axes=tuple(range(1, len(shape) + 1))))
    spread_jit = jax.jit(spread_only)
    return dict(
        spread_ms=median_ms(lambda: spread_jit(positions, c)),
        fft_ms=median_ms(lambda: rfft(grid)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=[*PALLAS, *GMP_PALLAS], default="jax")
    parser.add_argument("--ndim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--M", type=int, default=4_000_000)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--order", choices=["chain", "random"], default="chain")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--prepare", action="store_true", help="only write the points")
    args = parser.parse_args()
    if args.prepare:
        load_points(args.cache_dir, args.ndim, args.n, args.M, args.order)
        return
    row = run(args)
    print(json.dumps(row), flush=True)
    if args.out is not None:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
