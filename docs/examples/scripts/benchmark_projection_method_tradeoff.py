from time import time

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cryojax.constants import PengScatteringFactorParameters
from cryojax.io import read_atoms_from_pdb
from cryojax.ndimage import operators as op


def setup_volumes_and_configs(n_iterations, n_atoms, box_size, pixel_size=2.0):
    """Setup volumes and image configs for different test conditions."""
    # Read atoms (subsample to get desired n_atoms)
    atom_positions, atom_types, atom_properties = read_atoms_from_pdb(
        "data/thyroglobulin_initial.pdb",
        center=True,
        loads_properties=True,
        selection_string="name CA",
    )

    # Subsample atoms if needed
    if n_atoms < len(atom_positions):
        indices = jax.random.choice(
            jax.random.key(42), len(atom_positions), (n_atoms,), replace=False
        )
        atom_positions = atom_positions[indices]
        atom_types = [atom_types[i] for i in indices]
        atom_properties = {k: v[indices] for k, v in atom_properties.items()}

    # Create volumes
    scattering_parameters = PengScatteringFactorParameters(atom_types)
    volume_gmm = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        scattering_parameters,
        extra_b_factors=atom_properties["b_factors"],
    )

    # Fourier slice volume (pre-computed grid)
    real_voxel_grid = volume_gmm.to_real_voxel_grid(
        shape=(box_size, box_size, box_size), voxel_size=pixel_size
    )

    times = []
    for _ in range(n_iterations + 1):
        start_time = time()
        # The volume rendering is already done in setup, but we simulate the cost
        _ = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=2)
        jax.block_until_ready(_)
        times.append(time() - start_time)
    avg_time = np.mean(times[1:])
    print(f"Average time to create FourierVoxelGridVolume: {avg_time * 1000:.2f} ms")

    volume_fourier_grid = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        real_voxel_grid, pad_scale=2
    )

    # Atom volume for direct projection
    atom_volume = cxs.IndependentAtomVolume(
        position_pytree=atom_positions,
        scattering_factor_pytree=op.FourierGaussian(amplitude=1.0, b_factor=10.0),
    )

    # Image config
    pad_options = dict(shape=(int(box_size * 1.5), int(box_size * 1.5)))
    config = cxs.BasicImageConfig(
        shape=(box_size, box_size),
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
        pad_options=pad_options,
    )

    return volume_fourier_grid, avg_time, volume_gmm, atom_volume, config


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None, None))
def simulate_image_batch(image_config, pose, transfer_theory, volume, integrator):
    """Simulate a batch of images."""
    image_model = cxs.make_image_model(
        volume_parametrization=volume,
        image_config=image_config,
        pose=pose,
        transfer_theory=transfer_theory,
        volume_integrator=integrator,
    )
    return image_model.simulate()


def generate_random_parameters(n_projections, config):
    """Generate random poses and CTF parameters."""
    from cryojax.rotations import SO3

    keys = jax.random.split(jax.random.key(12345), n_projections)

    def make_single_params(key):
        key, subkey = jax.random.split(key)
        rotation = SO3.sample_uniform(subkey)

        key, subkey = jax.random.split(key)
        ny, nx = config.shape
        offset_in_angstroms = (
            jax.random.uniform(subkey, (2,), minval=-0.1, maxval=0.1)
            * jnp.asarray((nx, ny))
            / 2
            * config.pixel_size
        )

        pose = cxs.EulerAnglePose.from_rotation_and_translation(
            rotation, offset_in_angstroms
        )

        key, subkey = jax.random.split(key)
        defocus = jax.random.uniform(subkey, (), minval=10000, maxval=15000)

        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=50.0,
                astigmatism_angle=0.0,
                spherical_aberration_in_mm=2.7,
            ),
            amplitude_contrast_ratio=0.1,
        )

        return config, pose, transfer_theory

    configs, poses, transfer_theories = eqx.filter_vmap(make_single_params)(keys)
    return configs, poses, transfer_theories


def benchmark_projection_methods(
    n_projections_list,
    n_atoms_list,
    box_sizes,
    n_iterations,
    antialias,
    use_error_functions,
):
    """Benchmark both projection methods across different conditions."""
    results = []

    for n_atoms in n_atoms_list:
        for box_size in box_sizes:
            print(f"Testing n_atoms={n_atoms}, box_size={box_size}")

            # Setup volumes
            volume_fourier_grid, volume_render_time, gmm_volume, atom_volume, config = (
                setup_volumes_and_configs(n_iterations, n_atoms, box_size)
            )

            for n_projections in n_projections_list:
                print(f"  n_projections={n_projections}")

                # Generate parameters
                configs, poses, transfer_theories = generate_random_parameters(
                    n_projections, config
                )

                # Benchmark Fourier Slice Extraction
                times = []
                integrator = cxs.FourierSliceExtraction()
                for _ in range(n_iterations + 1):
                    start_time = time()
                    images = simulate_image_batch(
                        configs,
                        poses,
                        transfer_theories,
                        volume_fourier_grid,
                        integrator,
                    )
                    images.block_until_ready()
                    times.append(time() - start_time)

                fs_time_per_projection = np.mean(times[1:])
                fs_total_time = volume_render_time + fs_time_per_projection

                # Benchmark Atom Projection (FFT)
                times = []
                integrator = cxs.FFTAtomProjection(eps=1e-16, antialias=antialias)
                for _ in range(n_iterations + 1):
                    start_time = time()
                    images = simulate_image_batch(
                        configs,
                        poses,
                        transfer_theories,
                        atom_volume,
                        integrator,
                    )
                    images.block_until_ready()
                    times.append(time() - start_time)

                atom_time_total = np.mean(times[1:])

                # Benchmark GMM Projection (FFT)
                times = []
                integrator = cxs.GaussianMixtureProjection(
                    use_error_functions=use_error_functions
                )
                for _ in range(n_iterations + 1):
                    start_time = time()
                    images = simulate_image_batch(
                        configs,
                        poses,
                        transfer_theories,
                        gmm_volume,
                        integrator,
                    )
                    images.block_until_ready()
                    times.append(time() - start_time)

                gmm_time_total = np.mean(times[1:])

                results.append(
                    {
                        "n_atoms": n_atoms,
                        "box_size": box_size,
                        "n_projections": n_projections,
                        "fourier_slice_time": fs_total_time,
                        "atom_projection_time": atom_time_total,
                        "fs_volume_render_time": volume_render_time,
                        "fs_projection_time": fs_time_per_projection,
                        "gmm_projection_time": gmm_time_total,
                    }
                )

    return pd.DataFrame(results)


def plot_crossover_analysis(df, datetimestamp, antialias, use_error_functions):
    """Plot the crossover analysis showing when each method is faster."""
    n_atoms_list = df["n_atoms"].unique()
    box_sizes = df["box_size"].unique()

    fig, axes = plt.subplots(
        len(n_atoms_list),
        len(box_sizes),
        figsize=(4 * len(box_sizes), 4 * len(n_atoms_list)),
        squeeze=False,
    )

    for i, n_atoms in enumerate(n_atoms_list):
        for j, box_size in enumerate(box_sizes):
            ax = axes[i, j]

            subset = df[(df["n_atoms"] == n_atoms) & (df["box_size"] == box_size)]

            ax.plot(
                subset["n_projections"],
                subset["fourier_slice_time"] * 1000,
                "o-",
                label="Fourier Slice",
                color="blue",
                linewidth=2,
            )
            ax.plot(
                subset["n_projections"],
                subset["atom_projection_time"] * 1000,
                "s-",
                label=f"Atom Projection (antialias={antialias})",
                color="red",
                linewidth=2,
            )
            ax.plot(
                subset["n_projections"],
                subset["gmm_projection_time"] * 1000,
                "^-",
                label=f"GMM Projection (use_error_functions={use_error_functions})",
                color="green",
                linewidth=2,
            )

            ax.set_xlabel("Number of Projections")
            ax.set_ylabel("Total Time (ms)")
            ax.set_title(f"{n_atoms} atoms, {box_size}×{box_size} box")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        f"benchmark_projection_method_tradeoff_{datetimestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def find_crossover_points(df):
    """Find crossover points where methods have equal performance."""
    crossovers = []

    for (n_atoms, box_size), group in df.groupby(["n_atoms", "box_size"]):
        group = group.sort_values("n_projections")

        fs_times = group["fourier_slice_time"].values
        gmm_times = group["gmm_projection_time"].values
        n_proj = group["n_projections"].values

        # Find where lines cross
        diff = fs_times - gmm_times
        sign_changes = np.where(np.diff(np.signbit(diff)))[0]

        for idx in sign_changes:
            # Linear interpolation to find precise crossover
            x1, x2 = n_proj[idx], n_proj[idx + 1]
            y1, y2 = diff[idx], diff[idx + 1]
            crossover_n_proj = x1 - y1 * (x2 - x1) / (y2 - y1)

            crossovers.append(
                {
                    "n_atoms": n_atoms,
                    "box_size": box_size,
                    "crossover_n_projections": crossover_n_proj,
                }
            )

    return pd.DataFrame(crossovers)


if __name__ == "__main__":
    # Test parameters
    n_projections_list = [1, 3, 10, 30, 100]
    n_atoms_list = [30, 100, 300]
    box_sizes = [
        32,
        64,
    ]

    print("Running projection method crossover benchmark...")
    print("This will test Fourier slicing vs GMM projection across different conditions")

    # Run benchmark
    antialias = False
    use_error_functions = True
    results_df = benchmark_projection_methods(
        n_projections_list,
        n_atoms_list,
        box_sizes,
        n_iterations=3,
        antialias=antialias,
        use_error_functions=use_error_functions,
    )

    # Save results
    datetimestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(
        f"benchmark_projection_method_tradeoff_{datetimestamp}.csv", index=False
    )
    print(f"Results saved to benchmark_projection_method_tradeoff_{datetimestamp}.csv")

    # Plot results
    plot_crossover_analysis(results_df, datetimestamp, antialias, use_error_functions)

    # Find and display crossover points
    crossovers = find_crossover_points(results_df)
    print("\nCrossover points (where methods have equal performance):")
    print(crossovers)

    # Summary statistics
    print("\nSummary:")
    for (n_atoms, box_size), group in results_df.groupby(["n_atoms", "box_size"]):
        fs_faster = (group["fourier_slice_time"] < group["gmm_projection_time"]).sum()
        gmm_faster = len(group) - fs_faster
        print(
            f"{n_atoms} atoms, {box_size}×{box_size}: "
            f"Fourier slice faster in {fs_faster}/{len(group)} cases, "
            f"GMM projection faster in {gmm_faster}/{len(group)} cases"
        )
