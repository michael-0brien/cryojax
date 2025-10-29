from time import time

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.constants import PengScatteringFactorParameters
from cryojax.dataset import RelionParticleParameterFile
from cryojax.io import read_atoms_from_pdb
from cryojax.ndimage import operators as op
from cryojax.rotations import SO3
from jaxtyping import PRNGKeyArray


def setup(num_images, path_to_pdb, path_to_starfile):
    @eqx.filter_vmap(in_axes=(0, None))
    def make_particle_parameters(key: PRNGKeyArray, config: cxs.BasicImageConfig):
        """Generate random parameters."""
        # Pose
        # ... instantiate rotations
        key, subkey = jax.random.split(
            key
        )  # split the key to use for the next random number
        rotation = SO3.sample_uniform(subkey)

        # ... now in-plane translation
        ny, nx = config.shape

        key, subkey = jax.random.split(key)  # do this everytime you use a key!!
        offset_in_angstroms = (
            jax.random.uniform(subkey, (2,), minval=-0.1, maxval=0.1)
            * jnp.asarray((nx, ny))
            / 2
            * config.pixel_size
        )
        # ... build the pose
        pose = cxs.EulerAnglePose.from_rotation_and_translation(
            rotation, offset_in_angstroms
        )

        # CTF Parameters
        # ... defocus
        key, subkey = jax.random.split(key)
        defocus_in_angstroms = jax.random.uniform(subkey, (), minval=10000, maxval=15000)
        # ... astigmatism
        key, subkey = jax.random.split(key)
        astigmatism_in_angstroms = jax.random.uniform(subkey, (), minval=0, maxval=100)
        key, subkey = jax.random.split(key)
        astigmatism_angle = jax.random.uniform(subkey, (), minval=0, maxval=jnp.pi)
        # Now non-random values
        spherical_aberration_in_mm = 2.7
        amplitude_contrast_ratio = 0.1
        # Build the CTF
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(
                defocus_in_angstroms=defocus_in_angstroms,
                astigmatism_in_angstroms=astigmatism_in_angstroms,
                astigmatism_angle=astigmatism_angle,
                spherical_aberration_in_mm=spherical_aberration_in_mm,
            ),
            amplitude_contrast_ratio=amplitude_contrast_ratio,
        )

        return {
            "image_config": config,
            "pose": pose,
            "transfer_theory": transfer_theory,
        }

    # Generate particle parameters. First, the image config
    pad_options = dict(shape=(200, 200))

    config = cxs.BasicImageConfig(
        shape=(150, 150),
        pixel_size=2.0,
        voltage_in_kilovolts=300.0,
        pad_options=pad_options,
    )
    # ... RNG keys
    keys = jax.random.split(jax.random.key(0), num_images)
    # ... make parameters
    particle_parameters = make_particle_parameters(keys, config)

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",  # writing mode!
        exists_ok=True,  # in case the file already exists
        broadcasts_image_config=True,
        pad_options=pad_options,
    )
    parameter_file.append(particle_parameters)

    atom_positions, atom_types, atom_properties = read_atoms_from_pdb(
        path_to_pdb,
        center=True,
        loads_properties=True,
        selection_string="name CA",  # C-Alphas for simplicity
    )
    scattering_parameters = PengScatteringFactorParameters(atom_types)
    volume_gmm = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        scattering_parameters,
        extra_b_factors=atom_properties["b_factors"],
    )

    real_voxel_grid = volume_gmm.to_real_voxel_grid(shape=(150, 150, 150), voxel_size=1.0)
    volume_fourier_grid = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        real_voxel_grid, pad_scale=2
    )
    # print(atom_properties["b_factors"].mean() * 8 * jnp.pi**2)
    atom_volume = cxs.IndependentAtomVolume(
        position_pytree=atom_positions,
        scattering_factor_pytree=op.FourierGaussian(amplitude=1.0, b_factor=10.0),
    )
    return (
        particle_parameters,
        parameter_file,
        volume_gmm,
        atom_volume,
        volume_fourier_grid,
    )


# def compute_image(particle_parameters, volume, volume_integrator, rng_key):
#     # Build image model, including normalization within a circular mask
#     # around each particle
#     image_config, pose, transfer_theory = (
#         particle_parameters["image_config"],
#         particle_parameters["pose"],
#         particle_parameters["transfer_theory"],
#     )
#     mask = tf.CircularCosineMask(
#         coordinate_grid=image_config.coordinate_grid_in_angstroms,
#         radius=150.0,
#         rolloff_width=0.0,
#         xy_offset=pose.offset_in_angstroms,
#     )
#     image_model = cxs.make_image_model(
#         volume_parametrisation=volume,
#         pose=pose,
#         image_config=image_config,
#         transfer_theory=transfer_theory,
#         volume_integrator=volume_integrator,
#         normalizes_signal=True,
#         signal_region=mask.get() == 1.0,
#     )
#     # Build noise model at a randomly sampled SNR within a
#     # uniform range, then simulate
#     simulator_rng_key, snr_rng_key = jax.random.split(rng_key, 2)
#     snr = jax.random.uniform(snr_rng_key, minval=0.01, maxval=0.1)
#     noise_model = cxs.UncorrelatedGaussianNoiseModel(
#         image_model,
#         variance=1.0,
#         signal_scale_factor=jnp.sqrt(snr),
#     )

#     return noise_model.sample(rng_key=simulator_rng_key)


# def benchmark_projection_microscope_noise_no_vmap(
#     n_iterations, path_to_pdb, path_to_starfile
# ):
#     _, parameter_file, volume_gmm, volume_fourier_grid = setup(
#         path_to_pdb, path_to_starfile
#     )

#     time_list = []
#     for _ in range(n_iterations + 1):
#         start_time = time()
#         gmm_image = compute_image(
#             parameter_file[0],
#             volume_gmm,
#             cxs.GaussianMixtureProjection(),
#             jax.random.key(1234),
#         )
#         gmm_image.block_until_ready()
#         end_time = time()
#         time_list.append(end_time - start_time)
#     gmm_avg_time = jnp.mean(jnp.array(time_list[1:]))
#     gmm_std_time = jnp.std(jnp.array(time_list[1:]))
#     print(f"GMM: {gmm_avg_time:.4f} +/- {gmm_std_time:.4f} s")

#     time_list = []
#     for _ in range(n_iterations + 1):
#         start_time = time()
#         gmm_image = eqx.filter_jit(compute_image)(
#             parameter_file[0],
#             volume_gmm,
#             cxs.GaussianMixtureProjection(),
#             jax.random.key(1234),
#         )
#         gmm_image.block_until_ready()
#         end_time = time()
#         time_list.append(end_time - start_time)
#     jit_gmm_avg_time = jnp.mean(jnp.array(time_list[1:]))
#     jit_gmm_std_time = jnp.std(jnp.array(time_list[1:]))
#     print(
#         f"JIT, GMM: {jit_gmm_avg_time:.4f} +/- {jit_gmm_std_time:.4f} s"
#     )

#     time_list = []
#     for _ in range(n_iterations + 1):
#         start_time = time()
#         fs_image = compute_image(
#             parameter_file[0],
#             volume_fourier_grid,
#             cxs.FourierSliceExtraction(),
#             jax.random.key(1234),
#         )
#         fs_image.block_until_ready()
#         end_time = time()
#         time_list.append(end_time - start_time)
#     fs_avg_time = jnp.mean(jnp.array(time_list[1:]))
#     fs_std_time = jnp.std(jnp.array(time_list[1:]))
#     print(f"Fourier Slice: {fs_avg_time:.4f} +/- {fs_std_time:.4f} s")

#     time_list = []
#     for _ in range(n_iterations + 1):
#         start_time = time()
#         fs_image = eqx.filter_jit(compute_image)(
#             parameter_file[0],
#             volume_fourier_grid,
#             cxs.FourierSliceExtraction(),
#             jax.random.key(1234),
#         )
#         fs_image.block_until_ready()
#         end_time = time()
#         time_list.append(end_time - start_time)
#     fs_avg_time = jnp.mean(jnp.array(time_list[1:]))
#     fs_std_time = jnp.std(jnp.array(time_list[1:]))
#     print(f"JIT, Fourier Slice: {fs_avg_time:.4f} +/- {fs_std_time:.4f} s")


@eqx.filter_vmap(in_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None, None))
def simulate_image_nojit(
    image_config, pose, transfer_theory, potential, volume_integrator
):
    image_model = cxs.make_image_model(
        volume_parametrization=potential,
        image_config=image_config,
        pose=pose,
        transfer_theory=transfer_theory,
        volume_integrator=volume_integrator,
    )
    return image_model.simulate()


simulate_image_jit = eqx.filter_jit(simulate_image_nojit)


def benchmark_fourier_slice_vs_gmm(
    n_iterations, num_images, path_to_pdb, path_to_starfile
):
    particle_parameters, _, volume_gmm, atom_volume, volume_fourier_grid = setup(
        num_images, path_to_pdb, path_to_starfile
    )

    image_config, pose, transfer_theory = (
        particle_parameters["image_config"],
        particle_parameters["pose"],
        particle_parameters["transfer_theory"],
    )

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        fft_image = simulate_image_nojit(
            image_config,
            pose,
            transfer_theory,
            atom_volume,
            cxs.FFTAtomProjection(eps=1e-16),
        )
        fft_image.block_until_ready()
        end_time = time()
        time_list.append(end_time - start_time)
    fft_avg_time = jnp.mean(jnp.array(time_list[1:]))
    fft_std_time = jnp.std(jnp.array(time_list[1:]))
    print(f"FFTproj (no JIT): {1000 * fft_avg_time:.2f} +/- {1000 * fft_std_time:.2f} ms")

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        fft_image = simulate_image_jit(
            image_config,
            pose,
            transfer_theory,
            atom_volume,
            cxs.FFTAtomProjection(eps=1e-16),
        )
        fft_image.block_until_ready()
        end_time = time()
        time_list.append(end_time - start_time)
    fft_avg_time = jnp.mean(jnp.array(time_list[1:]))
    fft_std_time = jnp.std(jnp.array(time_list[1:]))
    print(f"FFTproj (JIT): {1000 * fft_avg_time:.2f} +/- {1000 * fft_std_time:.2f} ms")

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        gmm_image = simulate_image_nojit(
            image_config,
            pose,
            transfer_theory,
            volume_gmm,
            cxs.GaussianMixtureProjection(),
        )
        gmm_image.block_until_ready()
        end_time = time()
        time_list.append(end_time - start_time)
    gmm_avg_time = jnp.mean(jnp.array(time_list[1:]))
    gmm_std_time = jnp.std(jnp.array(time_list[1:]))
    print(f"GMM (no JIT): {1000 * gmm_avg_time:.2f} +/- {1000 * gmm_std_time:.2f} ms")

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        gmm_image = simulate_image_jit(
            image_config,
            pose,
            transfer_theory,
            volume_gmm,
            cxs.GaussianMixtureProjection(),
        )
        gmm_image.block_until_ready()
        end_time = time()
        time_list.append(end_time - start_time)
    jit_gmm_avg_time = jnp.mean(jnp.array(time_list[1:]))
    jit_gmm_std_time = jnp.std(jnp.array(time_list[1:]))
    print(
        f"GMM (JIT): {1000 * jit_gmm_avg_time:.2f} +/- {1000 * jit_gmm_std_time:.2f} ms"
    )

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        fs_image = simulate_image_nojit(
            image_config,
            pose,
            transfer_theory,
            volume_fourier_grid,
            cxs.FourierSliceExtraction(),
        )
        fs_image.block_until_ready()
        end_time = time()
        time_list.append(end_time - start_time)
    fs_avg_time = jnp.mean(jnp.array(time_list[1:]))
    fs_std_time = jnp.std(jnp.array(time_list[1:]))
    print(
        f"Fourier Slice (no JIT): {1000 * fs_avg_time:.2f} "
        f"+/- {1000 * fs_std_time:.2f} ms"
    )

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        fs_image = simulate_image_jit(
            image_config,
            pose,
            transfer_theory,
            volume_fourier_grid,
            cxs.FourierSliceExtraction(),
        )
        fs_image.block_until_ready()
        end_time = time()
        time_list.append(end_time - start_time)
    fs_avg_time = jnp.mean(jnp.array(time_list[1:]))
    fs_std_time = jnp.std(jnp.array(time_list[1:]))
    print(
        f"Fourier Slice (JIT): {1000 * fs_avg_time:.2f} +/- {1000 * fs_std_time:.2f} ms"
    )


if __name__ == "__main__":
    n_iterations, n_images = 10, 100
    print(
        f"Benchmarking image simulation of {n_images} images "
        f"averaged over {n_iterations} iterations"
    )
    path_to_pdb = "../data/thyroglobulin_initial.pdb"
    path_to_starfile = "../outputs/particles.star"
    benchmark_fourier_slice_vs_gmm(n_iterations, n_images, path_to_pdb, path_to_starfile)
