from time import time

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.constants import PengScatteringFactorParameters
from cryojax.io import read_atoms_from_pdb
from cryojax.rotations import SO3
from jaxtyping import PRNGKeyArray


def setup(num_images, path_to_pdb):
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
    config = cxs.BasicImageConfig(
        shape=(150, 150),
        pixel_size=2.0,
        voltage_in_kilovolts=300.0,
        padded_shape=(200, 200),
    )
    # ... RNG keys
    keys = jax.random.split(jax.random.key(0), num_images)
    # ... make parameters
    particle_parameters = make_particle_parameters(keys, config)

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
    render_fn = cxs.GaussianMixtureRenderFn(shape=(200, 200, 200), voxel_size=1.0)
    volume_fourier_grid = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(volume_gmm), pad_scale=2
    )
    atom_volume = cxs.IndependentAtomVolume(
        positions=atom_positions,
        scattering_factors=im.FourierGaussian(amplitude=1.0, b_factor=10.0),
    )
    return (particle_parameters, volume_gmm, atom_volume, volume_fourier_grid)


@eqx.filter_vmap(in_axes=(eqx.if_array(0), eqx.if_array(0), eqx.if_array(0), None, None))
def simulate_image_nojit(
    image_config, pose, transfer_theory, potential, volume_integrator
):
    image_model = cxs.make_image_model(
        volume=potential,
        image_config=image_config,
        pose=pose,
        transfer_theory=transfer_theory,
        volume_integrator=volume_integrator,
    )
    return image_model.simulate()


simulate_image_jit = eqx.filter_jit(simulate_image_nojit)


def run(n_iterations, num_images, path_to_pdb):
    particle_parameters, volume_gmm, atom_volume, volume_fourier_grid = setup(
        num_images, path_to_pdb
    )

    image_config, pose, transfer_theory = (
        particle_parameters["image_config"],
        particle_parameters["pose"],
        particle_parameters["transfer_theory"],
    )
    dim, num_atoms = image_config.padded_y_dim, atom_volume.positions.shape[0]
    print(
        f"Benchmarking simulation of {n_images} {dim}x{dim} images of "
        f"a structure with {num_atoms} atoms..."
    )

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        with jax.disable_jit():
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
    print(f"NUFFT (no JIT): {1000 * fft_avg_time:.2f} +/- {1000 * fft_std_time:.2f} ms")

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
    print(f"NUFFT (JIT): {1000 * fft_avg_time:.2f} +/- {1000 * fft_std_time:.2f} ms")

    time_list = []
    for _ in range(n_iterations + 1):
        start_time = time()
        with jax.disable_jit():
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
        with jax.disable_jit():
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
    n_iterations, n_images = 25, 10
    # Necessary to pull from git LFS
    path_to_pdb = "../docs/examples/data/thyroglobulin_initial.pdb"
    run(n_iterations, n_images, path_to_pdb)
