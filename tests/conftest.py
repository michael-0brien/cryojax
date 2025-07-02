import os

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax.experimental as cxe
    import cryojax.simulator as cxs
    from cryojax.io import read_array_with_spacing_from_mrc
    from cryojax.ndimage import operators as op, rfftn, transforms as tf


# jax.config.update("jax_numpy_dtype_promotion", "strict")
# jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sample_mrc_path():
    return os.path.join(os.path.dirname(__file__), "data", "3j9g_potential_ps4_4.mrc")


@pytest.fixture
def sample_pdb_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao.pdb")


@pytest.fixture
def sample_pdb_path_assembly():
    return os.path.join(os.path.dirname(__file__), "data", "1uao_assembly.pdb")


@pytest.fixture
def sample_cif_path():
    return os.path.join(os.path.dirname(__file__), "data", "8zpm.cif")


@pytest.fixture
def sample_pdb_url():
    return "https://files.rcsb.org/download/2XJX.pdb"


@pytest.fixture
def sample_waterbox_pdb():
    return os.path.join(os.path.dirname(__file__), "data", "relaxed_small_box_tip3p.pdb")


@pytest.fixture
def toy_gaussian_cloud():
    atom_positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    num_atoms = atom_positions.shape[0]
    ff_a = jnp.array(
        num_atoms
        * [
            [1.0, 0.5],
        ]
    )

    ff_b = jnp.array(
        num_atoms
        * [
            [0.3, 0.2],
        ]
    )

    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.05
    return (atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size)


@pytest.fixture
def pixel_size():
    return 4.4


@pytest.fixture
def voltage_in_kilovolts():
    return 300.0


@pytest.fixture
def config(pixel_size, voltage_in_kilovolts):
    return cxs.InstrumentConfig(
        (65, 66),
        pixel_size,
        voltage_in_kilovolts,
        electrons_per_angstrom_squared=50.0,
        pad_scale=1.1,
    )


@pytest.fixture
def projection_method():
    return cxs.FourierSliceExtraction()


@pytest.fixture
def potential(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(sample_mrc_path)
    return cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size, pad_scale=1.3
    )


@pytest.fixture
def filters(config):
    return tf.LowpassFilter(config.padded_frequency_grid_in_pixels)


@pytest.fixture
def masks(config):
    return tf.CircularCosineMask(
        config.padded_coordinate_grid_in_angstroms,
        radius=20 * float(config.pixel_size),
        rolloff_width=3 * float(config.pixel_size),
    )


@pytest.fixture
def transfer_theory():
    return cxs.ContrastTransferTheory(ctf=cxs.CTF())


@pytest.fixture
def detector():
    return cxs.PoissonDetector(cxs.NullDQE())


@pytest.fixture
def pose():
    return cxs.EulerAnglePose(
        phi_angle=30.0,
        theta_angle=100.0,
        psi_angle=-10.0,
        offset_x_in_angstroms=10.0,
        offset_y_in_angstroms=-5.0,
    )


@pytest.fixture
def specimen(potential, pose):
    return cxs.Structure(potential, pose)


@pytest.fixture
def solvent():
    return cxe.GRFSolvent(
        thickness_in_angstroms=100.0, power_spectrum_function=op.Constant(1.0)
    )


@pytest.fixture
def theory(projection_method, transfer_theory, solvent):
    return cxs.WeakPhaseScatteringTheory(projection_method, transfer_theory, solvent)


@pytest.fixture
def theory_with_solvent(projection_method, transfer_theory, solvent):
    return cxs.WeakPhaseScatteringTheory(projection_method, transfer_theory, solvent)


@pytest.fixture
def noiseless_model(specimen, config, theory):
    return cxs.IntensityImageModel(
        structure=specimen, instrument_config=config, scattering_theory=theory
    )


@pytest.fixture
def noisy_model(specimen, config, theory_with_solvent, detector):
    return cxs.ElectronCountsImageModel(
        structure=specimen,
        instrument_config=config,
        scattering_theory=theory_with_solvent,
        detector=detector,
    )


@pytest.fixture
def test_noisy_image(noisy_model):
    image = noisy_model.render(jr.key(1234))
    return rfftn(image)


@pytest.fixture
def test_noiseless_image(noiseless_model):
    image = noiseless_model.render(jr.key(1234))
    return rfftn(image)
