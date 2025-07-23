import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.constants import (
        get_tabulated_scattering_factor_parameters,
        read_peng_element_scattering_factor_parameter_table,
    )
    from cryojax.coordinates import make_coordinate_grid
    from cryojax.io import read_atoms_from_pdb
    from cryojax.ndimage import irfftn
    from cryojax.simulator import (
        BasicConfig,
        GaussianMixtureAtomicPotential,
        GaussianMixtureProjection,
        PengAtomicPotential,
    )


config.update("jax_enable_x64", True)


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


@pytest.mark.parametrize("shape", ((64, 64), (63, 63), (63, 64), (64, 63)))
def test_atom_integrator_shape(sample_pdb_path, shape):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
        loads_b_factors=True,
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )
    atom_potential = PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
        b_factors=b_factors,
    )
    pixel_size = 0.5

    integrator = GaussianMixtureProjection(upsampling_factor=2)
    # # ... and the configuration of the imaging instrument
    config = BasicConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volumetric_potential
    fourier_integrated_potential = integrator.integrate(
        atom_potential, config, outputs_real_space=False
    )

    assert fourier_integrated_potential.shape == (shape[0], shape[1] // 2 + 1)


def test_downsampled_gmm_potential_agreement(sample_pdb_path):
    """Integration test ensuring that rasterized voxel grids roughly
    agree with downsampled versions.
    """
    atom_positions, atom_identities = read_atoms_from_pdb(
        sample_pdb_path,
        loads_b_factors=False,
        center=True,
        selection_string="not element H",
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )
    atom_potential = PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
    )

    # Parameters for rasterization
    shape = (128, 128)
    pixel_size = 0.25

    # Downsampling parameters
    downsampling_factor = 2
    downsampled_shape = (
        int(shape[0] / downsampling_factor),
        int(shape[1] / downsampling_factor),
    )
    downsampled_pixel_size = pixel_size * downsampling_factor

    integrator_int_hires = GaussianMixtureProjection(
        upsampling_factor=downsampling_factor
    )
    integrator_int_lowres = GaussianMixtureProjection(upsampling_factor=1)
    # ... and the configuration of the imaging instrument
    config = BasicConfig(
        shape=downsampled_shape,
        pixel_size=downsampled_pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volumetric_potential
    image_from_hires = integrator_int_hires.integrate(atom_potential, config)
    image_lowres = integrator_int_lowres.integrate(atom_potential, config)

    assert image_from_hires.shape == image_lowres.shape


def test_peng_vs_gmm_agreement(sample_pdb_path):
    """Integration test ensuring that Peng Potential and GMM potential agree when
    gaussians are identical"""

    # Load atoms and build potentials
    atom_positions, atom_identities = read_atoms_from_pdb(
        sample_pdb_path,
        loads_b_factors=False,
        center=True,
        selection_string="not element H",
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )
    atom_potential = PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
    )

    gaussian_variances = atom_potential.scattering_factor_b / (8.0 * jnp.pi**2)
    gaussian_amplitudes = atom_potential.scattering_factor_a

    gmm_potential = GaussianMixtureAtomicPotential(
        atom_positions, gaussian_amplitudes, gaussian_variances
    )

    # Create instrument configuration
    shape = (64, 64)
    pixel_size = 0.5
    config = BasicConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )

    # Compute projections
    integrator = GaussianMixtureProjection(upsampling_factor=1)
    projection_gmm = integrator.integrate(gmm_potential, config)
    projection_peng = integrator.integrate(atom_potential, config)

    np.testing.assert_allclose(projection_gmm, projection_peng)


class TestBuildRealSpaceVoxelsFromAtoms:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        """
        Test that the maxima of the potential are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud

        n_pixels_per_side = n_voxels_per_side[:2]
        ff_a = ff_a.at[largest_atom].add(1.0)
        coordinate_grid = make_coordinate_grid(n_pixels_per_side, voxel_size)

        # Build the potential
        atomic_potential = GaussianMixtureAtomicPotential(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        config = BasicConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the potential integrators
        integrator = GaussianMixtureProjection()
        # Compute projections
        projection = integrator.integrate(atomic_potential, config)
        projection = irfftn(projection)

        # Find the maximum
        maximum_index = jnp.argmax(projection)
        maximum_position = coordinate_grid.reshape(-1, 2)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom][:2])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """
        Test that the maxima of the potential are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud

        n_pixels_per_side = n_voxels_per_side[:2]
        # Build the potential
        atomic_potential = GaussianMixtureAtomicPotential(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        config = BasicConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the potential integrators
        integrator = GaussianMixtureProjection()
        # Compute projections
        projection = integrator.integrate(atomic_potential, config)
        projection = irfftn(projection)

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(4 * jnp.pi * ff_a))
