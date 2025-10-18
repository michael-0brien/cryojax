import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from jaxtyping import Array, Float, install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.coordinates import make_coordinate_grid
    from cryojax.io import read_atoms_from_pdb
    from cryojax.ndimage import downsample_with_fourier_cropping, ifftn, irfftn
    from cryojax.simulator import (
        BasicImageConfig,
        FourierVoxelGridVolume,
        GaussianMixtureProjection,
        GaussianMixtureVolume,
        PengAtomicVolume,
        PengScatteringFactorParameters,
        RealVoxelGridVolume,
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
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
        loads_b_factors=True,
    )
    atom_potential = PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
        extra_b_factors=b_factors,
    )
    pixel_size = 0.5

    integrator = GaussianMixtureProjection(upsampling_factor=2)
    # # ... and the configuration of the imaging instrument
    image_config = BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volumetric_potential
    fourier_integrated_potential = integrator.integrate(
        atom_potential, image_config, outputs_real_space=False
    )

    assert fourier_integrated_potential.shape == (shape[0], shape[1] // 2 + 1)


#
# Test different representations
#
def test_voxel_potential_loaders():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    fourier_potential = FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    real_potential = RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)

    assert isinstance(
        fourier_potential.frequency_slice_in_pixels,
        Float[Array, "1 _ _ 3"],  # type: ignore
    )
    assert isinstance(real_potential.coordinate_grid_in_pixels, Float[Array, "_ _ _ 3"])  # type: ignore


#
# Test rendering
#
def test_fourier_vs_real_voxel_potential_agreement(sample_pdb_path):
    """
    Integration test ensuring that the VoxelGrid classes
    produce comparable electron densities when loaded from PDB.
    """
    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.5

    # Load the PDB file
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        loads_b_factors=False,
        selection_string="not element H",
    )
    # Load atomistic potential
    atom_potential = PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grid
    potential_as_real_voxel_grid = atom_potential.to_real_voxel_grid(
        n_voxels_per_side, voxel_size
    )
    fourier_potential = FourierVoxelGridVolume.from_real_voxel_grid(
        potential_as_real_voxel_grid
    )
    # Since Voxelgrid is in Frequency space by default, we have to first
    # transform back into real space.
    fvg_real = ifftn(jnp.fft.ifftshift(fourier_potential.fourier_voxel_grid)).real

    vg = RealVoxelGridVolume.from_real_voxel_grid(potential_as_real_voxel_grid)

    np.testing.assert_allclose(fvg_real, vg.real_voxel_grid, atol=1e-12)


def test_downsampled_voxel_potential_agreement(sample_pdb_path):
    """Integration test ensuring that rasterized voxel grids roughly
    agree with downsampled versions.
    """
    # Parameters for rasterization
    shape = (128, 128, 128)
    voxel_size = 0.25
    # Downsampling parameters
    downsampling_factor = 2
    downsampled_shape = (
        int(shape[0] / downsampling_factor),
        int(shape[1] / downsampling_factor),
        int(shape[2] / downsampling_factor),
    )
    downsampled_voxel_size = voxel_size * downsampling_factor
    # Load the PDB file
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        loads_b_factors=False,
        selection_string="not element H",
    )
    # Load atomistic potential
    atom_potential = PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grids
    low_resolution_potential_grid = atom_potential.to_real_voxel_grid(
        downsampled_shape, downsampled_voxel_size
    )
    high_resolution_potential_grid = atom_potential.to_real_voxel_grid(shape, voxel_size)
    downsampled_potential_grid = downsample_with_fourier_cropping(
        high_resolution_potential_grid, downsampling_factor
    )

    assert low_resolution_potential_grid.shape == downsampled_potential_grid.shape


#
# TODO: organize
#
def test_downsampled_gmm_potential_agreement(sample_pdb_path):
    """Integration test ensuring that rasterized voxel grids roughly
    agree with downsampled versions.
    """
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        loads_b_factors=False,
        center=True,
        selection_string="not element H",
    )
    atom_potential = PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
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
    image_config = BasicImageConfig(
        shape=downsampled_shape,
        pixel_size=downsampled_pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volumetric_potential
    image_from_hires = integrator_int_hires.integrate(atom_potential, image_config)
    image_lowres = integrator_int_lowres.integrate(atom_potential, image_config)

    assert image_from_hires.shape == image_lowres.shape


@pytest.mark.parametrize("shape", ((128, 127, 126),))
def test_compute_rectangular_voxel_grid(sample_pdb_path, shape):
    voxel_size = 0.5

    # Load the PDB file
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        loads_b_factors=False,
        selection_string="not element H",
    )
    # Load atomistic potential
    atom_potential = PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grid
    voxels = atom_potential.to_real_voxel_grid(shape, voxel_size)
    assert voxels.shape == shape


@pytest.mark.parametrize(
    "batch_size, n_batches",
    ((1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (2, 2)),
)
def test_z_plane_batched_vs_non_batched_loop_agreement(
    sample_pdb_path, batch_size, n_batches
):
    shape = (128, 128, 128)
    voxel_size = 0.5

    # Load the PDB file
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        loads_b_factors=False,
        selection_string="not element H",
    )
    # Load atomistic potential
    atom_potential = PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grid
    voxels = atom_potential.to_real_voxel_grid(shape, voxel_size)
    voxels_with_batching = atom_potential.to_real_voxel_grid(
        shape,
        voxel_size,
        batch_options=dict(batch_size=batch_size, n_batches=n_batches),
    )
    np.testing.assert_allclose(voxels, voxels_with_batching)


class TestIntegrateGMMToPixels:
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
        atomic_potential = GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        image_config = BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the potential integrators
        integrator = GaussianMixtureProjection()
        # Compute projections
        projection = integrator.integrate(atomic_potential, image_config)
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
        atomic_potential = GaussianMixtureVolume(
            atom_positions, 4 * jnp.pi * ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        image_config = BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the potential integrators
        integrator = GaussianMixtureProjection()
        # Compute projections
        projection = integrator.integrate(atomic_potential, image_config)
        projection = irfftn(projection)

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(4 * jnp.pi * ff_a))


class TestRenderGMMToVoxels:
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
        ff_a = ff_a.at[largest_atom].add(1.0)

        # Build the potential
        gmm_volume = GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8 * jnp.pi**2))
        real_voxel_grid = gmm_volume.to_real_voxel_grid(n_voxels_per_side, voxel_size)
        coordinate_grid = make_coordinate_grid(n_voxels_per_side, voxel_size)

        # Find the maximum
        maximum_index = jnp.argmax(real_voxel_grid)
        maximum_position = coordinate_grid.reshape(-1, 3)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom])

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

        # Build the potential
        gmm_volume = GaussianMixtureVolume(
            atom_positions, 4 * jnp.pi * ff_a, ff_b / (8 * jnp.pi**2)
        )
        real_voxel_grid = gmm_volume.to_real_voxel_grid(n_voxels_per_side, voxel_size)

        integral = jnp.sum(real_voxel_grid) * voxel_size**3
        assert jnp.isclose(integral, jnp.sum(4 * jnp.pi * ff_a))


def test_gmm_shape():
    n_atoms, n_gaussians = 10, 2
    pos = np.zeros((n_atoms, 3))
    make_gmm = lambda amp, var: GaussianMixtureVolume(pos, amp, var)
    gmm = make_gmm(1.0, 1.0)
    assert gmm.variances.shape == gmm.amplitudes.shape == (n_atoms, 1)
    gmm = make_gmm(np.ones((n_atoms,)), np.ones((n_atoms,)))
    assert gmm.variances.shape == gmm.amplitudes.shape == (n_atoms, 1)
    gmm = make_gmm(np.ones((n_atoms, n_gaussians)), np.ones((n_atoms, n_gaussians)))
    assert gmm.variances.shape == gmm.amplitudes.shape == (n_atoms, n_gaussians)
    gmm1, gmm2 = make_gmm(1.0, np.ones((n_atoms,))), make_gmm(np.ones((n_atoms,)), 1.0)
    assert (
        gmm1.variances.shape
        == gmm1.amplitudes.shape
        == gmm2.variances.shape
        == gmm2.amplitudes.shape
        == (n_atoms, 1)
    )
    gmm1, gmm2 = (
        make_gmm(1.0, np.ones((n_atoms, n_gaussians))),
        make_gmm(np.ones((n_atoms, n_gaussians)), 1.0),
    )
    assert (
        gmm1.variances.shape
        == gmm1.amplitudes.shape
        == gmm2.variances.shape
        == gmm2.amplitudes.shape
        == (n_atoms, n_gaussians)
    )
    gmm1, gmm2 = (
        make_gmm(np.asarray(1.0), np.ones((n_atoms, n_gaussians))),
        make_gmm(np.ones((n_atoms, n_gaussians)), np.asarray(1.0)),
    )
    assert (
        gmm1.variances.shape
        == gmm1.amplitudes.shape
        == gmm2.variances.shape
        == gmm2.amplitudes.shape
        == (n_atoms, n_gaussians)
    )
    gmm1, gmm2 = (
        make_gmm(np.ones((n_atoms,)), np.ones((n_atoms, n_gaussians))),
        make_gmm(np.ones((n_atoms, n_gaussians)), np.ones((n_atoms,))),
    )
    assert (
        gmm1.variances.shape
        == gmm1.amplitudes.shape
        == gmm2.variances.shape
        == gmm2.amplitudes.shape
        == (n_atoms, n_gaussians)
    )
