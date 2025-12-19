import warnings
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax.ndimage as im
    import cryojax.simulator as cxs
    from cryojax.constants import (
        LobatoScatteringFactorParameters,
        PengScatteringFactorParameters,
        check_atomic_numbers_supported,
    )
    from cryojax.io import read_array_from_mrc, read_atoms_from_pdb
    from cryojax.ndimage import make_coordinate_grid

try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


@pytest.fixture
def pdb_info(sample_pdb_path):
    return read_atoms_from_pdb(sample_pdb_path, center=True, loads_properties=True)


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


#
# Test volume tabulations
#
@pytest.mark.parametrize("tabulation", ("peng", "lobato"))
def test_load_atom_volume(tabulation, sample_pdb_path: str):
    import pathlib

    import mmdf

    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.IndependentAtomVolume,
        tabulation=tabulation,
    )
    assert isinstance(atom_volume, cxs.IndependentAtomVolume)
    if tabulation == "peng":
        atom_volume = cxs.load_tabulated_volume(
            sample_pdb_path,
            output_type=cxs.GaussianMixtureVolume,
            tabulation=tabulation,
        )
        assert isinstance(atom_volume, cxs.GaussianMixtureVolume)
    else:
        with pytest.raises(ValueError):
            atom_volume = cxs.load_tabulated_volume(
                sample_pdb_path,
                output_type=cxs.GaussianMixtureVolume,
                tabulation=tabulation,
            )
    atom_data = mmdf.read(pathlib.Path(sample_pdb_path))
    atom_volume = cxs.load_tabulated_volume(
        atom_data,
        output_type=cxs.IndependentAtomVolume,
        tabulation=tabulation,
    )
    assert isinstance(atom_volume, cxs.IndependentAtomVolume)


def test_scattering_factor_parameters_correct(
    peng_parameters_path, lobato_parameters_path
):
    from cryojax.constants._scattering_factor_parameters import _SUPPORTED_ATOMIC_NUMBERS

    atomic_numbers = np.asarray(_SUPPORTED_ATOMIC_NUMBERS)

    # Test Peng
    params = PengScatteringFactorParameters(atomic_numbers)
    a1, b1 = (params.a, params.b)

    peng_table = np.load(peng_parameters_path)
    a2, b2 = peng_table[:, atomic_numbers, :]

    np.testing.assert_equal(a1, a2)
    np.testing.assert_equal(b1, b2)

    # Test lobato
    params = LobatoScatteringFactorParameters(atomic_numbers)
    a1, b1 = (params.a, params.b)

    lobato_table = np.load(lobato_parameters_path)
    a2, b2 = lobato_table[:, atomic_numbers, :]

    np.testing.assert_equal(a1, a2)
    np.testing.assert_equal(b1, b2)


def test_compare_hydrogen_scattering_factor():
    shape, pixel_size = (32, 32), 1.0
    frequencies = im.make_frequency_grid(shape, pixel_size)
    hydrogen_id = np.asarray([1], dtype=int)
    p, l = (
        PengScatteringFactorParameters(hydrogen_id),
        LobatoScatteringFactorParameters(hydrogen_id),
    )
    p_fac = cxs.PengScatteringFactor(p.a[0], p.b[0])
    l_fac = cxs.LobatoScatteringFactor(l.a[0], l.b[0])
    p_arr, l_arr = p_fac(frequencies), l_fac(frequencies)
    np.testing.assert_allclose(p_arr, l_arr, atol=1e-3)


@pytest.mark.parametrize("tabulation", ("peng", "lobato"))
def test_invalid_atomic_numbers(tabulation):
    # Make sure has nan when expected
    bad_nan = np.asarray([2, 6])
    params_nan = _make_scattering_parameters(bad_nan, tabulation)
    assert np.any(np.isnan(params_nan.a))
    assert np.any(np.isnan(params_nan.b))
    # Make sure throws out of bounds error when expected
    bad_oob = np.asarray([1, 31])
    with pytest.raises(IndexError):
        _make_scattering_parameters(bad_oob, tabulation)
    # Make sure error checks work
    with pytest.raises(ValueError):
        check_atomic_numbers_supported(bad_nan)
    with pytest.raises(ValueError):
        check_atomic_numbers_supported(bad_oob)


#
# Test volume rendering
#
def test_render_voxels(sample_pdb_path):
    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
    )
    render_fn = cxs.AutoVolumeRenderFn((16, 16, 16), voxel_size=4.0)
    for cls in [
        cxs.FourierVoxelGridVolume,
        cxs.FourierVoxelSplineVolume,
        cxs.RealVoxelGridVolume,
    ]:
        assert (
            type(cxs.render_voxel_volume(atom_volume, render_fn, output_type=cls)) == cls
        )


def test_voxel_volume_loaders():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    fourier_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    real_volume = cxs.RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)

    assert isinstance(
        fourier_volume.frequency_slice_in_pixels,
        Float[Array, "1 _ _ 3"],  # type: ignore
    )
    assert isinstance(real_volume.coordinate_grid_in_pixels, Float[Array, "_ _ _ 3"])  # type: ignore


@pytest.mark.parametrize("pad_scale", (1, 1.1))
def test_sinc_correction(sample_mrc_path, pad_scale):
    real_voxel_grid = read_array_from_mrc(sample_mrc_path)
    _ = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        real_voxel_grid, sinc_correction=True, pad_scale=pad_scale
    )


def test_fourier_vs_real_agreement(sample_pdb_path):
    """
    Integration test ensuring that the VoxelGrid classes
    produce comparable electron densities when loaded from PDB.
    """
    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.5

    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
        selection_string="not element H",
    )
    render_fn = cxs.AutoVolumeRenderFn(
        n_voxels_per_side,
        voxel_size,
    )
    fourier_volume = cxs.render_voxel_volume(
        atom_volume, render_fn, output_type=cxs.FourierVoxelGridVolume
    )
    real_volume = cxs.render_voxel_volume(
        atom_volume, render_fn, output_type=cxs.RealVoxelGridVolume
    )
    real_voxel_grid = im.ifftn(jnp.fft.ifftshift(fourier_volume.fourier_voxel_grid)).real

    np.testing.assert_allclose(real_voxel_grid, real_volume.real_voxel_grid, atol=1e-12)


def test_downsampled_voxel_volume_agreement(sample_pdb_path):
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
        selection_string="not element H",
    )
    # Load atomistic volume
    atom_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grids
    lowres_render_fn = cxs.GaussianMixtureRenderFn(
        downsampled_shape, downsampled_voxel_size
    )
    low_resolution_volume_grid = lowres_render_fn(atom_volume)
    highres_render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    high_resolution_volume_grid = highres_render_fn(atom_volume)
    downsampled_volume_grid = im.fourier_crop_downsample(
        high_resolution_volume_grid, downsampling_factor
    )

    assert low_resolution_volume_grid.shape == downsampled_volume_grid.shape


def test_render_options(pdb_info):
    width, voxel_size, shape = (1.0, 1.0, (31, 32, 33))
    atom_positions, _, _ = pdb_info
    volumes, render_fns = [], []
    gaussian_volume, gaussian_render_fn = (
        cxs.GaussianMixtureVolume(
            atom_positions,
            amplitudes=1.0,
            variances=width**2,
        ),
        cxs.GaussianMixtureRenderFn(shape, voxel_size),
    )
    volumes.append(gaussian_volume)
    render_fns.append(gaussian_render_fn)
    if jnufft is not None:
        volumes.append(
            cxs.IndependentAtomVolume(
                positions=atom_positions,
                scattering_factors=im.FourierGaussian(
                    amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)
                ),
            )
        )
        render_fns.append(cxs.FFTAtomRenderFn(shape, voxel_size, eps=1e-16))
    for volume, render_fn in zip(volumes, render_fns):
        real_voxel_grid = render_fn(volume, outputs_real_space=True)
        assert real_voxel_grid.shape == shape
        assert not jnp.iscomplexobj(real_voxel_grid)
        fftn_voxel_grid = render_fn(
            volume,
            outputs_real_space=False,
            outputs_rfft=False,
        )
        assert fftn_voxel_grid.shape == shape
        assert jnp.iscomplexobj(fftn_voxel_grid)
        rfftn_voxel_grid = render_fn(
            volume,
            outputs_real_space=False,
            outputs_rfft=True,
        )
        assert rfftn_voxel_grid.shape == (*shape[0:2], shape[2] // 2 + 1)
        assert jnp.iscomplexobj(rfftn_voxel_grid)


@pytest.mark.parametrize(
    "width, voxel_size, shape",
    ((1.0, 0.5, (64, 64, 64)), (1.0, 0.5, (63, 63, 63))),
)
def test_fft_atom_render(pdb_info, width, voxel_size, shape):
    if jnufft is not None:
        atom_positions, _, _ = pdb_info
        gaussian_volume = cxs.GaussianMixtureVolume(
            atom_positions,
            amplitudes=1.0,
            variances=width**2,
        )
        atom_volume = cxs.IndependentAtomVolume(
            positions=atom_positions,
            scattering_factors=im.FourierGaussian(
                amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)
            ),
        )
        gaussian_render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
        fft_render_fn = cxs.FFTAtomRenderFn(shape, voxel_size, eps=1e-16)
        voxels_by_gaussians = gaussian_render_fn(gaussian_volume)
        voxels_by_fft = fft_render_fn(atom_volume)

        np.testing.assert_allclose(voxels_by_gaussians, voxels_by_fft, atol=1e-8)
    else:
        warnings.warn(
            "Could not test rendering method `FFTAtomRenderFn`, "
            "most likely because `jax_finufft` is not installed. "
            f"Error traceback is:\n{JAX_FINUFFT_IMPORT_ERROR}"
        )


#
# TODO: organize
#
@pytest.mark.parametrize("shape", ((128, 127, 126),))
def test_compute_rectangular_voxel_grid(sample_pdb_path, shape):
    voxel_size = 0.5

    # Load the PDB file
    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
    )
    # Load atomistic volume
    atom_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grid
    render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    voxels = render_fn(atom_volume)
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
    # Load atomistic volume
    atom_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    # Build the grid
    render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    voxels = render_fn(atom_volume)
    batched_render_fn = cxs.GaussianMixtureRenderFn(
        shape,
        voxel_size,
        batch_options=dict(batch_size=batch_size, n_batches=n_batches),
    )
    voxels_with_batching = batched_render_fn(atom_volume)
    np.testing.assert_allclose(voxels, voxels_with_batching)


class TestIntegrateGMMToPixels:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        """
        Test that the maxima of the volume are in the correct positions.
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

        # Build the volume
        atomic_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        image_config = cxs.BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the volume integrators
        integrator = cxs.GaussianMixtureProjection()
        # Compute projections
        projection = integrator.integrate(atomic_volume, image_config)
        projection = im.irfftn(projection)

        # Find the maximum
        maximum_index = jnp.argmax(projection)
        maximum_position = coordinate_grid.reshape(-1, 2)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom][:2])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """
        Test that the maxima of the volume are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud

        n_pixels_per_side = n_voxels_per_side[:2]
        # Build the volume
        atomic_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        image_config = cxs.BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        # Build the volume integrators
        integrator = cxs.GaussianMixtureProjection()
        # Compute projections
        projection = integrator.integrate(atomic_volume, image_config)
        projection = im.irfftn(projection)

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(ff_a))


class TestRenderGMMToVoxels:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        """
        Test that the maxima of the volume are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud
        ff_a = ff_a.at[largest_atom].add(1.0)

        # Build the volume
        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size)
        real_voxel_grid = render_fn(gmm_volume)
        coordinate_grid = make_coordinate_grid(n_voxels_per_side, voxel_size)

        # Find the maximum
        maximum_index = jnp.argmax(real_voxel_grid)
        maximum_position = coordinate_grid.reshape(-1, 3)[maximum_index]

        # Check that the maximum is in the correct position
        assert jnp.allclose(maximum_position, atom_positions[largest_atom])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """
        Test that the maxima of the volume are in the correct positions.
        """
        (
            atom_positions,
            ff_a,
            ff_b,
            n_voxels_per_side,
            voxel_size,
        ) = toy_gaussian_cloud

        # Build the volume
        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size)
        real_voxel_grid = render_fn(gmm_volume)

        integral = jnp.sum(real_voxel_grid) * voxel_size**3
        assert jnp.isclose(integral, jnp.sum(ff_a))


def test_gmm_shape():
    n_atoms, n_gaussians = 10, 2
    pos = np.zeros((n_atoms, 3))
    make_gmm = lambda amp, var: cxs.GaussianMixtureVolume(pos, amp, var)
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


def _make_scattering_parameters(
    atomic_numbers: np.ndarray, tabulation: Literal["peng", "lobato"]
):
    return (
        PengScatteringFactorParameters(atomic_numbers)
        if tabulation == "peng"
        else LobatoScatteringFactorParameters(atomic_numbers)
    )
