"""Tests for atom-based volume representations and their integrators/render fns.

Covers GaussianMixtureVolume and IndependentAtomVolume, with their associated
AbstractVolumeIntegrators (GaussianMixtureProjection, IndependentAtomProjection)
and AbstractVolumeRenderFns (GaussianMixtureRenderFn, IndependentAtomRenderFn).

Tests demand precise numerical agreement between implementations and between
the two NUFFT backends (nufftax and jax-finufft).
"""

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.atom_util import split_atoms_by_element
from cryojax.constants import (
    PengScatteringFactorParameters,
    check_atomic_numbers_supported,
)
from cryojax.ndimage import make_coordinate_grid
from jaxtyping import Array


try:
    import jax_finufft as jnufft
except ModuleNotFoundError:
    jnufft = None

_jax_finufft = pytest.mark.skipif(jnufft is None, reason="jax-finufft not installed")
_backends = pytest.mark.parametrize(
    "backend", ["nufftax", pytest.param("jax-finufft", marks=_jax_finufft)]
)


# ── fixtures ──────────────────────────────────────────────────────────────────


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
    ff_a = jnp.array(num_atoms * [[1.0, 0.5]])
    ff_b = jnp.array(num_atoms * [[0.3, 0.2]])
    n_voxels_per_side = (128, 128, 128)
    voxel_size = 0.05
    return (atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size)


# ── JIT helpers ───────────────────────────────────────────────────────────────


@eqx.filter_jit
def compute_projection(
    volume: cxs.AbstractVolumeRepresentation,
    integrator: cxs.AbstractVolumeIntegrator,
    image_config: cxs.BasicImageConfig,
) -> Array:
    fourier_projection = integrator.integrate(
        volume, image_config, outputs_real_space=False
    )
    return im.crop_to_shape(
        im.irfftn(fourier_projection, s=image_config.padded_shape),
        image_config.shape,
    )


@eqx.filter_jit
def compute_projection_at_pose(
    volume: cxs.AbstractVolumeRepresentation,
    integrator: cxs.AbstractVolumeIntegrator,
    pose: cxs.AbstractPose,
    image_config: cxs.BasicImageConfig,
) -> Array:
    rotated_volume = volume.rotate_to_pose(pose)
    fourier_projection = integrator.integrate(
        rotated_volume, image_config, outputs_real_space=False
    )
    translation_operator = pose.compute_translation_operator(
        image_config.padded_shape,
        image_config.pixel_size,
    )
    return im.crop_to_shape(
        im.irfftn(
            pose.translate_image(
                fourier_projection,
                translation_operator,
                image_config.padded_shape,
            ),
            s=image_config.padded_shape,
        ),
        image_config.shape,
    )


# ── tabulated volume loading ──────────────────────────────────────────────────


@pytest.mark.parametrize("tabulation", ("peng",))
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
            cxs.load_tabulated_volume(
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


# ── scattering factor parameters ──────────────────────────────────────────────


def test_scattering_factor_parameters_correct(peng_parameters_path):
    from cryojax.constants._scattering_factor_parameters import _SUPPORTED_ATOMIC_NUMBERS

    atomic_numbers = np.asarray(_SUPPORTED_ATOMIC_NUMBERS)

    params = PengScatteringFactorParameters(atomic_numbers)
    a1, b1 = params.a, params.b
    peng_table = np.load(peng_parameters_path)
    a2, b2 = peng_table[:, atomic_numbers, :]
    np.testing.assert_equal(a1, a2)
    np.testing.assert_equal(b1, b2)


def test_invalid_atomic_numbers():
    bad_nan = np.asarray([2, 6])
    params_nan = PengScatteringFactorParameters(bad_nan)
    assert np.any(np.isnan(params_nan.a))
    assert np.any(np.isnan(params_nan.b))
    bad_oob = np.asarray([1, 31])
    with pytest.raises(IndexError):
        PengScatteringFactorParameters(bad_oob)
    with pytest.raises(ValueError):
        check_atomic_numbers_supported(bad_nan)
    with pytest.raises(ValueError):
        check_atomic_numbers_supported(bad_oob)


# ── GaussianMixtureVolume shape ───────────────────────────────────────────────


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
    gmm1, gmm2 = (
        make_gmm(1.0, np.ones((n_atoms,))),
        make_gmm(np.ones((n_atoms,)), 1.0),
    )
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


# ── GaussianMixtureProjection integrator ─────────────────────────────────────


@pytest.mark.parametrize("shape", ((64, 64), (63, 63), (63, 64), (64, 63)))
def test_gmm_integrator_shape(sample_pdb_path, shape):
    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
        include_b_factors=True,
        selection_string="not element H",
    )
    pixel_size = 0.5
    integrator = cxs.GaussianMixtureProjection(shape=(2 * shape[0], 2 * shape[1]))
    image_config = cxs.BasicImageConfig(
        shape=shape, pixel_size=pixel_size, voltage_in_kilovolts=300.0
    )
    result = integrator.integrate(atom_volume, image_config, outputs_real_space=False)
    assert result.shape == (shape[0], shape[1] // 2 + 1)


class TestIntegrateGMMToPixels:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        """Maxima of the projection must be at the correct atom positions."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        n_pixels_per_side = n_voxels_per_side[:2]
        ff_a = ff_a.at[largest_atom].add(1.0)
        coordinate_grid = make_coordinate_grid(n_pixels_per_side, voxel_size)

        atomic_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        image_config = cxs.BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        integrator = cxs.GaussianMixtureProjection()
        projection = im.irfftn(integrator.integrate(atomic_volume, image_config))

        maximum_index = jnp.argmax(projection)
        maximum_position = coordinate_grid.reshape(-1, 2)[maximum_index]
        assert jnp.allclose(maximum_position, atom_positions[largest_atom][:2])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """Integral of the projection must equal the sum of amplitudes."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        n_pixels_per_side = n_voxels_per_side[:2]

        atomic_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        image_config = cxs.BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        integrator = cxs.GaussianMixtureProjection()
        projection = im.irfftn(integrator.integrate(atomic_volume, image_config))

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(ff_a))


# ── GaussianMixtureRenderFn ───────────────────────────────────────────────────


class TestRenderGMMToVoxels:
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        """Maxima of the rendered voxel grid must be at the correct positions."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        ff_a = ff_a.at[largest_atom].add(1.0)

        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size)
        real_voxel_grid = render_fn(gmm_volume)
        coordinate_grid = make_coordinate_grid(n_voxels_per_side, voxel_size)

        maximum_index = jnp.argmax(real_voxel_grid)
        maximum_position = coordinate_grid.reshape(-1, 3)[maximum_index]
        assert jnp.allclose(maximum_position, atom_positions[largest_atom])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        """Integral of the rendered voxel grid must equal the sum of amplitudes."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud

        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size)
        real_voxel_grid = render_fn(gmm_volume)

        integral = jnp.sum(real_voxel_grid) * voxel_size**3
        assert jnp.isclose(integral, jnp.sum(ff_a))


# ── IndependentAtomVolume: bad instantiation ──────────────────────────────────


def test_fft_atom_bad_instantiation():
    with pytest.raises(ValueError):
        _ = cxs.IndependentAtomVolume(
            positions=np.zeros((10, 3)),
            kernel_fns=(im.FourierGaussian(),),
        )


# ── IndependentAtomProjection: each backend vs GaussianMixtureProjection ─────


def _make_fft_projection_exact_volumes(pdb_info):
    atom_positions, _, _ = pdb_info
    pixel_size, shape = 0.5, (64, 64)
    image_config = cxs.BasicImageConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, padded_shape=(128, 128)
    )
    amplitude, b_factor = 1.0, 100.0
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions,
        amplitudes=amplitude,
        variances=b_factor / (8 * np.pi**2),
    )
    gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="point")
    atom_volume = cxs.IndependentAtomVolume(
        positions=atom_positions,
        kernel_fns=im.FourierGaussian(amplitude=amplitude, b_factor=b_factor),
    )
    return gaussian_volume, gaussian_integrator, atom_volume, image_config


@pytest.mark.parametrize(
    "pixel_size, shape",
    ((1.0, (32, 32)), (1.0, (32, 31)), (1.0, (31, 32)), (1.0, (31, 31))),
)
@_backends
def test_fft_atom_projection_exact(pdb_info, pixel_size, shape, backend):
    gaussian_volume, gaussian_integrator, atom_volume, image_config = (
        _make_fft_projection_exact_volumes(pdb_info)
    )
    atom_integrator = cxs.IndependentAtomProjection(
        backend=backend,  # type: ignore
        sampling_mode="point",
        eps=1e-10,
    )
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atom = compute_projection(atom_volume, atom_integrator, image_config)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_atom, atol=1e-8)


@pytest.mark.skipif(jnufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize(
    "pixel_size, shape",
    ((1.0, (32, 32)), (1.0, (32, 31)), (1.0, (31, 32)), (1.0, (31, 31))),
)
def test_fft_atom_projection_exact_backends_agree(pdb_info, pixel_size, shape):
    """nufftax and jax-finufft must produce identical projections."""
    _, _, atom_volume, image_config = _make_fft_projection_exact_volumes(pdb_info)
    proj_nufftax = compute_projection(
        atom_volume,
        cxs.IndependentAtomProjection(
            backend="nufftax", sampling_mode="point", eps=1e-10
        ),
        image_config,
    )
    proj_jax_finufft = compute_projection(
        atom_volume,
        cxs.IndependentAtomProjection(
            backend="jax-finufft", sampling_mode="point", eps=1e-10
        ),
        image_config,
    )
    np.testing.assert_allclose(proj_nufftax, proj_jax_finufft, atol=1e-6)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (0.5, (63, 63)),
        (0.5, (64, 64)),
    ),
)
def test_real_atom_projection_exact(pdb_info, pixel_size, shape):
    atom_positions, _, _ = pdb_info
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    amplitude, variance = 0.75, 20.0 / (8 * np.pi**2)
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions, amplitudes=amplitude, variances=variance
    )
    gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    atom_volume = cxs.IndependentAtomVolume(
        positions=atom_positions,
        kernel_fns=im.RealGaussian(amplitude=amplitude, variance=variance),
    )
    atom_integrator = cxs.IndependentAtomProjection(sampling_mode="average", eps=1e-12)
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atom = compute_projection(atom_volume, atom_integrator, image_config)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_atom, atol=1e-5)


def _make_antialias_volumes(pdb_info, width, pixel_size, shape):
    atom_positions, _, _ = pdb_info
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions, amplitudes=1.0, variances=width**2
    )
    atom_volume = cxs.IndependentAtomVolume(
        positions=atom_positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)),
    )
    gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    padded_shape = (2 * shape[0], 2 * shape[1])
    image_config = cxs.BasicImageConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, padded_shape=padded_shape
    )
    return gaussian_volume, gaussian_integrator, atom_volume, image_config


@pytest.mark.parametrize(
    "width, pixel_size, shape",
    ((5.0, 0.5, (64, 64)), (1.0, 0.5, (64, 64)), (2.0, 1.0, (32, 32))),
)
@_backends
def test_fft_atom_projection_antialias(pdb_info, width, pixel_size, shape, backend):
    gaussian_volume, gaussian_integrator, atom_volume, image_config = (
        _make_antialias_volumes(pdb_info, width, pixel_size, shape)
    )
    atom_integrator = cxs.IndependentAtomProjection(
        eps=1e-10,
        backend=backend,  # type: ignore
        upsample_factor=2.0,
    )
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atoms = compute_projection(atom_volume, atom_integrator, image_config)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_atoms, atol=1e-8)


@pytest.mark.skipif(jnufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize(
    "width, pixel_size, shape",
    ((5.0, 0.5, (64, 64)), (1.0, 0.5, (64, 64)), (2.0, 1.0, (32, 32))),
)
def test_fft_atom_projection_antialias_backends_agree(pdb_info, width, pixel_size, shape):
    """nufftax and jax-finufft must produce identical antialiased projections."""
    _, _, atom_volume, image_config = _make_antialias_volumes(
        pdb_info, width, pixel_size, shape
    )
    proj_nufftax = compute_projection(
        atom_volume,
        cxs.IndependentAtomProjection(backend="nufftax", eps=1e-10, upsample_factor=2.0),
        image_config,
    )
    proj_jax_finufft = compute_projection(
        atom_volume,
        cxs.IndependentAtomProjection(
            backend="jax-finufft", eps=1e-10, upsample_factor=2.0
        ),
        image_config,
    )
    np.testing.assert_allclose(proj_nufftax, proj_jax_finufft, atol=1e-6)


def _make_peng_volumes(pdb_info, upsampfac):
    atom_positions, atom_ids, _ = pdb_info
    positions_by_id, unique_atom_ids = split_atoms_by_element(atom_ids, atom_positions)
    peng_parameters = PengScatteringFactorParameters(atom_ids)
    peng_parameters_by_id = PengScatteringFactorParameters(unique_atom_ids)
    gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions, peng_parameters, extra_b_factors=4.0
    )
    atom_volume = cxs.IndependentAtomVolume.from_tabulated_parameters(
        positions_by_id,
        peng_parameters_by_id,
        b_factor_by_element=4.0,
        use_real_space=False,
    )
    gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    return gaussian_volume, gaussian_integrator, atom_volume


@pytest.mark.parametrize(
    "pixel_size, shape, upsampfac, eps",
    (
        (0.25, (134, 134), 2, 1e-5),
        (0.25, (133, 133), 2, 1e-5),
        (0.25, (134, 133), 2, 1e-5),
        (0.25, (133, 134), 2, 1e-5),
        (0.25, (134, 134), 3, 1e-5),
        (0.25, (133, 133), 3, 1e-5),
        (0.25, (134, 133), 3.0, 1e-5),
        (0.25, (133, 134), 3.0, 1e-5),
        (0.25, (134, 134), 2, 1e-6),
        (0.25, (133, 133), 2, 1e-6),
        (0.25, (134, 133), 2, 1e-6),
        (0.25, (133, 134), 2, 1e-6),
        (0.25, (134, 134), 3, 1e-6),
        (0.25, (133, 133), 3, 1e-6),
        (0.25, (134, 133), 3, 1e-6),
        (0.25, (133, 134), 3, 1e-6),
    ),
)
@_backends
def test_fft_projection_peng(pdb_info, pixel_size, shape, upsampfac, eps, backend):
    gaussian_volume, gaussian_integrator, atom_volume = _make_peng_volumes(
        pdb_info, upsampfac
    )
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_integrator = cxs.IndependentAtomProjection(
        backend=backend,  # type: ignore
        sampling_mode="average",
        upsample_factor=upsampfac,
        eps=1e-10,
    )
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atoms = compute_projection(atom_volume, atom_integrator, image_config)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_atoms, atol=5e-3)


@pytest.mark.skipif(jnufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize(
    "pixel_size, shape, upsampfac",
    (
        (0.25, (134, 134), 2),
        (0.25, (133, 133), 2),
    ),
)
def test_fft_projection_peng_backends_agree(pdb_info, pixel_size, shape, upsampfac):
    """nufftax and jax-finufft must produce identical Peng-tabulated projections."""
    _, _, atom_volume = _make_peng_volumes(pdb_info, upsampfac)
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    proj_nufftax = compute_projection(
        atom_volume,
        cxs.IndependentAtomProjection(
            backend="nufftax",
            sampling_mode="average",
            upsample_factor=upsampfac,
            eps=1e-10,
        ),
        image_config,
    )
    proj_jax_finufft = compute_projection(
        atom_volume,
        cxs.IndependentAtomProjection(
            backend="jax-finufft",
            sampling_mode="average",
            upsample_factor=upsampfac,
            eps=1e-10,
        ),
        image_config,
    )
    np.testing.assert_allclose(proj_nufftax, proj_jax_finufft, atol=1e-6)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (0.5, (64, 64)),
        (0.5, (63, 63)),
        (0.5, (64, 63)),
        (0.5, (63, 64)),
    ),
)
def test_real_projection_peng_erf(pdb_info, pixel_size, shape):
    atom_positions, atom_ids, _ = pdb_info
    positions_by_id, unique_atom_ids = split_atoms_by_element(atom_ids, atom_positions)
    peng_parameters = PengScatteringFactorParameters(atom_ids)
    peng_parameters_by_id = PengScatteringFactorParameters(unique_atom_ids)
    gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions, peng_parameters
    )
    atom_volume = cxs.IndependentAtomVolume.from_tabulated_parameters(
        positions_by_id, peng_parameters_by_id, use_real_space=True
    )
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    atom_integrator = cxs.IndependentAtomProjection(sampling_mode="average", eps=1e-16)
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atoms = compute_projection(atom_volume, atom_integrator, image_config)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_atoms, atol=1e-6)


@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_projection_custom_upsampfac(upsampfac):
    """Smoke test: custom upsampfac via options (nufftax) runs without error."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, pixel_size = (8, 8), 1.0
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_volume = cxs.IndependentAtomVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    atom_integrator = cxs.IndependentAtomProjection(
        backend="nufftax",
        sampling_mode="point",
        eps=1e-6,
        options={"upsampfac": upsampfac},
    )
    result = atom_integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )
    assert result.shape == (shape[0], shape[1] // 2 + 1)


@_jax_finufft
@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_projection_custom_upsampfac_jax_finufft(upsampfac):
    """Smoke test: custom opts via options (jax-finufft) runs without error."""
    from jax_finufft.options import NestedOpts, Opts

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, pixel_size = (8, 8), 1.0
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_volume = cxs.IndependentAtomVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    opts = NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )
    atom_integrator = cxs.IndependentAtomProjection(
        backend="jax-finufft",
        sampling_mode="point",
        eps=1e-6,
        options={"opts": opts},
    )
    result = atom_integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )
    assert result.shape == (shape[0], shape[1] // 2 + 1)


# ── IndependentAtomRenderFn: each backend vs GaussianMixtureRenderFn ──────────


def _make_render_volumes(pdb_info, width):
    atom_positions, _, _ = pdb_info
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions, amplitudes=1.0, variances=width**2
    )
    atom_volume = cxs.IndependentAtomVolume(
        positions=atom_positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)),
    )
    return gaussian_volume, atom_volume


@pytest.mark.parametrize(
    "width, voxel_size, shape",
    (
        (1.0, 0.5, (64, 64, 64)),
        (1.0, 0.5, (63, 63, 63)),
    ),
)
@_backends
def test_fft_atom_render(pdb_info, width, voxel_size, shape, backend):
    gaussian_volume, atom_volume = _make_render_volumes(pdb_info, width)
    gaussian_render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    atom_render_fn = cxs.IndependentAtomRenderFn(
        shape,
        voxel_size,
        eps=1e-10,
        backend=backend,  # type: ignore
    )
    voxels_by_gaussians = gaussian_render_fn(gaussian_volume)
    voxels_by_atoms = atom_render_fn(atom_volume)
    np.testing.assert_allclose(voxels_by_gaussians, voxels_by_atoms, atol=1e-8)


@pytest.mark.skipif(jnufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize(
    "width, voxel_size, shape",
    (
        (1.0, 0.5, (64, 64, 64)),
        (1.0, 0.5, (63, 63, 63)),
    ),
)
def test_fft_atom_render_backends_agree(pdb_info, width, voxel_size, shape):
    """nufftax and jax-finufft must produce identical rendered voxel grids."""
    _, atom_volume = _make_render_volumes(pdb_info, width)
    voxels_nufftax = cxs.IndependentAtomRenderFn(
        shape, voxel_size, eps=1e-10, backend="nufftax"
    )(atom_volume)
    voxels_jax_finufft = cxs.IndependentAtomRenderFn(
        shape, voxel_size, eps=1e-10, backend="jax-finufft"
    )(atom_volume)
    np.testing.assert_allclose(voxels_nufftax, voxels_jax_finufft, atol=1e-6)


def test_render_options(pdb_info):
    width, voxel_size, shape = (1.0, 1.0, (31, 32, 33))
    atom_positions, _, _ = pdb_info
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions, amplitudes=1.0, variances=width**2
    )
    gaussian_render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    volumes, render_fns = [gaussian_volume], [gaussian_render_fn]
    if jnufft is not None:
        atom_volume = cxs.IndependentAtomVolume(
            positions=atom_positions,
            kernel_fns=im.FourierGaussian(
                amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)
            ),
        )
        volumes.append(atom_volume)  # type: ignore
        render_fns.append(cxs.IndependentAtomRenderFn(shape, voxel_size, eps=1e-10))  # type: ignore
    for volume, render_fn in zip(volumes, render_fns):
        real_voxel_grid = render_fn(volume, outputs_real_space=True)
        assert real_voxel_grid.shape == shape
        assert not jnp.iscomplexobj(real_voxel_grid)
        fftn_voxel_grid = render_fn(volume, outputs_real_space=False, outputs_rfft=False)
        assert fftn_voxel_grid.shape == shape
        assert jnp.iscomplexobj(fftn_voxel_grid)
        rfftn_voxel_grid = render_fn(volume, outputs_real_space=False, outputs_rfft=True)
        assert rfftn_voxel_grid.shape == (*shape[0:2], shape[2] // 2 + 1)
        assert jnp.iscomplexobj(rfftn_voxel_grid)


@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_render_custom_upsampfac(upsampfac):
    """Smoke test: custom upsampfac via options (nufftax) runs without error."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, voxel_size = (8, 8, 8), 1.0
    atom_volume = cxs.IndependentAtomVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    render_fn = cxs.IndependentAtomRenderFn(
        shape,
        voxel_size,
        backend="nufftax",
        eps=1e-6,
        options={"upsampfac": upsampfac},
    )
    result = render_fn(atom_volume, outputs_real_space=True)
    assert result.shape == shape


@_jax_finufft
@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_render_custom_upsampfac_jax_finufft(upsampfac):
    """Smoke test: custom opts via options (jax-finufft) runs without error."""
    from jax_finufft.options import NestedOpts, Opts

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, voxel_size = (8, 8, 8), 1.0
    atom_volume = cxs.IndependentAtomVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    opts = NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )
    render_fn = cxs.IndependentAtomRenderFn(
        shape, voxel_size, backend="jax-finufft", eps=1e-6, options={"opts": opts}
    )
    result = render_fn(atom_volume, outputs_real_space=True)
    assert result.shape == shape
