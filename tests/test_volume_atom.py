"""Tests for atom-based volume representations and their integrators/render fns.

Covers GaussianMixtureVolume and GaussianFourierVolume, with their associated
AbstractVolumeIntegrators (GaussianMixtureProjection, GaussianFourierProjection)
and AbstractVolumeRenderFns (GaussianMixtureRenderFn, GaussianFourierRenderFn).

Tests demand precise numerical agreement between implementations and between
the two NUFFT backends (nufftax and jax-finufft).
"""

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.atom_util import split_atoms_by_element
from cryojax.constants import (
    PengScatteringFactorParameters,
    check_atomic_numbers_supported,
)
from cryojax.ndimage import _nufft, make_coordinate_grid
from cryojax.simulator._volume.internal import make_frequencies_1d
from jaxtyping import Array


# `tests/conftest.py` also does this, but the precise numerical agreement
# checks below need float64 even if this module is imported/run standalone,
# without pytest loading `conftest.py` first.
jax.config.update("jax_enable_x64", True)


try:
    import jax_finufft as jnufft
except ModuleNotFoundError:
    jnufft = None

_skip_without_jax_finufft = pytest.mark.skipif(
    jnufft is None, reason="jax-finufft not installed"
)
_backends = pytest.mark.parametrize(
    "backend", ["nufftax", pytest.param("jax-finufft", marks=_skip_without_jax_finufft)]
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
        jnp.fft.irfftn(fourier_projection, s=image_config.padded_shape),
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
    return im.crop_to_shape(
        jnp.fft.irfftn(
            pose.translate_fft(
                fourier_projection,
                image_config.padded_shape,
                image_config.pixel_size,
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
        output_type=cxs.GaussianFourierVolume,
        tabulation=tabulation,
    )
    assert isinstance(atom_volume, cxs.GaussianFourierVolume)
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
        output_type=cxs.GaussianFourierVolume,
        tabulation=tabulation,
    )
    assert isinstance(atom_volume, cxs.GaussianFourierVolume)


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
        projection = jnp.fft.irfftn(integrator.integrate(atomic_volume, image_config))

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
        projection = jnp.fft.irfftn(integrator.integrate(atomic_volume, image_config))

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(ff_a))


class TestIntegrateGMMToPixelsWithSpreadingBackend:
    """Same checks as `TestIntegrateGMMToPixels`, but for `n_spread is not None`,
    which routes through the `common.spread_2d` spreading backend instead of
    the dense gaussian-integral backend.
    """

    @pytest.mark.parametrize("sampling_mode", ["average", "point"])
    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(
        self, toy_gaussian_cloud, largest_atom, sampling_mode
    ):
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
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode=sampling_mode,
            # `point` spreads on a twice-finer grid
            n_spread=11 if sampling_mode == "average" else 21,
        )
        projection = jnp.fft.irfftn(integrator.integrate(atomic_volume, image_config))

        maximum_index = jnp.argmax(projection)
        maximum_position = coordinate_grid.reshape(-1, 2)[maximum_index]
        assert jnp.allclose(maximum_position, atom_positions[largest_atom][:2])

    @pytest.mark.parametrize("sampling_mode", ["average", "point"])
    def test_integral_is_correct(self, toy_gaussian_cloud, sampling_mode):
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
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode=sampling_mode,
            # `point` spreads on a twice-finer grid
            n_spread=11 if sampling_mode == "average" else 21,
        )
        projection = jnp.fft.irfftn(integrator.integrate(atomic_volume, image_config))

        integral = jnp.sum(projection) * voxel_size**2
        assert jnp.isclose(integral, jnp.sum(ff_a), atol=1e-4)

    @pytest.mark.parametrize("sampling_mode", ["average", "point"])
    def test_agrees_with_dense_backend(self, toy_gaussian_cloud, sampling_mode):
        """The spreading backend (`n_spread` set) should closely agree with the
        dense gaussian-integral backend (`n_spread=None`) for a high `n_spread`.
        """
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
        dense_integrator = cxs.GaussianMixtureProjection(sampling_mode=sampling_mode)
        spread_integrator = cxs.GaussianMixtureProjection(
            sampling_mode=sampling_mode,
            # `point` spreads on a twice-finer grid
            n_spread=13 if sampling_mode == "average" else 25,
        )
        dense_projection = dense_integrator.integrate(
            atomic_volume, image_config, outputs_real_space=True
        )
        spread_projection = spread_integrator.integrate(
            atomic_volume, image_config, outputs_real_space=True
        )

        assert jnp.allclose(dense_projection, spread_projection, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("sampling_mode", ["average", "point"])
    def test_gradients_are_finite(self, toy_gaussian_cloud, sampling_mode):
        """Gradients through the spreading backend w.r.t. positions,
        amplitudes, and variances must be finite (regression check for the
        custom VJP rule)."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        n_pixels_per_side = n_voxels_per_side[:2]
        image_config = cxs.BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode=sampling_mode,
            # `point` spreads on a twice-finer grid
            n_spread=11 if sampling_mode == "average" else 21,
        )

        def loss(positions, amplitudes, variances):
            volume = cxs.GaussianMixtureVolume(positions, amplitudes, variances)
            projection = integrator.integrate(volume, image_config)
            return jnp.sum(jnp.abs(projection) ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2))(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        for g in grads:
            assert jnp.all(jnp.isfinite(g))

    @pytest.mark.parametrize("sampling_mode", ["average", "point"])
    def test_gradients_agree_with_dense_backend(self, toy_gaussian_cloud, sampling_mode):
        """Gradients through the spreading backend (`n_spread` set) should
        closely agree with gradients through the dense gaussian-integral
        backend (`n_spread=None`) for a high `n_spread`."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        n_pixels_per_side = n_voxels_per_side[:2]
        image_config = cxs.BasicImageConfig(
            shape=n_pixels_per_side,
            pixel_size=voxel_size,
            voltage_in_kilovolts=300.0,
        )
        dense_integrator = cxs.GaussianMixtureProjection(sampling_mode=sampling_mode)
        spread_integrator = cxs.GaussianMixtureProjection(
            sampling_mode=sampling_mode,
            # `point` spreads on a twice-finer grid
            n_spread=13 if sampling_mode == "average" else 25,
        )

        def make_loss(integrator):
            def loss(positions, amplitudes, variances):
                volume = cxs.GaussianMixtureVolume(positions, amplitudes, variances)
                projection = integrator.integrate(volume, image_config)
                return jnp.sum(jnp.abs(projection) ** 2)

            return loss

        args = (atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
        dense_grads = jax.grad(make_loss(dense_integrator), argnums=(0, 1, 2))(*args)
        spread_grads = jax.grad(make_loss(spread_integrator), argnums=(0, 1, 2))(*args)
        for g_dense, g_spread in zip(dense_grads, spread_grads):
            assert jnp.allclose(g_dense, g_spread, atol=1e-4, rtol=1e-4)


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


class TestRenderGMMToVoxelsWithSpreadingBackend:
    """Same checks as `TestRenderGMMToVoxels`, but for `n_spread is not None`, which
    routes through the `common.spread_3d` spreading backend instead of the
    dense gaussian-integral backend.
    """

    @pytest.mark.parametrize("largest_atom", range(0, 3))
    def test_maxima_are_in_right_positions(self, toy_gaussian_cloud, largest_atom):
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        ff_a = ff_a.at[largest_atom].add(1.0)

        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        render_fn = cxs.GaussianMixtureRenderFn(
            n_voxels_per_side, voxel_size, n_spread=11
        )
        real_voxel_grid = render_fn(gmm_volume)
        coordinate_grid = make_coordinate_grid(n_voxels_per_side, voxel_size)

        maximum_index = jnp.argmax(real_voxel_grid)
        maximum_position = coordinate_grid.reshape(-1, 3)[maximum_index]
        assert jnp.allclose(maximum_position, atom_positions[largest_atom])

    def test_integral_is_correct(self, toy_gaussian_cloud):
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud

        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        render_fn = cxs.GaussianMixtureRenderFn(
            n_voxels_per_side, voxel_size, n_spread=11
        )
        real_voxel_grid = render_fn(gmm_volume)

        integral = jnp.sum(real_voxel_grid) * voxel_size**3
        assert jnp.isclose(integral, jnp.sum(ff_a), atol=1e-4)

    def test_agrees_with_dense_backend(self, toy_gaussian_cloud):
        """The spreading backend (`n_spread` set) should closely agree with the
        dense gaussian-integral backend (`n_spread=None`) for a high `n_spread`.
        """
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud

        gmm_volume = cxs.GaussianMixtureVolume(
            atom_positions, ff_a, ff_b / (8 * jnp.pi**2)
        )
        dense_render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size)
        spread_render_fn = cxs.GaussianMixtureRenderFn(
            n_voxels_per_side, voxel_size, n_spread=13
        )
        dense_voxel_grid = dense_render_fn(gmm_volume)
        spread_voxel_grid = spread_render_fn(gmm_volume)

        assert jnp.allclose(dense_voxel_grid, spread_voxel_grid, atol=1e-4, rtol=1e-4)

    def test_gradients_are_finite(self, toy_gaussian_cloud):
        """Gradients through the spreading backend w.r.t. positions,
        amplitudes, and variances must be finite (regression check for the
        custom VJP rule)."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        render_fn = cxs.GaussianMixtureRenderFn(
            n_voxels_per_side, voxel_size, n_spread=11
        )

        def loss(positions, amplitudes, variances):
            volume = cxs.GaussianMixtureVolume(positions, amplitudes, variances)
            real_voxel_grid = render_fn(volume)
            return jnp.sum(real_voxel_grid**2)

        grads = jax.grad(loss, argnums=(0, 1, 2))(
            atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
        )
        for g in grads:
            assert jnp.all(jnp.isfinite(g))

    def test_gradients_agree_with_dense_backend(self, toy_gaussian_cloud):
        """Gradients through the spreading backend (`n_spread` set) should
        closely agree with gradients through the dense gaussian-integral
        backend (`n_spread=None`) for a high `n_spread`."""
        atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
        dense_render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size)
        spread_render_fn = cxs.GaussianMixtureRenderFn(
            n_voxels_per_side, voxel_size, n_spread=13
        )

        def make_loss(render_fn):
            def loss(positions, amplitudes, variances):
                volume = cxs.GaussianMixtureVolume(positions, amplitudes, variances)
                real_voxel_grid = render_fn(volume)
                return jnp.sum(real_voxel_grid**2)

            return loss

        args = (atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
        dense_grads = jax.grad(make_loss(dense_render_fn), argnums=(0, 1, 2))(*args)
        spread_grads = jax.grad(make_loss(spread_render_fn), argnums=(0, 1, 2))(*args)
        for g_dense, g_spread in zip(dense_grads, spread_grads):
            assert jnp.allclose(g_dense, g_spread, atol=1e-4, rtol=1e-4)


# ── GaussianMixture*: n_batches agreement (dense and spread backends) ────────


@pytest.mark.parametrize("n_batches", (1, 2, 3))
@pytest.mark.parametrize("n_spread", (None, 9))
def test_gmm_projection_n_batches_agreement(toy_gaussian_cloud, n_spread, n_batches):
    atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
    n_pixels_per_side = n_voxels_per_side[:2]

    atomic_volume = cxs.GaussianMixtureVolume(
        atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
    )
    image_config = cxs.BasicImageConfig(
        shape=n_pixels_per_side, pixel_size=voxel_size, voltage_in_kilovolts=300.0
    )
    reference = cxs.GaussianMixtureProjection(n_spread=n_spread).integrate(
        atomic_volume, image_config, outputs_real_space=True
    )
    batched = cxs.GaussianMixtureProjection(
        n_spread=n_spread, n_batches=n_batches
    ).integrate(atomic_volume, image_config, outputs_real_space=True)
    assert jnp.allclose(reference, batched, atol=1e-6)


@pytest.mark.parametrize("n_batches", (1, 2, 3))
@pytest.mark.parametrize("n_spread", (None, 9))
def test_gmm_render_n_batches_agreement(toy_gaussian_cloud, n_spread, n_batches):
    atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud

    gmm_volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
    reference = cxs.GaussianMixtureRenderFn(
        n_voxels_per_side, voxel_size, n_spread=n_spread
    )(gmm_volume)
    batched = cxs.GaussianMixtureRenderFn(
        n_voxels_per_side, voxel_size, n_spread=n_spread, n_batches=n_batches
    )(gmm_volume)
    assert jnp.allclose(reference, batched, atol=1e-6)


# ── explicit center-of-box parity check (even and odd grid sizes) ───────────
#
# `toy_gaussian_cloud` only exercises a fixed, even grid (128), so none of the
# tests above actually verify the real-space-center-at-index-N//2 convention
# (see `common.normalize_positions_to_grid`) for odd `N`. These tests check,
# directly (not via cross-backend agreement, which could hide a shared bug),
# that a single atom at the origin peaks at grid index `N // 2` for both
# parities, across the dense and spreading backends of both volume types.


@pytest.mark.parametrize("n", (10, 11))
@pytest.mark.parametrize("n_spread", (None, 9))
def test_gmm_projection_peak_at_box_center(n, n_spread):
    shape = (n, n)
    pixel_size = 1.0
    volume = cxs.GaussianMixtureVolume(np.zeros((1, 3)), amplitudes=1.0, variances=0.1)
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    integrator = cxs.GaussianMixtureProjection(sampling_mode="point", n_spread=n_spread)
    projection = integrator.integrate(volume, image_config, outputs_real_space=True)
    peak_index = tuple(int(i) for i in jnp.unravel_index(jnp.argmax(projection), shape))
    assert peak_index == (n // 2, n // 2)


@pytest.mark.parametrize("n", (10, 11))
@pytest.mark.parametrize("n_spread", (None, 9))
def test_gmm_render_peak_at_box_center(n, n_spread):
    shape = (n, n, n)
    voxel_size = 1.0
    volume = cxs.GaussianMixtureVolume(np.zeros((1, 3)), amplitudes=1.0, variances=0.1)
    render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size, n_spread=n_spread)
    real_voxel_grid = render_fn(volume)
    peak_index = tuple(
        int(i) for i in jnp.unravel_index(jnp.argmax(real_voxel_grid), shape)
    )
    assert peak_index == (n // 2, n // 2, n // 2)


# ── GaussianFourierVolume: bad instantiation ──────────────────────────────────


def test_fft_atom_bad_instantiation():
    with pytest.raises(ValueError):
        _ = cxs.GaussianFourierVolume(
            positions=np.zeros((10, 3)),
            kernel_fns=(im.FourierGaussian(),),
        )


# ── GaussianFourierProjection: each backend vs GaussianMixtureProjection ─────


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
    atom_volume = cxs.GaussianFourierVolume(
        positions=atom_positions,
        kernel_fns=im.FourierGaussian(amplitude=amplitude, b_factor=b_factor),
    )
    return gaussian_volume, gaussian_integrator, atom_volume, image_config


@pytest.mark.parametrize(
    "pixel_size, shape",
    ((1.0, (32, 32)), (1.0, (32, 31)), (1.0, (31, 32)), (1.0, (31, 31))),
)
@_backends
def test_fft_atom_projection_exact(monkeypatch, pdb_info, pixel_size, shape, backend):
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", backend)
    gaussian_volume, gaussian_integrator, atom_volume, image_config = (
        _make_fft_projection_exact_volumes(pdb_info)
    )
    atom_integrator = cxs.GaussianFourierProjection(
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
def test_fft_atom_projection_exact_backends_agree(
    monkeypatch, pdb_info, pixel_size, shape
):
    """nufftax and jax-finufft must produce identical projections."""
    _, _, atom_volume, image_config = _make_fft_projection_exact_volumes(pdb_info)
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "nufftax")
    proj_nufftax = compute_projection(
        atom_volume,
        cxs.GaussianFourierProjection(sampling_mode="point", eps=1e-10),
        image_config,
    )
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    proj_jax_finufft = compute_projection(
        atom_volume,
        cxs.GaussianFourierProjection(sampling_mode="point", eps=1e-10),
        image_config,
    )
    np.testing.assert_allclose(proj_nufftax, proj_jax_finufft, atol=1e-6)


def _make_antialias_volumes(pdb_info, width, pixel_size, shape):
    atom_positions, _, _ = pdb_info
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions, amplitudes=1.0, variances=width**2
    )
    atom_volume = cxs.GaussianFourierVolume(
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
def test_fft_atom_projection_antialias(
    monkeypatch, pdb_info, width, pixel_size, shape, backend
):
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", backend)
    gaussian_volume, gaussian_integrator, atom_volume, image_config = (
        _make_antialias_volumes(pdb_info, width, pixel_size, shape)
    )
    atom_integrator = cxs.GaussianFourierProjection(
        sampling_mode="average", eps=1e-10, upsample_factor=2
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
def test_fft_atom_projection_antialias_backends_agree(
    monkeypatch, pdb_info, width, pixel_size, shape
):
    """nufftax and jax-finufft must produce identical antialiased projections."""
    _, _, atom_volume, image_config = _make_antialias_volumes(
        pdb_info, width, pixel_size, shape
    )
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "nufftax")
    proj_nufftax = compute_projection(
        atom_volume,
        cxs.GaussianFourierProjection(
            sampling_mode="average", eps=1e-10, upsample_factor=2
        ),
        image_config,
    )
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    proj_jax_finufft = compute_projection(
        atom_volume,
        cxs.GaussianFourierProjection(
            sampling_mode="average", eps=1e-10, upsample_factor=2
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
    atom_volume = cxs.GaussianFourierVolume.from_tabulated_parameters(
        positions_by_id,
        peng_parameters_by_id,
        b_factor_by_element=4.0,
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
        (0.25, (134, 133), 3, 1e-5),
        (0.25, (133, 134), 3, 1e-5),
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
def test_fft_projection_peng(
    monkeypatch, pdb_info, pixel_size, shape, upsampfac, eps, backend
):
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", backend)
    gaussian_volume, gaussian_integrator, atom_volume = _make_peng_volumes(
        pdb_info, upsampfac
    )
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_integrator = cxs.GaussianFourierProjection(
        sampling_mode="average",
        upsample_factor=upsampfac,
        eps=eps,
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
def test_fft_projection_peng_backends_agree(
    monkeypatch, pdb_info, pixel_size, shape, upsampfac
):
    """nufftax and jax-finufft must produce identical Peng-tabulated projections."""
    _, _, atom_volume = _make_peng_volumes(pdb_info, upsampfac)
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "nufftax")
    proj_nufftax = compute_projection(
        atom_volume,
        cxs.GaussianFourierProjection(
            sampling_mode="average",
            upsample_factor=upsampfac,
            eps=1e-10,
        ),
        image_config,
    )
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    proj_jax_finufft = compute_projection(
        atom_volume,
        cxs.GaussianFourierProjection(
            sampling_mode="average",
            upsample_factor=upsampfac,
            eps=1e-10,
        ),
        image_config,
    )
    np.testing.assert_allclose(proj_nufftax, proj_jax_finufft, atol=1e-6)


@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_projection_custom_upsampfac(upsampfac):
    """Smoke test: custom upsampfac via options (nufftax) runs without error."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, pixel_size = (8, 8), 1.0
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_volume = cxs.GaussianFourierVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    atom_integrator = cxs.GaussianFourierProjection(
        sampling_mode="point",
        eps=1e-6,
        options={"upsampfac": upsampfac},
    )
    result = atom_integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )
    assert result.shape == (shape[0], shape[1] // 2 + 1)


@_skip_without_jax_finufft
@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_projection_custom_upsampfac_jax_finufft(monkeypatch, upsampfac):
    """Smoke test: custom opts via options (jax-finufft) runs without error."""
    from jax_finufft.options import NestedOpts, Opts

    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, pixel_size = (8, 8), 1.0
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_volume = cxs.GaussianFourierVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    opts = NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )
    atom_integrator = cxs.GaussianFourierProjection(
        sampling_mode="point",
        eps=1e-6,
        options={"opts": opts},
    )
    result = atom_integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )
    assert result.shape == (shape[0], shape[1] // 2 + 1)


# ── GaussianFourierRenderFn: each backend vs GaussianMixtureRenderFn ──────────


def _make_render_volumes(pdb_info, width):
    atom_positions, _, _ = pdb_info
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions, amplitudes=1.0, variances=width**2
    )
    atom_volume = cxs.GaussianFourierVolume(
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
def test_fft_atom_render(monkeypatch, pdb_info, width, voxel_size, shape, backend):
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", backend)
    gaussian_volume, atom_volume = _make_render_volumes(pdb_info, width)
    gaussian_render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    atom_render_fn = cxs.GaussianFourierRenderFn(
        shape, voxel_size, sampling_mode="average", eps=1e-10
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
def test_fft_atom_render_backends_agree(monkeypatch, pdb_info, width, voxel_size, shape):
    """nufftax and jax-finufft must produce identical rendered voxel grids."""
    _, atom_volume = _make_render_volumes(pdb_info, width)
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "nufftax")
    voxels_nufftax = cxs.GaussianFourierRenderFn(shape, voxel_size, eps=1e-10)(
        atom_volume
    )
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    voxels_jax_finufft = cxs.GaussianFourierRenderFn(shape, voxel_size, eps=1e-10)(
        atom_volume
    )
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
        atom_volume = cxs.GaussianFourierVolume(
            positions=atom_positions,
            kernel_fns=im.FourierGaussian(
                amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)
            ),
        )
        volumes.append(atom_volume)  # type: ignore
        render_fns.append(cxs.GaussianFourierRenderFn(shape, voxel_size, eps=1e-10))  # type: ignore
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
    atom_volume = cxs.GaussianFourierVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    render_fn = cxs.GaussianFourierRenderFn(
        shape,
        voxel_size,
        eps=1e-6,
        options={"upsampfac": upsampfac},
    )
    result = render_fn(atom_volume, outputs_real_space=True)
    assert result.shape == shape


@_skip_without_jax_finufft
@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_render_custom_upsampfac_jax_finufft(monkeypatch, upsampfac):
    """Smoke test: custom opts via options (jax-finufft) runs without error."""
    from jax_finufft.options import NestedOpts, Opts

    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, voxel_size = (8, 8, 8), 1.0
    atom_volume = cxs.GaussianFourierVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    opts = NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )
    render_fn = cxs.GaussianFourierRenderFn(
        shape, voxel_size, eps=1e-6, options={"opts": opts}
    )
    result = render_fn(atom_volume, outputs_real_space=True)
    assert result.shape == shape


# ── per-gaussian-component `n_spread` (tuple) ─────────────────────────────────
#
# `n_spread` may be a single `int` (one spread width shared by every gaussian
# component) or a `tuple[int, ...]` (one width per component, e.g. from
# `cxs.suggest_n_spread`). A tuple of `n_spread` values that are all equal
# must agree exactly with passing that value as a plain `int` -- the tuple
# path uses a different (Python-loop, not `vmap`) code path internally, so
# this is a real regression check, not a redundant one.


def test_gmm_projection_tuple_n_spread_matches_scalar(toy_gaussian_cloud):
    atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
    n_pixels_per_side = n_voxels_per_side[:2]
    n_gaussians = ff_a.shape[-1]

    atomic_volume = cxs.GaussianMixtureVolume(
        atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2)
    )
    image_config = cxs.BasicImageConfig(
        shape=n_pixels_per_side, pixel_size=voxel_size, voltage_in_kilovolts=300.0
    )
    scalar_projection = cxs.GaussianMixtureProjection(n_spread=9).integrate(
        atomic_volume, image_config, outputs_real_space=True
    )
    tuple_projection = cxs.GaussianMixtureProjection(
        n_spread=(9,) * n_gaussians
    ).integrate(atomic_volume, image_config, outputs_real_space=True)
    assert jnp.allclose(scalar_projection, tuple_projection, atol=1e-6)


def test_gmm_render_tuple_n_spread_matches_scalar(toy_gaussian_cloud):
    atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
    n_gaussians = ff_a.shape[-1]

    gmm_volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
    scalar_grid = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size, n_spread=9)(
        gmm_volume
    )
    tuple_grid = cxs.GaussianMixtureRenderFn(
        n_voxels_per_side, voxel_size, n_spread=(9,) * n_gaussians
    )(gmm_volume)
    assert jnp.allclose(scalar_grid, tuple_grid, atol=1e-6)


def test_gmm_tuple_n_spread_heterogeneous_differs_from_uniform(toy_gaussian_cloud):
    """A sanity check that genuinely different per-component `n_spread`
    values actually take effect (not silently ignored/collapsed to one
    value): using a much smaller `n_spread` for one wide-enough-to-matter
    component should change the result relative to a uniform `n_spread`.
    """
    atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
    gmm_volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
    uniform = cxs.GaussianMixtureRenderFn(
        n_voxels_per_side, voxel_size, n_spread=(11, 11)
    )(gmm_volume)
    heterogeneous = cxs.GaussianMixtureRenderFn(
        n_voxels_per_side, voxel_size, n_spread=(3, 11)
    )(gmm_volume)
    assert not jnp.allclose(uniform, heterogeneous, atol=1e-6)


def test_gmm_tuple_n_spread_wrong_length_raises(toy_gaussian_cloud):
    atom_positions, ff_a, ff_b, n_voxels_per_side, voxel_size = toy_gaussian_cloud
    gmm_volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
    render_fn = cxs.GaussianMixtureRenderFn(n_voxels_per_side, voxel_size, n_spread=(5,))
    with pytest.raises(ValueError, match="n_spread"):
        render_fn(gmm_volume)


# ── suggest_n_spread ──────────────────────────────────────────────────────────


def test_suggest_n_spread_gmm_global_and_termwise(toy_gaussian_cloud):
    atom_positions, ff_a, ff_b, _, voxel_size = toy_gaussian_cloud
    variances = ff_b / (8.0 * jnp.pi**2)
    n_gaussians = ff_a.shape[-1]
    volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, variances)

    global_n_spread = cxs.suggest_n_spread(volume, voxel_size, mode="global")
    termwise_n_spread = cxs.suggest_n_spread(volume, voxel_size, mode="termwise")

    assert isinstance(global_n_spread, int)
    assert isinstance(termwise_n_spread, tuple)
    assert len(termwise_n_spread) == n_gaussians
    assert all(isinstance(n, int) for n in termwise_n_spread)
    # The global value must be at least as large as every termwise value
    # (it's sized to the single widest component across all of them).
    assert global_n_spread >= max(termwise_n_spread)
    # Each termwise entry must match calling `variance_to_nspread` directly
    # on that component's own column of variances.
    for i, n_i in enumerate(termwise_n_spread):
        expected = im.variance_to_nspread(variances[:, i], voxel_size)
        assert n_i == expected


def test_suggest_n_spread_clamping(toy_gaussian_cloud):
    atom_positions, ff_a, ff_b, _, voxel_size = toy_gaussian_cloud
    volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))

    unclamped = cxs.suggest_n_spread(volume, voxel_size, mode="termwise")
    clamped_min = cxs.suggest_n_spread(
        volume, voxel_size, mode="termwise", min_n_spread=1000
    )
    clamped_max = cxs.suggest_n_spread(
        volume, voxel_size, mode="termwise", max_n_spread=1
    )
    assert all(n == 1000 for n in clamped_min)
    assert all(n == 1 for n in clamped_max)
    assert unclamped != clamped_min

    global_clamped = cxs.suggest_n_spread(
        volume, voxel_size, mode="global", min_n_spread=1000
    )
    assert global_clamped == 1000


def test_suggest_n_spread_unsupported_volume_type_raises():
    with pytest.raises(ValueError, match="suggest_n_spread"):
        cxs.suggest_n_spread(object(), 1.0)  # type: ignore[arg-type]


def test_suggest_n_spread_invalid_mode_raises(toy_gaussian_cloud):
    atom_positions, ff_a, ff_b, _, voxel_size = toy_gaussian_cloud
    volume = cxs.GaussianMixtureVolume(atom_positions, ff_a, ff_b / (8.0 * jnp.pi**2))
    with pytest.raises(ValueError, match="mode"):
        cxs.suggest_n_spread(volume, voxel_size, mode="bogus")  # type: ignore[arg-type]


def test_suggest_n_spread_output_is_plain_python_not_jax():
    """`suggest_n_spread` must be usable to set `n_spread` (a static,
    shape-determining argument), so its outputs must be plain Python `int`s
    -- not 0-d JAX/numpy arrays, which would be the wrong type to use as a
    static argument and would signal that the function accidentally
    round-tripped through JAX instead of staying eager/numpy-only
    throughout.
    """
    volume = cxs.GaussianMixtureVolume(
        np.zeros((1, 3)),
        amplitudes=jnp.array([[1.0, 0.5]]),
        variances=jnp.array([[0.1, 0.05]]),
    )
    global_n_spread = cxs.suggest_n_spread(volume, 1.0, mode="global")
    termwise_n_spread = cxs.suggest_n_spread(volume, 1.0, mode="termwise")
    assert type(global_n_spread) is int
    assert all(type(n) is int for n in termwise_n_spread)


def test_make_frequencies_1d_rfft():
    shape_u, p = (12, 10), jnp.asarray(1.3)
    full = make_frequencies_1d(shape_u, p, modeord=1)
    rfft = make_frequencies_1d(shape_u, p, modeord=1, outputs_rfft=True)
    np.testing.assert_allclose(rfft[0], jnp.fft.rfftfreq(shape_u[-1], p))
    np.testing.assert_allclose(rfft[1], full[1])
    with pytest.raises(ValueError, match="modeord"):
        make_frequencies_1d(shape_u, p, modeord=0, outputs_rfft=True)


# ── GaussianMixtureProjection: point sampling ────────────────────────────────


def _max_shell_relative_error(a, b, image_config):
    """The largest relative L2 error of rfft spectrum `a` against `b` over radial
    shells, with the corners beyond Nyquist as one shell. The bin at an even y and x
    Nyquist is left out: four frequencies share it, and `GaussianFourierProjection`
    fills it from the opposite corner."""
    n_y, n_x = image_config.padded_shape
    pixel_size = float(image_config.pixel_size)
    frequencies_y = np.fft.fftfreq(n_y, pixel_size)[:, None]
    frequencies_x = np.fft.rfftfreq(n_x, pixel_size)
    radius = 2 * pixel_size * np.hypot(frequencies_y, frequencies_x)
    n_shells = min(n_y, n_x) // 2
    shells = np.minimum(np.floor(radius * n_shells), n_shells)
    a, b = np.asarray(a), np.asarray(b)
    keep = np.ones(a.shape, bool)
    if n_y % 2 == 0 and n_x % 2 == 0:
        keep[n_y // 2, -1] = False
    return max(
        np.linalg.norm((a - b)[keep & (shells == s)])
        / np.linalg.norm(b[keep & (shells == s)])
        for s in np.unique(shells)
    )


def _blur_variance(p, r):
    """The blur that keeps folded-in content below 1e-2 at pixel size `p` and
    upsampling factor `r`."""
    return p**2 * np.log(200) / (2 * np.pi**2 * r * (r - 1))


def _make_gaussian_volumes(positions, variance):
    gaussian_volume = cxs.GaussianMixtureVolume(
        positions, amplitudes=1.0, variances=variance
    )
    atom_volume = cxs.GaussianFourierVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(b_factor=8 * np.pi**2 * variance),
    )
    return gaussian_volume, atom_volume


class TestGMMPointSampling:
    exact_integrator = cxs.GaussianFourierProjection(sampling_mode="point", eps=1e-10)

    @pytest.mark.parametrize("n_spread", (None, 11))
    def test_average_ignores_point_only_args(self, pdb_info, n_spread):
        volume, _ = _make_gaussian_volumes(pdb_info[0], 0.5)
        image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
        default = cxs.GaussianMixtureProjection(n_spread=n_spread)
        ignored = cxs.GaussianMixtureProjection(
            n_spread=n_spread, upsample_factor=3, blur_variance=0.5
        )
        np.testing.assert_array_equal(
            ignored.integrate(volume, image_config),
            default.integrate(volume, image_config),
        )

    @pytest.mark.parametrize("variance", (0.05, 0.5))
    @pytest.mark.parametrize("shape", ((32, 32), (31, 31), (31, 32), (32, 30)))
    @pytest.mark.parametrize("upsample_factor", (2, 3))
    def test_point_matches_exact(self, pdb_info, upsample_factor, shape, variance):
        gaussian_volume, atom_volume = _make_gaussian_volumes(pdb_info[0], variance)
        image_config = cxs.BasicImageConfig(shape, 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode="point",
            upsample_factor=upsample_factor,
            blur_variance=_blur_variance(1.0, upsample_factor),
        )
        projection = integrator.integrate(gaussian_volume, image_config)
        exact = self.exact_integrator.integrate(atom_volume, image_config)
        assert _max_shell_relative_error(projection, exact, image_config) <= 1e-2

    def test_point_requires_upsample_factor_2(self):
        with pytest.raises(
            ValueError, match="GaussianMixtureProjection.*upsample_factor.*1"
        ):
            cxs.GaussianMixtureProjection(sampling_mode="point", upsample_factor=1)
        integrator = cxs.GaussianMixtureProjection()
        assert integrator.sampling_mode == "average"
        assert integrator.upsample_factor == 2

    @pytest.mark.parametrize(
        "kwargs, keyword",
        (
            ({"upsample_factor": 0}, "upsample_factor"),
            ({"upsample_factor": 1.5}, "upsample_factor"),
            ({"blur_variance": -1.0}, "blur_variance"),
        ),
    )
    def test_invalid_values(self, kwargs, keyword):
        with pytest.raises(ValueError, match=keyword):
            cxs.GaussianMixtureProjection(sampling_mode="point", **kwargs)

    @pytest.mark.parametrize(
        "shape, new_shape", (((64, 64), (32, 32)), ((66, 66), (33, 33)))
    )
    def test_point_nests_under_fourier_crop(self, shape, new_shape):
        positions = np.random.default_rng(0).uniform(-8.0, 8.0, size=(20, 3))
        volume, _ = _make_gaussian_volumes(positions, 0.6)
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode="point", upsample_factor=4
        )
        fine = integrator.integrate(
            volume,
            cxs.BasicImageConfig(shape, 1.0, voltage_in_kilovolts=300.0),
            outputs_real_space=True,
        )
        cropped = im.fourier_crop_to_shape(
            fine, new_shape, outputs_real_space=False, preserve_mean=True
        )
        coarse = integrator.integrate(
            volume, cxs.BasicImageConfig(new_shape, shape[0] / new_shape[0], 300.0)
        )
        np.testing.assert_allclose(cropped, coarse, atol=1e-10 * np.abs(coarse).max())

    def test_average_nests_under_block_mean(self):
        positions = np.random.default_rng(1).uniform(-6.0, 6.0, size=(20, 3))
        integrator = cxs.GaussianMixtureProjection(sampling_mode="average")

        def render(positions, shape, pixel_size):
            volume, _ = _make_gaussian_volumes(positions, 0.3)
            image_config = cxs.BasicImageConfig(shape, pixel_size, 300.0)
            return np.asarray(
                integrator.integrate(volume, image_config, outputs_real_space=True)
            )

        fine = render(positions, (32, 32), 1.0)
        # With an even coarse size, each block's center sits half a fine pixel above
        # its coarse pixel's center on both axes
        coarse = render(positions - np.array([0.5, 0.5, 0.0]), (16, 16), 2.0)
        block_mean = fine.reshape(16, 2, 16, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(block_mean, coarse, atol=1e-10 * np.abs(coarse).max())

    def test_point_traced_pixel_size(self, pdb_info):
        volume, _ = _make_gaussian_volumes(pdb_info[0], 0.05)
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode="point", blur_variance=_blur_variance(1.0, 2)
        )

        def project(pixel_size):
            image_config = cxs.BasicImageConfig((31, 32), pixel_size, 300.0)
            return integrator.integrate(volume, image_config)

        np.testing.assert_allclose(
            jax.jit(project)(jnp.asarray(1.1)), project(1.1), rtol=1e-10, atol=1e-12
        )

    def test_upsampled_real_space_matches_fourier(self, pdb_info):
        volume, _ = _make_gaussian_volumes(pdb_info[0], 0.05)
        image_config = cxs.BasicImageConfig((31, 32), 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode="point",
            upsample_factor=2,
            blur_variance=_blur_variance(1.0, 2),
        )
        real = integrator.integrate(volume, image_config, outputs_real_space=True)
        fourier = integrator.integrate(volume, image_config)
        np.testing.assert_allclose(
            real,
            jnp.fft.irfftn(fourier, s=image_config.padded_shape),
            atol=1e-12 * np.abs(real).max(),
        )

    def test_upsampled_with_shape_field(self, pdb_info):
        # Even, so the 2n grid's corner bin, filled differently by the reference,
        # crops to the n grid's corner bin, which `_max_shell_relative_error` leaves out
        n = 32
        gaussian_volume, atom_volume = _make_gaussian_volumes(pdb_info[0], 0.05)
        image_config = cxs.BasicImageConfig((n, n), 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection(
            shape=(2 * n, 2 * n),
            sampling_mode="point",
            upsample_factor=2,
            blur_variance=_blur_variance(1.0, 2),
        )
        exact_integrator = cxs.GaussianFourierProjection(
            sampling_mode="point", eps=1e-10, shape=(2 * n, 2 * n)
        )
        projection = integrator.integrate(gaussian_volume, image_config)
        exact = exact_integrator.integrate(atom_volume, image_config)
        assert projection.shape == (n, n // 2 + 1)
        assert _max_shell_relative_error(projection, exact, image_config) <= 1e-2

    @pytest.mark.parametrize("n_spread", (None, 11))
    def test_upsampled_n_batches_agree(self, pdb_info, n_spread):
        volume, _ = _make_gaussian_volumes(pdb_info[0], 0.05)
        image_config = cxs.BasicImageConfig((31, 32), 1.0, voltage_in_kilovolts=300.0)
        kwargs = dict(
            sampling_mode="point",
            upsample_factor=2,
            blur_variance=_blur_variance(1.0, 2),
            n_spread=n_spread,
        )
        reference = cxs.GaussianMixtureProjection(**kwargs).integrate(
            volume, image_config
        )
        batched = cxs.GaussianMixtureProjection(n_batches=2, **kwargs).integrate(
            volume, image_config
        )
        np.testing.assert_allclose(
            batched, reference, atol=1e-12 * np.abs(reference).max()
        )

    def test_point_gradients(self, pdb_info):
        shape, variance, pixel_size = (31, 32), 0.05, 1.0
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode="point",
            upsample_factor=2,
            blur_variance=_blur_variance(pixel_size, 2),
        )

        # Random complex weights over the inner half of the band, so that the loss
        # sees phases (center shifts) where the spectrum is accurate to well below 1e-2
        n_y, n_x = shape
        key_re, key_im = jax.random.split(jax.random.key(0))
        weights = jax.random.normal(key_re, (n_y, n_x // 2 + 1)) + 1j * (
            jax.random.normal(key_im, (n_y, n_x // 2 + 1))
        )
        radius = (
            2
            * pixel_size
            * jnp.hypot(
                jnp.fft.fftfreq(n_y, pixel_size)[:, None],
                jnp.fft.rfftfreq(n_x, pixel_size),
            )
        )
        weights = weights * (radius < 0.5)

        def make_grad_fn(volume_index, integrator):
            def loss(positions, variance, pixel_size):
                volume = _make_gaussian_volumes(positions, variance)[volume_index]
                image_config = cxs.BasicImageConfig(
                    shape, pixel_size, voltage_in_kilovolts=300.0
                )
                spectrum = integrator.integrate(volume, image_config)
                return jnp.sum(jnp.real(spectrum * weights))

            return jax.grad(loss, argnums=(0, 1, 2))

        args = (jnp.asarray(pdb_info[0]), jnp.asarray(variance), jnp.asarray(pixel_size))
        grads = make_grad_fn(0, integrator)(*args)
        exact_grads = make_grad_fn(1, self.exact_integrator)(*args)
        for grad, exact_grad in zip(grads, exact_grads):
            error = np.linalg.norm(grad - exact_grad) / np.linalg.norm(exact_grad)
            assert error <= 1e-2

    def test_point_float32(self, pdb_info):
        gaussian_volume, atom_volume = _make_gaussian_volumes(pdb_info[0], 0.05)
        image_config = cxs.BasicImageConfig((31, 32), 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection(
            sampling_mode="point",
            upsample_factor=2,
            blur_variance=_blur_variance(1.0, 2),
        )
        exact = self.exact_integrator.integrate(atom_volume, image_config)
        with jax.enable_x64(False):
            projection = integrator.integrate(gaussian_volume, image_config)
        assert projection.dtype == jnp.complex64
        assert _max_shell_relative_error(projection, exact, image_config) <= 1e-2


def _make_two_component_volumes(positions, variances=(0.05, 0.5)):
    gaussian_volume = cxs.GaussianMixtureVolume(
        positions,
        amplitudes=np.ones((len(positions), 2)),
        variances=np.broadcast_to(variances, (len(positions), 2)),
    )
    atom_volume = cxs.GaussianFourierVolume(
        positions=(positions, positions),
        kernel_fns=tuple(
            im.FourierGaussian(b_factor=8 * np.pi**2 * v) for v in variances
        ),
    )
    return gaussian_volume, atom_volume


class TestGMMWithCalibration:
    def test_with_calibration_blur_formula(self, pdb_info):
        volume, _ = _make_two_component_volumes(pdb_info[0])
        image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection.with_calibration(
            volume,
            image_config,
            sampling_mode="point",
            upsample_factor=2,
        )
        assert integrator.blur_variance == pytest.approx(np.log(200) / (4 * np.pi**2))

    def test_with_calibration_sizes_n_spread_on_blurred_fine_grid(self, pdb_info):
        volume, _ = _make_two_component_volumes(pdb_info[0])
        image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection.with_calibration(
            volume,
            image_config,
            sampling_mode="point",
            upsample_factor=2,
        )
        blurred = eqx.tree_at(
            lambda v: v.variances, volume, volume.variances + integrator.blur_variance
        )
        expected = cxs.suggest_n_spread(blurred, 0.5, cutoff_sigma=4.0)
        assert isinstance(integrator.n_spread, tuple)
        assert integrator.n_spread == expected

    def test_with_calibration_average(self, pdb_info):
        volume, _ = _make_two_component_volumes(pdb_info[0])
        image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection.with_calibration(volume, image_config)
        assert integrator.blur_variance == 0.0
        assert integrator.n_spread == cxs.suggest_n_spread(volume, 1.0)

    def test_with_calibration_refuses_factor_1(self, pdb_info):
        volume, _ = _make_two_component_volumes(pdb_info[0])
        image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
        with pytest.raises(ValueError, match="upsample_factor"):
            cxs.GaussianMixtureProjection.with_calibration(
                volume,
                image_config,
                sampling_mode="point",
                upsample_factor=1,
            )

    @pytest.mark.parametrize("tolerance", (0.0, -1e-2, 2.0, 5.0))
    def test_with_calibration_invalid_tolerance(self, pdb_info, tolerance):
        volume, _ = _make_two_component_volumes(pdb_info[0])
        image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
        with pytest.raises(ValueError, match="tolerance"):
            cxs.GaussianMixtureProjection.with_calibration(
                volume,
                image_config,
                tolerance=tolerance,
                sampling_mode="point",
                upsample_factor=2,
            )

    @pytest.mark.parametrize("float32", (False, True))
    @pytest.mark.parametrize("shape", ((32, 30), (31, 31)))
    @pytest.mark.parametrize("upsample_factor", (2, 3))
    def test_calibrated_point_deblur_within_tolerance(
        self, pdb_info, upsample_factor, shape, float32
    ):
        gaussian_volume, atom_volume = _make_two_component_volumes(pdb_info[0])
        image_config = cxs.BasicImageConfig(shape, 1.0, voltage_in_kilovolts=300.0)
        integrator = cxs.GaussianMixtureProjection.with_calibration(
            gaussian_volume,
            image_config,
            sampling_mode="point",
            upsample_factor=upsample_factor,
        )
        exact = TestGMMPointSampling.exact_integrator.integrate(atom_volume, image_config)
        with jax.enable_x64(not float32):
            projection = integrator.integrate(gaussian_volume, image_config)
        assert _max_shell_relative_error(projection, exact, image_config) <= 1e-2


# ── GaussianFourier upsampling: folding aliases onto the output grid ─────────


def _max_relative_error(actual, expected):
    return np.abs(actual - expected).max() / np.abs(expected).max()


@pytest.mark.parametrize("shape", ((32, 32), (31, 32)))
@pytest.mark.parametrize(
    "b_factor, upsample_factor, rtol",
    ((5.0, 8, 1e-8), (20.0, 3, 1e-4), (20.0, 8, 1e-8)),
)
def test_fft_projection_average_converges_to_pixel_average(
    shape, b_factor, upsample_factor, rtol
):
    positions = np.random.default_rng(0).uniform(-8.0, 8.0, size=(20, 3))
    gaussian_volume, atom_volume = _make_gaussian_volumes(
        positions, b_factor / (8 * np.pi**2)
    )
    image_config = cxs.BasicImageConfig(shape, 1.0, voltage_in_kilovolts=300.0)
    exact = cxs.GaussianMixtureProjection(sampling_mode="average").integrate(
        gaussian_volume, image_config, outputs_real_space=True
    )
    integrator = cxs.GaussianFourierProjection(
        sampling_mode="average", eps=1e-12, upsample_factor=upsample_factor
    )
    projection = integrator.integrate(atom_volume, image_config, outputs_real_space=True)
    assert _max_relative_error(projection, exact) <= rtol


@pytest.mark.parametrize("upsample_factor, rtol", ((3, 1e-4), (4, 1e-8)))
def test_fft_render_upsampling_converges_to_voxel_average(upsample_factor, rtol):
    shape, voxel_size = (15, 16, 17), 1.0
    # Away from the edges, since the fourier render is periodic and the reference not
    positions = np.random.default_rng(0).uniform(-3.0, 3.0, size=(20, 3))
    gaussian_volume, atom_volume = _make_gaussian_volumes(
        positions, 20.0 / (8 * np.pi**2)
    )
    exact = cxs.GaussianMixtureRenderFn(shape, voxel_size)(gaussian_volume)
    render_fn = cxs.GaussianFourierRenderFn(
        shape,
        voxel_size,
        sampling_mode="average",
        eps=1e-12,
        upsample_factor=upsample_factor,
    )
    assert _max_relative_error(render_fn(atom_volume), exact) <= rtol


@pytest.mark.parametrize("upsample_factor", (0, 1.5))
def test_fft_upsample_factor_must_be_a_positive_integer(upsample_factor):
    with pytest.raises(ValueError, match="upsample_factor"):
        cxs.GaussianFourierProjection(upsample_factor=upsample_factor)
    with pytest.raises(ValueError, match="upsample_factor"):
        cxs.GaussianFourierRenderFn((8, 8, 8), 1.0, upsample_factor=upsample_factor)


# ── GaussianFourier: sampling modes ───────────────────────────────────────────


def _corner_mask(shape):
    """True except at an rfft's bin where both axes are at an even Nyquist, which four
    frequencies share and `GaussianFourierProjection` fills from the opposite corner."""
    mask = np.ones((shape[0], shape[1] // 2 + 1), bool)
    if shape[0] % 2 == 0 and shape[1] % 2 == 0:
        mask[shape[0] // 2, -1] = False
    return mask


@pytest.mark.parametrize(
    "shape, new_shape", (((64, 64), (32, 32)), ((66, 66), (33, 33)), ((64, 64), (31, 31)))
)
def test_fft_point_nests_under_fourier_crop(shape, new_shape):
    positions = np.random.default_rng(0).uniform(-8.0, 8.0, size=(20, 3))
    _, atom_volume = _make_gaussian_volumes(positions, 0.6)
    integrator = cxs.GaussianFourierProjection(sampling_mode="point", eps=1e-10)
    fine = integrator.integrate(
        atom_volume,
        cxs.BasicImageConfig(shape, 1.0, voltage_in_kilovolts=300.0),
        outputs_real_space=True,
    )
    cropped = np.asarray(
        im.fourier_crop_to_shape(
            fine, new_shape, outputs_real_space=False, preserve_mean=True
        )
    )
    coarse = np.asarray(
        integrator.integrate(
            atom_volume,
            cxs.BasicImageConfig(new_shape, shape[0] / new_shape[0], 300.0),
        )
    )
    mask = _corner_mask(new_shape)
    np.testing.assert_allclose(
        # The NUFFT is least accurate at the edge of its band, the Nyquist lines here
        cropped[mask],
        coarse[mask],
        atol=1e-6 * np.abs(coarse).max(),
    )


def test_fft_point_ignores_upsample_factor():
    positions = np.random.default_rng(0).uniform(-8.0, 8.0, size=(20, 3))
    _, atom_volume = _make_gaussian_volumes(positions, 0.6)
    image_config = cxs.BasicImageConfig((32, 32), 1.0, voltage_in_kilovolts=300.0)
    projections = [
        cxs.GaussianFourierProjection(
            sampling_mode="point", upsample_factor=factor
        ).integrate(atom_volume, image_config)
        for factor in (2, 4)
    ]
    np.testing.assert_array_equal(*projections)


def test_fft_defaults():
    for integrator in (
        cxs.GaussianFourierProjection(),
        cxs.GaussianFourierRenderFn((8, 8, 8), 1.0),
    ):
        assert integrator.sampling_mode == "point"
        assert integrator.upsample_factor == 2


def test_fft_average_requires_upsample_factor_2():
    with pytest.raises(ValueError, match="GaussianFourierProjection.*upsample_factor.*1"):
        cxs.GaussianFourierProjection(sampling_mode="average", upsample_factor=1)
    with pytest.raises(ValueError, match="GaussianFourierRenderFn.*upsample_factor.*1"):
        cxs.GaussianFourierRenderFn(
            (8, 8, 8), 1.0, sampling_mode="average", upsample_factor=1
        )
