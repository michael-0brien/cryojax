"""Tests for FourierSliceExtraction volume integrator.

Strategy
--------
Property tests (shape, scaling, linearity) verify exact algebraic invariants.
Accuracy tests compare against GaussianMixtureProjection, which provides an
analytic ground truth for a single spherical Gaussian at the origin.

Measured accuracy on a 32³ grid with sigma=1 px (variance=1 Å²)
-----------------------------------------------------------------
- Identity / theta=90 phi=0: grid ≈ spline ≈ 1.8e-5 (limited by rendering)
- theta=45 phi=0:  grid ~0.4 %, grid+deconv ~0.2 %, spline ~0.1 % of peak

JIT strategy
------------
Module-level ``@eqx.filter_jit`` helpers are keyed on (pytree structure, static
leaf values).  Using a single ``_FSE`` integrator constant and a single
``_GMM_INTEGRATOR`` constant ensures every test that uses default parameters
hits the same compiled code, with no recompilation between test cases that
differ only in array content (different poses, different voxel values).

Separate compilations are only triggered when the static signature genuinely
differs: different ``image_config.padded_shape`` (shape tests), different
``outputs_integral`` flag (scaling test), or different volume shape (64³
deconvolution tests).

Notes on boundary behaviour
----------------------------
For in-plane phi rotations (theta=0) the rotated frequency slice has
coordinates outside the Fourier box; zero-fill of those coordinates is
correct ('fill' mode) but reduces the edge-to-peak contrast.
``test_phi_rotation_mean_conserved`` documents this: the projection mean
(= DC Fourier coefficient) is preserved even when corners are clipped.
"""

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array


# ── module-level integrator singletons (never recreated) ─────────────────────
# Sharing a single instance across every call guarantees filter_jit never sees
# a new static signature for the default integrators.

_FSE = cxs.FourierSliceExtraction()
_FSE_NO_SCALE = cxs.FourierSliceExtraction(outputs_integral=False)
_GMM_INTEGRATOR = cxs.GaussianMixtureProjection(sampling_mode="average")


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gmm_volume():
    """Single Gaussian at origin (amplitude=1, variance=1 Å², sigma=1 px)."""
    return cxs.GaussianMixtureVolume(np.zeros((1, 3)), amplitudes=1.0, variances=1.0)


@pytest.fixture(scope="module")
def fourier_grid_volume(gmm_volume):
    """FourierVoxelGridVolume rendered from the GMM volume."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(render_fn(gmm_volume))


@pytest.fixture(scope="module")
def fourier_grid_volume_deconv(gmm_volume):
    """FourierVoxelGridVolume with apply_deconvolve=True, same GMM."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(gmm_volume), apply_deconvolve=True
    )


@pytest.fixture(scope="module")
def fourier_spline_volume(gmm_volume):
    """FourierVoxelSplineVolume rendered from the GMM volume."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelSplineVolume.from_real_voxel_grid(render_fn(gmm_volume))


@pytest.fixture(scope="module")
def image_config():
    return cxs.BasicImageConfig((32, 32), pixel_size=1.0, voltage_in_kilovolts=300.0)


# ── JIT-compiled primitives ───────────────────────────────────────────────────
# All heavy computation goes through these.  The module-level integrator
# singletons and the shared ``image_config`` fixture mean that filter_jit
# compiles each function at most once per (volume-shape, image-shape) pair.


@eqx.filter_jit
def _project_real(volume, integrator, image_config) -> Array:
    return integrator.integrate(volume, image_config, outputs_real_space=True)


@eqx.filter_jit
def _project_fourier(volume, integrator, image_config) -> Array:
    return integrator.integrate(volume, image_config, outputs_real_space=False)


@eqx.filter_jit
def _max_abs_error(volume_a, volume_b, integrator_a, integrator_b, image_config) -> Array:
    """Max pixelwise absolute difference between two projections (compiled once)."""
    proj_a = integrator_a.integrate(volume_a, image_config, outputs_real_space=True)
    proj_b = integrator_b.integrate(volume_b, image_config, outputs_real_space=True)
    return jnp.max(jnp.abs(proj_a - proj_b))


@eqx.filter_jit
def _projection_mean(volume, integrator, image_config) -> Array:
    proj = integrator.integrate(volume, image_config, outputs_real_space=True)
    return jnp.mean(proj)


# ── output shape ───────────────────────────────────────────────────────────────
# Different shapes → new compilations (unavoidable); covered by shape-specific
# configs that are local to these tests.


@pytest.mark.parametrize("shape", [(32, 32), (24, 24), (16, 16)])
def test_output_shape_real(fourier_grid_volume, shape):
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    result = _project_real(fourier_grid_volume, _FSE, config)
    assert result.shape == config.padded_shape


@pytest.mark.parametrize("shape", [(32, 32), (24, 24), (16, 16)])
def test_output_shape_fourier(fourier_grid_volume, shape):
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    result = _project_fourier(fourier_grid_volume, _FSE, config)
    ny, nx = config.padded_shape
    assert result.shape == (ny, nx // 2 + 1)


def test_padded_output_shape(fourier_grid_volume):
    """When padded_shape > shape, output must match padded_shape."""
    config = cxs.BasicImageConfig(
        (32, 32), pixel_size=1.0, voltage_in_kilovolts=300.0, padded_shape=(64, 64)
    )
    result = _project_real(fourier_grid_volume, _FSE, config)
    assert result.shape == (64, 64)


# ── outputs_integral scaling ───────────────────────────────────────────────────
# Two separate integrators: _FSE (True) and _FSE_NO_SCALE (False).
# Each is compiled once regardless of how many parametrize cases run.


@pytest.mark.parametrize("pixel_size", [0.5, 1.0, 2.0])
def test_outputs_integral_scale(fourier_grid_volume, pixel_size):
    """outputs_integral=True must equal pixel_size * outputs_integral=False."""
    config = cxs.BasicImageConfig(
        (32, 32), pixel_size=pixel_size, voltage_in_kilovolts=300.0
    )
    with_scale = np.array(_project_fourier(fourier_grid_volume, _FSE, config))
    without_scale = np.array(_project_fourier(fourier_grid_volume, _FSE_NO_SCALE, config))
    np.testing.assert_allclose(with_scale, pixel_size * without_scale, atol=1e-12)


# ── linearity ──────────────────────────────────────────────────────────────────


def test_superposition(image_config):
    """proj(A + B) must equal proj(A) + proj(B) to machine precision."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    vox_a = np.array(
        render_fn(
            cxs.GaussianMixtureVolume(np.zeros((1, 3)), amplitudes=1.0, variances=20.0)
        )
    )
    vox_b = np.array(
        render_fn(
            cxs.GaussianMixtureVolume(
                np.array([[5.0, 0.0, 0.0]]), amplitudes=0.5, variances=15.0
            )
        )
    )
    vol_a = cxs.FourierVoxelGridVolume.from_real_voxel_grid(vox_a)
    vol_b = cxs.FourierVoxelGridVolume.from_real_voxel_grid(vox_b)
    vol_sum = cxs.FourierVoxelGridVolume.from_real_voxel_grid(vox_a + vox_b)

    proj_a = np.array(_project_fourier(vol_a, _FSE, image_config))
    proj_b = np.array(_project_fourier(vol_b, _FSE, image_config))
    proj_sum = np.array(_project_fourier(vol_sum, _FSE, image_config))

    np.testing.assert_allclose(proj_sum, proj_a + proj_b, atol=1e-10)


# ── accuracy vs GaussianMixtureProjection (ground truth) ──────────────────────
#
# GaussianMixtureProjection evaluates the projection analytically and is used
# as ground truth.  FourierSliceExtraction first renders the volume to voxels
# (introduces rendering error ~3e-6) then samples a Fourier slice.
#
# Exact poses (identity, theta=90/phi=0): the rotated slice falls on grid
# planes so no interpolation is needed.  Total error ≈ rendering error ~3e-6.
#
# Off-axis poses (theta∈[30°,60°], phi=0): bilinear interpolation introduces
# ~5–7% peak error; cubic spline introduces ~1%.
#
# _max_abs_error is compiled once for (32³ grid, GMM, FSE) and reused across
# every accuracy test that shares the same volume shapes.


def test_accuracy_identity_grid(gmm_volume, fourier_grid_volume, image_config):
    """Grid at identity must match GMM to within rendering discretization."""
    err = float(
        _max_abs_error(
            gmm_volume, fourier_grid_volume, _GMM_INTEGRATOR, _FSE, image_config
        )
    )
    assert err < 5e-5


def test_accuracy_identity_spline(gmm_volume, fourier_spline_volume, image_config):
    """Spline at identity must match GMM to within rendering discretization."""
    err = float(
        _max_abs_error(
            gmm_volume, fourier_spline_volume, _GMM_INTEGRATOR, _FSE, image_config
        )
    )
    assert err < 5e-5


def test_accuracy_theta90_phi0_grid(gmm_volume, fourier_grid_volume, image_config):
    """theta=90, phi=0: slice on exact grid planes; matches GMM to 1e-5."""
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0)
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_grid_volume.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            image_config,
        )
    )
    assert err < 5e-5


def test_accuracy_theta90_phi0_spline(gmm_volume, fourier_spline_volume, image_config):
    """Same exact case for spline volume."""
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0)
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_spline_volume.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            image_config,
        )
    )
    assert err < 5e-5


@pytest.mark.parametrize("theta_angle", [30.0, 45.0, 60.0])
def test_accuracy_offaxis_grid(
    gmm_volume, fourier_grid_volume, image_config, theta_angle
):
    """Off-axis bilinear accuracy: must agree with GMM to within 1 % of peak.

    Measured on a 32³ grid with variance=1 Å²: ~0.35–0.41 % at these tilts.
    """
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=theta_angle, psi_angle=0.0)
    proj_gmm = np.array(
        _project_real(gmm_volume.rotate_to_pose(pose), _GMM_INTEGRATOR, image_config)
    )
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_grid_volume.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            image_config,
        )
    )
    assert err < 0.01 * float(proj_gmm.max())


@pytest.mark.parametrize("theta_angle", [30.0, 45.0, 60.0])
def test_accuracy_offaxis_grid_deconv(
    gmm_volume, fourier_grid_volume_deconv, image_config, theta_angle
):
    """Off-axis bilinear accuracy with apply_deconvolve=True: within 0.5 % of peak.

    Measured on a 32³ grid with variance=1 Å²:
      theta=30°: ~0.23 %   theta=45°: ~0.20 %   theta=60°: ~0.23 %

    With variance=1 (sigma=1 px), the volume has significant high-frequency
    content and deconvolution reduces peak error relative to the uncorrected
    bilinear case (~0.35–0.41 %).  Note that at exact poses (theta=0°, 90°)
    deconvolution is slightly harmful because no interpolation occurs there and
    the sinc² pre-whitening introduces ringing.
    """
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=theta_angle, psi_angle=0.0)
    proj_gmm = np.array(
        _project_real(gmm_volume.rotate_to_pose(pose), _GMM_INTEGRATOR, image_config)
    )
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_grid_volume_deconv.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            image_config,
        )
    )
    assert err < 0.005 * float(proj_gmm.max())


@pytest.mark.parametrize("theta_angle", [30.0, 45.0, 60.0])
def test_accuracy_offaxis_spline(
    gmm_volume, fourier_spline_volume, image_config, theta_angle
):
    """Off-axis spline accuracy: must agree with GMM to within 0.5 % of peak.

    Measured on a 32³ grid with variance=1 Å²: ~0.10–0.11 % at these tilts.
    """
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=theta_angle, psi_angle=0.0)
    proj_gmm = np.array(
        _project_real(gmm_volume.rotate_to_pose(pose), _GMM_INTEGRATOR, image_config)
    )
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_spline_volume.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            image_config,
        )
    )
    assert err < 0.005 * float(proj_gmm.max())


# ── grid vs spline consistency ─────────────────────────────────────────────────


def test_grid_vs_spline_identity(
    fourier_grid_volume, fourier_spline_volume, image_config
):
    """At identity, grid and spline must agree to 1e-6."""
    err = float(
        _max_abs_error(
            fourier_grid_volume, fourier_spline_volume, _FSE, _FSE, image_config
        )
    )
    assert err < 1e-6


def test_grid_vs_spline_theta90_phi0(
    fourier_grid_volume, fourier_spline_volume, image_config
):
    """At theta=90/phi=0 (exact case), grid and spline must agree to 1e-6."""
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0)
    err = float(
        _max_abs_error(
            fourier_grid_volume.rotate_to_pose(pose),
            fourier_spline_volume.rotate_to_pose(pose),
            _FSE,
            _FSE,
            image_config,
        )
    )
    assert err < 1e-6


# ── from_fourier_voxel_grid constructor equivalence ───────────────────────────


def test_grid_from_fourier_voxel_grid_matches_from_real(gmm_volume, image_config):
    """from_fourier_voxel_grid(fftn(grid)) must equal from_real_voxel_grid(grid)."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    real_grid = render_fn(gmm_volume)
    vol_real = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_grid)
    vol_fourier = cxs.FourierVoxelGridVolume.from_fourier_voxel_grid(im.fftn(real_grid))
    err = float(_max_abs_error(vol_real, vol_fourier, _FSE, _FSE, image_config))
    assert err == 0.0


def test_spline_from_fourier_voxel_grid_matches_from_real(gmm_volume, image_config):
    """from_fourier_voxel_grid(fftn(grid)) must equal from_real_voxel_grid(grid)."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    real_grid = render_fn(gmm_volume)
    vol_real = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(real_grid)
    vol_fourier = cxs.FourierVoxelSplineVolume.from_fourier_voxel_grid(im.fftn(real_grid))
    err = float(_max_abs_error(vol_real, vol_fourier, _FSE, _FSE, image_config))
    assert err == 0.0


# ── in-plane phi rotation: mean (DC) is conserved ─────────────────────────────
#
# For in-plane phi rotations (theta=0) the frequency slice extends beyond the
# Fourier box at corner frequencies.  The 'fill' boundary mode zeros those
# coefficients, which is physically correct but changes the image appearance
# (edge values rise because high-frequency content is lost).  The DC Fourier
# coefficient — equal to the projection mean — is always in-bounds and must be
# conserved regardless of phi.
#
# Each iteration of the loop calls _projection_mean with a different array
# (different pose) but the same structure/static values → single compilation.


def test_phi_rotation_mean_conserved(fourier_grid_volume, image_config):
    """Mean of projection is conserved under in-plane phi rotations."""
    mean_id = float(_projection_mean(fourier_grid_volume, _FSE, image_config))
    for phi in [15.0, 30.0, 45.0, 60.0]:
        pose = cxs.EulerAnglePose(phi_angle=phi, theta_angle=0.0, psi_angle=0.0)
        mean_rot = float(
            _projection_mean(fourier_grid_volume.rotate_to_pose(pose), _FSE, image_config)
        )
        np.testing.assert_allclose(mean_rot, mean_id, rtol=1e-5)


# ── apply_deconvolve: should reduce interpolation error at off-axis poses ─────
#
# Deconvolution pre-whitens the voxel grid to counteract the triangular
# interpolation kernel (sinc² attenuation), which recovers high-frequency
# components that would otherwise be under-represented.
#
# Effect is most pronounced for volumes with significant high-frequency content
# (small sigma relative to grid size) at intermediate tilt angles.
# Measured on a 64³ grid, variance=4 Å² (sigma=2 px), phi=0°:
#   theta=30°, 45°, 60°: max error drops by ~30–50 % with deconvolution.
#   theta=0°, 90° (exact): deconvolution is neutral or slightly harmful
#   because no interpolation takes place and pre-whitening adds ringing.
#
# _max_abs_error compiles separately for the 64³ volume shape (different
# static pytree structure) but is reused across all three theta parametrize
# cases.


@pytest.fixture(scope="module")
def _gmm_volume_narrow():
    return cxs.GaussianMixtureVolume(np.zeros((1, 3)), amplitudes=1.0, variances=1.0)


@pytest.fixture(scope="module")
def _image_config_64():
    return cxs.BasicImageConfig((64, 64), pixel_size=1.0, voltage_in_kilovolts=300.0)


@pytest.fixture(scope="module")
def _grid_volume_plain(_gmm_volume_narrow):
    """64³ grid volume without interpolation deconvolution."""
    render_fn = cxs.GaussianMixtureRenderFn((64, 64, 64), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(_gmm_volume_narrow), apply_deconvolve=False
    )


@pytest.fixture(scope="module")
def _grid_volume_deconv(_gmm_volume_narrow):
    """64³ grid volume with interpolation deconvolution applied."""
    render_fn = cxs.GaussianMixtureRenderFn((64, 64, 64), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(_gmm_volume_narrow), apply_deconvolve=True
    )


@pytest.mark.parametrize("theta_angle", [30.0, 45.0, 60.0])
def test_deconvolve_reduces_offaxis_error(
    _gmm_volume_narrow,
    _grid_volume_plain,
    _grid_volume_deconv,
    _image_config_64,
    theta_angle,
):
    """apply_deconvolve=True must give lower max error than False at off-axis tilt.

    Tests a 64³ grid with variance=1 Å² (sigma=1 px).  Measured improvement:
      plain ~0.15–0.16 %,  deconv ~0.09–0.10 % of projection peak.
    """
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=theta_angle, psi_angle=0.0)
    gmm_rot = _gmm_volume_narrow.rotate_to_pose(pose)

    err_plain = float(
        _max_abs_error(
            gmm_rot,
            _grid_volume_plain.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            _image_config_64,
        )
    )
    err_deconv = float(
        _max_abs_error(
            gmm_rot,
            _grid_volume_deconv.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            _image_config_64,
        )
    )

    assert err_deconv < err_plain, (
        f"Deconvolution should reduce off-axis error at theta={theta_angle}: "
        f"deconv={err_deconv:.4e} plain={err_plain:.4e}"
    )


# ── error handling ─────────────────────────────────────────────────────────────


def test_odd_dimension_raises():
    """FourierVoxelGridVolume must reject odd-dimension input."""
    with pytest.raises(ValueError, match="odd"):
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(np.ones((31, 31, 31)))


def test_spline_odd_dimension_raises():
    """FourierVoxelSplineVolume must also reject odd-dimension input."""
    with pytest.raises(ValueError, match="odd"):
        cxs.FourierVoxelSplineVolume.from_real_voxel_grid(np.ones((31, 31, 31)))


def test_wrong_volume_type_raises(image_config):
    """FourierSliceExtraction must raise for unsupported volume types."""
    wrong_volume = cxs.GaussianMixtureVolume(
        np.zeros((1, 3)), amplitudes=1.0, variances=1.0
    )
    with pytest.raises((ValueError, AttributeError)):
        _FSE.integrate(wrong_volume, image_config)  # type: ignore[arg-type]
