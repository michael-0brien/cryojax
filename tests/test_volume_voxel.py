"""Tests for voxel-based volume representations and their integrators.

Covers FourierVoxelGridVolume, FourierVoxelSplineVolume, RealVoxelGridVolume,
and RealVoxelCloudVolume.  GaussianMixtureVolume / GaussianMixtureProjection
serve as analytic ground truth throughout.

FourierSliceExtraction accuracy
--------------------------------
Measured on a 32³ grid with sigma=1 px (variance=1 Å²):
- Identity / theta=90 phi=0: grid ≈ spline ≈ 1.8e-5 (limited by rendering)
- theta=45 phi=0:  grid ~0.4 %, grid+deconv ~0.2 %, spline ~0.1 % of peak

JIT strategy
------------
Module-level ``@eqx.filter_jit`` helpers are keyed on (pytree structure,
static leaf values).  Reusing ``_FSE`` and ``_GMM_INTEGRATOR`` singletons
avoids recompilation across tests that differ only in array content.

Notes on boundary behaviour
----------------------------
For in-plane phi rotations (theta=0) the rotated frequency slice extends
outside the Fourier box.  Zero-fill is correct ('fill' mode) but reduces
edge-to-peak contrast.  ``test_phi_rotation_mean_conserved`` documents that
the DC coefficient (projection mean) is preserved regardless.
"""

import math

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.constants import PengScatteringFactorParameters
from cryojax.io import read_atoms_from_pdb
from jaxtyping import Array, Float


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


# ── module-level integrator singletons ───────────────────────────────────────

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


@eqx.filter_jit
def _project_real(volume, integrator, image_config) -> Array:
    return integrator.integrate(volume, image_config, outputs_real_space=True)


@eqx.filter_jit
def _project_fourier(volume, integrator, image_config) -> Array:
    return integrator.integrate(volume, image_config, outputs_real_space=False)


@eqx.filter_jit
def _max_abs_error(volume_a, volume_b, integrator_a, integrator_b, image_config) -> Array:
    """Max pixelwise absolute difference between two projections."""
    proj_a = integrator_a.integrate(volume_a, image_config, outputs_real_space=True)
    proj_b = integrator_b.integrate(volume_b, image_config, outputs_real_space=True)
    return jnp.max(jnp.abs(proj_a - proj_b))


@eqx.filter_jit
def _projection_mean(volume, integrator, image_config) -> Array:
    proj = integrator.integrate(volume, image_config, outputs_real_space=True)
    return jnp.mean(proj)


@eqx.filter_jit
def _compute_projection(volume, integrator, image_config) -> Array:
    fourier_proj = integrator.integrate(volume, image_config, outputs_real_space=False)
    return im.crop_to_shape(
        im.irfftn(fourier_proj, s=image_config.padded_shape),
        image_config.shape,
    )


# ── render_voxel_volume ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "output_type",
    [
        cxs.FourierVoxelGridVolume,
        cxs.FourierVoxelSplineVolume,
        cxs.RealVoxelGridVolume,
        cxs.RealVoxelCloudVolume,
    ],
)
def test_render_voxel_volume_output_type(output_type, gmm_volume):
    """render_voxel_volume must return exactly the requested output type."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    result = cxs.render_voxel_volume(gmm_volume, render_fn, output_type=output_type)
    assert type(result) is output_type


@pytest.mark.parametrize(
    "output_type",
    [
        cxs.FourierVoxelGridVolume,
        cxs.FourierVoxelSplineVolume,
        cxs.RealVoxelGridVolume,
        cxs.RealVoxelCloudVolume,
    ],
)
def test_render_voxel_volume_auto_render_fn(output_type, gmm_volume):
    """AutoVolumeRenderFn must dispatch to the same result as explicit render_fn."""
    shape, voxel_size = (32, 32, 32), 1.0
    explicit_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    auto_fn = cxs.AutoVolumeRenderFn(shape, voxel_size)
    explicit = cxs.render_voxel_volume(gmm_volume, explicit_fn, output_type=output_type)
    auto = cxs.render_voxel_volume(gmm_volume, auto_fn, output_type=output_type)
    assert type(auto) is output_type
    # Both paths should produce identical results for a GMM volume
    if output_type is cxs.FourierVoxelGridVolume:
        np.testing.assert_array_equal(
            np.array(explicit.fourier_voxel_grid),
            np.array(auto.fourier_voxel_grid),
        )
    elif output_type is cxs.FourierVoxelSplineVolume:
        np.testing.assert_array_equal(
            np.array(explicit.spline_coefficients),
            np.array(auto.spline_coefficients),
        )
    elif output_type is cxs.RealVoxelGridVolume:
        np.testing.assert_array_equal(
            np.array(explicit.real_voxel_grid),
            np.array(auto.real_voxel_grid),
        )
    # RealVoxelCloudVolume: type check above is sufficient


def test_render_voxel_volume_matches_direct_construction(gmm_volume):
    """render_voxel_volume must be equivalent to from_real_voxel_grid."""
    shape, voxel_size = (32, 32, 32), 1.0
    render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    real_grid = render_fn(gmm_volume)

    for output_type, direct in [
        (
            cxs.FourierVoxelGridVolume,
            cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_grid),
        ),
        (
            cxs.FourierVoxelSplineVolume,
            cxs.FourierVoxelSplineVolume.from_real_voxel_grid(real_grid),
        ),
        (
            cxs.RealVoxelGridVolume,
            cxs.RealVoxelGridVolume.from_real_voxel_grid(real_grid),
        ),
        (
            cxs.RealVoxelCloudVolume,
            cxs.RealVoxelCloudVolume.from_real_voxel_grid(real_grid),
        ),
    ]:
        via_api = cxs.render_voxel_volume(gmm_volume, render_fn, output_type=output_type)
        if output_type is cxs.FourierVoxelGridVolume:
            np.testing.assert_array_equal(
                np.array(via_api.fourier_voxel_grid),
                np.array(direct.fourier_voxel_grid),
            )
        elif output_type is cxs.FourierVoxelSplineVolume:
            np.testing.assert_array_equal(
                np.array(via_api.spline_coefficients),
                np.array(direct.spline_coefficients),
            )
        elif output_type is cxs.RealVoxelGridVolume:
            np.testing.assert_array_equal(
                np.array(via_api.real_voxel_grid),
                np.array(direct.real_voxel_grid),
            )
        # RealVoxelCloudVolume: type identity is sufficient


def test_render_voxel_volume_projection_accuracy(gmm_volume, image_config):
    """Projecting a render_voxel_volume result must agree with GMM analytic projection."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    fourier_vol = cxs.render_voxel_volume(
        gmm_volume, render_fn, output_type=cxs.FourierVoxelGridVolume
    )
    err = float(
        _max_abs_error(gmm_volume, fourier_vol, _GMM_INTEGRATOR, _FSE, image_config)
    )
    assert err < 3e-5


# ── FourierSliceExtraction: output shape ──────────────────────────────────────


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


# ── FourierSliceExtraction: outputs_integral scaling ─────────────────────────


@pytest.mark.parametrize("pixel_size", [0.5, 1.0, 2.0])
def test_outputs_integral_scale(fourier_grid_volume, pixel_size):
    """outputs_integral=True must equal pixel_size * outputs_integral=False."""
    config = cxs.BasicImageConfig(
        (32, 32), pixel_size=pixel_size, voltage_in_kilovolts=300.0
    )
    with_scale = np.array(_project_fourier(fourier_grid_volume, _FSE, config))
    without_scale = np.array(_project_fourier(fourier_grid_volume, _FSE_NO_SCALE, config))
    np.testing.assert_allclose(with_scale, pixel_size * without_scale, atol=1e-12)


# ── FourierSliceExtraction: linearity ────────────────────────────────────────


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


# ── FourierSliceExtraction: accuracy vs GaussianMixtureProjection ─────────────


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
    """theta=90, phi=0: slice on exact grid planes; matches GMM to 5e-5."""
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
    """Off-axis bilinear+deconv accuracy: within 0.5 % of peak.

    Measured on a 32³ grid with variance=1 Å²:
      theta=30°: ~0.23 %   theta=45°: ~0.20 %   theta=60°: ~0.23 %

    With sigma=1 px, deconvolution improves over uncorrected bilinear
    (~0.35–0.41 %).  At exact poses (theta=0°, 90°) deconvolution is
    slightly harmful because no interpolation occurs there.
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


# ── FourierSliceExtraction: grid vs spline consistency ───────────────────────


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


# ── in-plane phi rotation: DC coefficient conserved ──────────────────────────


def test_phi_rotation_mean_conserved(fourier_grid_volume, image_config):
    """Mean of projection is conserved under in-plane phi rotations."""
    mean_id = float(_projection_mean(fourier_grid_volume, _FSE, image_config))
    for phi in [15.0, 30.0, 45.0, 60.0]:
        pose = cxs.EulerAnglePose(phi_angle=phi, theta_angle=0.0, psi_angle=0.0)
        mean_rot = float(
            _projection_mean(fourier_grid_volume.rotate_to_pose(pose), _FSE, image_config)
        )
        np.testing.assert_allclose(mean_rot, mean_id, rtol=1e-5)


# ── apply_deconvolve ──────────────────────────────────────────────────────────


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


# ── FourierSliceExtraction: error handling ────────────────────────────────────


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


# ── Voxel volume constructors and loaders ─────────────────────────────────────


def test_voxel_volume_loaders():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    fourier_grid = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    fourier_spline = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(real_voxel_grid)
    real_grid = cxs.RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    real_cloud = cxs.RealVoxelCloudVolume.from_real_voxel_grid(real_voxel_grid)

    assert isinstance(
        fourier_grid.frequency_slice_in_pixels,
        Float[Array, "1 _ _ 3"],  # type: ignore
    )
    assert isinstance(fourier_grid.fourier_voxel_grid, Array)
    assert isinstance(fourier_spline.spline_coefficients, Array)
    assert isinstance(
        real_grid.coordinate_grid_in_pixels,
        Float[Array, "_ _ _ 3"],  # type: ignore
    )
    assert isinstance(real_grid.real_voxel_grid, Array)
    assert isinstance(real_cloud, cxs.RealVoxelCloudVolume)


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
        cxs.RealVoxelCloudVolume,
    ]:
        assert (
            type(cxs.render_voxel_volume(atom_volume, render_fn, output_type=cls)) == cls
        )


# ── pad_scale ─────────────────────────────────────────────────────────────────


def _is_smooth(n: int) -> bool:
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@pytest.mark.parametrize("pad_scale", (1.5, 2.0))
def test_fourier_voxel_grid_pad_scale_produces_smooth_shape(pad_scale):
    shape = (10, 10, 10)
    real_voxel_grid = jnp.zeros(shape, dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        real_voxel_grid, pad_scale=pad_scale
    )
    padded_shape = vol.fourier_voxel_grid.shape
    for s, p in zip(shape, padded_shape):
        assert p >= math.ceil(pad_scale * s)
        assert _is_smooth(p)


def test_fourier_voxel_grid_pad_scale_one_unchanged():
    shape = (10, 10, 10)
    real_voxel_grid = jnp.zeros(shape, dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=1.0)
    assert vol.fourier_voxel_grid.shape == shape


def test_fourier_voxel_grid_pad_scale_less_than_one_raises():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    with pytest.raises(ValueError, match="pad_scale"):
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=0.5)


@pytest.mark.parametrize("pad_scale", (1.5, 2.0))
def test_fourier_voxel_spline_pad_scale_produces_smooth_shape(pad_scale):
    shape = (10, 10, 10)
    real_voxel_grid = jnp.zeros(shape, dtype=float)
    vol = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(
        real_voxel_grid, pad_scale=pad_scale
    )
    padded_shape = vol.spline_coefficients.shape
    for s, p in zip(shape, padded_shape):
        assert p >= math.ceil(pad_scale * s)
        assert _is_smooth(p - 2)


def test_fourier_voxel_spline_pad_scale_one_unchanged():
    dim = 10
    real_voxel_grid = jnp.zeros((dim, dim, dim), dtype=float)
    vol = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(
        real_voxel_grid, pad_scale=1.0
    )
    assert all(d - 2 == dim for d in vol.spline_coefficients.shape)


def test_fourier_voxel_spline_pad_scale_less_than_one_raises():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    with pytest.raises(ValueError, match="pad_scale"):
        cxs.FourierVoxelSplineVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=0.5)


# ── Fourier vs real voxel agreement ──────────────────────────────────────────


def test_fourier_vs_real_agreement(sample_pdb_path):
    """FourierVoxelGridVolume and RealVoxelGridVolume must agree when loaded
    from the same PDB."""
    shape = (128, 128, 128)
    voxel_size = 0.5

    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
        selection_string="not element H",
    )
    render_fn = cxs.AutoVolumeRenderFn(shape, voxel_size)
    fourier_volume = cxs.render_voxel_volume(
        atom_volume, render_fn, output_type=cxs.FourierVoxelGridVolume
    )
    real_volume = cxs.render_voxel_volume(
        atom_volume, render_fn, output_type=cxs.RealVoxelGridVolume
    )
    real_voxel_grid = jnp.fft.fftshift(
        im.ifftn(jnp.fft.ifftshift(fourier_volume.fourier_voxel_grid)).real
    )

    np.testing.assert_allclose(real_voxel_grid, real_volume.real_voxel_grid, atol=1e-12)


def test_downsampled_voxel_volume_agreement(sample_pdb_path):
    """Rasterized voxel grid must roughly agree with a downsampled version."""
    shape = (128, 128, 128)
    voxel_size = 0.25
    downsampling_factor = 2
    downsampled_shape = tuple(s // downsampling_factor for s in shape)
    downsampled_voxel_size = voxel_size * downsampling_factor

    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
    )
    atom_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    lowres_render_fn = cxs.GaussianMixtureRenderFn(
        downsampled_shape,  # type: ignore
        downsampled_voxel_size,
    )
    low_resolution_volume_grid = lowres_render_fn(atom_volume)
    highres_render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    high_resolution_volume_grid = highres_render_fn(atom_volume)
    downsampled_volume_grid = im.fourier_crop_downsample(
        high_resolution_volume_grid, downsampling_factor
    )

    assert low_resolution_volume_grid.shape == downsampled_volume_grid.shape


def test_compute_rectangular_voxel_grid(sample_pdb_path):
    shape = (128, 127, 126)
    voxel_size = 0.5

    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
    )
    atom_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
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

    atom_positions, atom_types = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        loads_b_factors=False,
        selection_string="not element H",
    )
    atom_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atom_types),
    )
    render_fn = cxs.GaussianMixtureRenderFn(shape, voxel_size)
    voxels = render_fn(atom_volume)
    batched_render_fn = cxs.GaussianMixtureRenderFn(
        shape,
        voxel_size,
        batch_options=dict(batch_size=batch_size, n_batches=n_batches),
    )
    voxels_with_batching = batched_render_fn(atom_volume)
    np.testing.assert_allclose(voxels, voxels_with_batching)


# ── Voxel volumes vs GMM analytic projection ─────────────────────────────────


def _make_gmm_voxel_scene(pdb_info):
    """Shared setup: GMM volume, render fn, and image config for nopose tests.

    Uses a 64³ grid at 1.0 Å/px (64 Å box) to ensure the molecule fits.
    """
    atom_positions, atom_types, atom_properties = pdb_info
    pixel_size, shape = 0.25, (128, 128)
    dim = max(*shape)
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    peng_parameters = PengScatteringFactorParameters(atom_types)
    gmm_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        peng_parameters,
        extra_b_factors=5.0,
    )
    render_fn = cxs.GaussianMixtureRenderFn((dim, dim, dim), pixel_size)
    real_voxel_grid = render_fn(gmm_volume)
    return gmm_volume, real_voxel_grid, image_config


@pytest.mark.parametrize(
    "volume_cls, integrator, tol",
    [
        (
            cxs.FourierVoxelGridVolume,
            cxs.FourierSliceExtraction(),
            1e-2,
        ),
        (
            cxs.FourierVoxelSplineVolume,
            cxs.FourierSliceExtraction(),
            1e-5,
        ),
        (
            cxs.RealVoxelCloudVolume,
            cxs.RealVoxelProjection(eps=1e-16, backend="nufftax"),
            1e-10,
        ),
    ],
    ids=["fourier_grid", "fourier_spline", "real_cloud_nufftax"],
)
def test_gaussian_vs_voxels_nopose(pdb_info, volume_cls, integrator, tol):
    """Voxel-based projection must agree with analytic GMM projection at identity pose."""
    gmm_volume, real_voxel_grid, image_config = _make_gmm_voxel_scene(pdb_info)
    gmm_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    volume = volume_cls.from_real_voxel_grid(real_voxel_grid)
    proj_ref = _compute_projection(gmm_volume, gmm_integrator, image_config)
    proj = _compute_projection(volume, integrator, image_config)

    np.testing.assert_allclose(proj_ref, proj, atol=tol)


@pytest.mark.skipif(jnufft is None, reason="jax-finufft not installed")
def test_gaussian_vs_voxels_nopose_jax_finufft(pdb_info):
    """RealVoxelProjection with jax-finufft must agree with GMM at identity pose."""
    gmm_volume, real_voxel_grid, image_config = _make_gmm_voxel_scene(pdb_info)
    gmm_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    cloud_volume = cxs.RealVoxelCloudVolume.from_real_voxel_grid(real_voxel_grid)
    integrator = cxs.RealVoxelProjection(eps=1e-16, backend="jax-finufft")  # type: ignore
    proj_ref = _compute_projection(gmm_volume, gmm_integrator, image_config)
    proj = _compute_projection(cloud_volume, integrator, image_config)
    np.testing.assert_allclose(proj_ref, proj, atol=1e-8)
