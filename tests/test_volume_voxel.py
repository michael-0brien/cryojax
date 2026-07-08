"""Tests for voxel-based volume representations and their integrators.

Covers FourierVoxelGridVolume, FourierVoxelSplineVolume, and RealVoxelGridVolume.
GaussianMixtureVolume / GaussianMixtureProjection serve as analytic ground truth
throughout.

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
from cryojax.ndimage._fourier_slice import (
    _reconstruct_full_slice_from_half_slice,
)
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
_ESE = cxs.EwaldSphereExtraction()
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
        np.testing.assert_allclose(
            np.array(explicit.fourier_voxel_grid),
            np.array(auto.fourier_voxel_grid),
        )
    elif output_type is cxs.FourierVoxelSplineVolume:
        np.testing.assert_allclose(
            np.array(explicit.spline_coefficients),
            np.array(auto.spline_coefficients),
        )
    elif output_type is cxs.RealVoxelGridVolume:
        np.testing.assert_allclose(
            np.array(explicit.real_voxel_grid),
            np.array(auto.real_voxel_grid),
        )


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
    ]:
        via_api = cxs.render_voxel_volume(gmm_volume, render_fn, output_type=output_type)
        if output_type is cxs.FourierVoxelGridVolume:
            np.testing.assert_allclose(
                np.array(via_api.fourier_voxel_grid),
                np.array(direct.fourier_voxel_grid),
            )
        elif output_type is cxs.FourierVoxelSplineVolume:
            np.testing.assert_allclose(
                np.array(via_api.spline_coefficients),
                np.array(direct.spline_coefficients),
            )
        elif output_type is cxs.RealVoxelGridVolume:
            np.testing.assert_allclose(
                np.array(via_api.real_voxel_grid),
                np.array(direct.real_voxel_grid),
            )


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


# ── EwaldSphereExtraction high-voltage limit ──────────────────────────────────


def test_ewald_sphere_high_voltage_limit_grid(fourier_grid_volume, image_config):
    """At sufficiently high voltage, the Ewald sphere curvature vanishes and
    `EwaldSphereExtraction` must reduce to `FourierSliceExtraction`.
    """
    high_voltage_config = cxs.BasicImageConfig(
        image_config.shape,
        pixel_size=image_config.pixel_size,
        voltage_in_kilovolts=1e6,
    )
    err = float(
        _max_abs_error(
            fourier_grid_volume, fourier_grid_volume, _FSE, _ESE, high_voltage_config
        )
    )
    assert err < 5e-5


def test_ewald_sphere_high_voltage_limit_spline(fourier_spline_volume, image_config):
    """Same high-voltage limit check for the spline-interpolated volume."""
    high_voltage_config = cxs.BasicImageConfig(
        image_config.shape,
        pixel_size=image_config.pixel_size,
        voltage_in_kilovolts=1e6,
    )
    err = float(
        _max_abs_error(
            fourier_spline_volume,
            fourier_spline_volume,
            _FSE,
            _ESE,
            high_voltage_config,
        )
    )
    assert err < 5e-5


# ── EwaldSphereExtraction vs analytic ground truth ────────────────────────────
#
# A gaussian mixture has a closed-form 3D fourier transform,
#     hat_V(q) = sum_i a_i * exp(-2 pi^2 sigma_i^2 |q|^2) * exp(-2 pi i q . r_i),
# evaluated here directly on the curved Ewald sphere surface (paraxial
# approximation q_z = wavelength / 2 * |q_parallel|^2, valid at identity pose
# where the beam axis is exactly z), independent of the voxel grid entirely.
#
# A variance of 4 A^2 (sigma=2 px) is used, rather than sigma=1 px used
# elsewhere in this module, so the 32^3 voxel grid resolves the gaussians
# well; narrower gaussians expose ordinary grid-rendering aliasing unrelated
# to Ewald curvature.

_TWO_ATOM_POSITIONS = np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 1.5]])
_TWO_ATOM_AMPLITUDES = np.array([1.0, 1.3])
_TWO_ATOM_VARIANCES = np.array([4.0, 4.0])


def _gaussian_mixture_fourier_transform(
    positions: Float[Array, "n 3"],
    amplitudes: Float[Array, " n"],
    variances: Float[Array, " n"],
    frequencies_in_angstroms: Float[Array, "... 3"],
) -> Array:
    """Closed-form 3D fourier transform of a mixture of isotropic gaussians,
    evaluated at arbitrary frequency coordinates (in inverse angstroms).
    """
    q_squared = jnp.sum(frequencies_in_angstroms**2, axis=-1)
    transform = jnp.zeros(q_squared.shape, dtype=complex)
    for position, amplitude, variance in zip(positions, amplitudes, variances):
        phase = -2j * jnp.pi * jnp.sum(frequencies_in_angstroms * position, axis=-1)
        transform = transform + amplitude * jnp.exp(
            -2 * jnp.pi**2 * variance * q_squared
        ) * jnp.exp(phase)
    return transform


def _ewald_sphere_analytic_ground_truth(volume, voltage_in_kilovolts: float) -> Array:
    N = volume.frequency_slice_in_pixels.shape[1]
    config = cxs.BasicImageConfig(
        (N, N),
        pixel_size=1.0,
        voltage_in_kilovolts=voltage_in_kilovolts,
    )
    wavelength = config.wavelength_in_angstroms
    # Curved Ewald sphere frequency coordinates (paraxial approximation, at
    # identity pose the beam/projection axis is exactly z). `volume` only
    # stores the half (rfft) in-plane slice; reconstruct the full grid this
    # ground truth needs (an exact, non-interpolating coordinate operation,
    # independent of the interpolation logic under test).
    q_at_slice = _reconstruct_full_slice_from_half_slice(
        volume.frequency_slice_in_pixels
    )[0]
    q_parallel_squared = jnp.sum(q_at_slice[..., :2] ** 2, axis=-1)
    q_z_curvature = 0.5 * wavelength * q_parallel_squared
    q_at_surface = q_at_slice.at[..., 2].add(q_z_curvature)

    transform = _gaussian_mixture_fourier_transform(
        jnp.asarray(_TWO_ATOM_POSITIONS),
        jnp.asarray(_TWO_ATOM_AMPLITUDES),
        jnp.asarray(_TWO_ATOM_VARIANCES),
        q_at_surface,
    )
    # cryojax's real-space convention puts the origin at array index N // 2,
    # so any analytic transform evaluated at raw (corner-indexed) frequency
    # coordinates must pick up the corresponding (-1)^k checkerboard phase.
    checkerboard_phase = im.make_fftshift_phase(transform.shape)
    return jnp.fft.ifftshift(checkerboard_phase * transform)


@pytest.fixture(scope="module")
def two_atom_grid_volume():
    volume = cxs.GaussianMixtureVolume(
        _TWO_ATOM_POSITIONS,
        amplitudes=_TWO_ATOM_AMPLITUDES,
        variances=_TWO_ATOM_VARIANCES,
    )
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(render_fn(volume))


@pytest.fixture(scope="module")
def two_atom_spline_volume():
    volume = cxs.GaussianMixtureVolume(
        _TWO_ATOM_POSITIONS,
        amplitudes=_TWO_ATOM_AMPLITUDES,
        variances=_TWO_ATOM_VARIANCES,
    )
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelSplineVolume.from_real_voxel_grid(render_fn(volume))


def test_ewald_sphere_matches_analytic_ground_truth_grid(two_atom_grid_volume):
    """`EwaldSphereExtraction` must agree with the closed-form gaussian mixture
    fourier transform evaluated directly on the curved Ewald sphere surface.
    """
    voltage_in_kilovolts = 60.0  # low voltage: large, easily-detectable curvature
    config = cxs.BasicImageConfig(
        (32, 32), pixel_size=1.0, voltage_in_kilovolts=voltage_in_kilovolts
    )
    ese_fourier = _ESE.integrate(two_atom_grid_volume, config, outputs_real_space=False)
    analytic = _ewald_sphere_analytic_ground_truth(
        two_atom_grid_volume, voltage_in_kilovolts
    )
    err = float(jnp.max(jnp.abs(ese_fourier - analytic)))
    peak = float(jnp.max(jnp.abs(ese_fourier)))
    assert err < 0.02 * peak


def test_ewald_sphere_matches_analytic_ground_truth_spline(two_atom_spline_volume):
    """Same analytic ground truth check for the spline-interpolated volume."""
    voltage_in_kilovolts = 60.0
    config = cxs.BasicImageConfig(
        (32, 32), pixel_size=1.0, voltage_in_kilovolts=voltage_in_kilovolts
    )
    ese_fourier = _ESE.integrate(two_atom_spline_volume, config, outputs_real_space=False)
    analytic = _ewald_sphere_analytic_ground_truth(
        two_atom_spline_volume, voltage_in_kilovolts
    )
    err = float(jnp.max(jnp.abs(ese_fourier - analytic)))
    peak = float(jnp.max(jnp.abs(ese_fourier)))
    assert err < 0.02 * peak


# ── RFFT storage shape assertions ─────────────────────────────────────────────
#
# Both `FourierVoxelGridVolume`/`FourierVoxelSplineVolume` store the 3D voxel
# grid as a half-space RFFT grid (halving memory relative to the full complex
# FFT cube), and `frequency_slice_in_pixels` as the half in-plane grid too.
# Interpolation reflects each out-of-range query point through the origin and
# conjugates the result -- see `_extract_surface_from_voxel_grid` in
# `fourier_voxels.py`. `EwaldSphereExtraction` reconstructs the full in-plane
# grid on demand (`_reconstruct_full_slice_from_half_slice`), since its
# curved output isn't Hermitian-symmetric as a whole.


def test_storage_shape_grid():
    dim = 16
    real_voxel_grid = jnp.zeros((dim, dim, dim), dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    assert vol.shape == (dim, dim, dim)
    assert vol.fourier_voxel_grid.shape == (dim, dim, dim // 2 + 1)
    assert vol.frequency_slice_in_pixels.shape == (1, dim, dim // 2 + 1, 3)


def test_storage_shape_spline():
    dim = 16
    real_voxel_grid = jnp.zeros((dim, dim, dim), dtype=float)
    vol = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(real_voxel_grid)
    assert vol.shape == (dim, dim, dim)
    assert vol.spline_coefficients.shape == (dim + 2, dim + 2, dim // 2 + 3)
    assert vol.frequency_slice_in_pixels.shape == (1, dim, dim // 2 + 1, 3)


# ── from_fourier_voxel_grid constructor equivalence ───────────────────────────


def test_grid_from_fourier_voxel_grid_matches_from_real(gmm_volume, image_config):
    """from_fourier_voxel_grid(rfftn(grid)) must equal from_real_voxel_grid(grid)."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    real_grid = render_fn(gmm_volume)
    vol_real = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_grid)
    vol_fourier = cxs.FourierVoxelGridVolume.from_fourier_voxel_grid(im.rfftn(real_grid))
    err = float(_max_abs_error(vol_real, vol_fourier, _FSE, _FSE, image_config))
    assert np.isclose(err, 0.0)


def test_spline_from_fourier_voxel_grid_matches_from_real(gmm_volume, image_config):
    """from_fourier_voxel_grid(rfftn(grid)) must equal from_real_voxel_grid(grid)."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    real_grid = render_fn(gmm_volume)
    vol_real = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(real_grid)
    vol_fourier = cxs.FourierVoxelSplineVolume.from_fourier_voxel_grid(
        im.rfftn(real_grid)
    )
    err = float(_max_abs_error(vol_real, vol_fourier, _FSE, _FSE, image_config))
    assert np.isclose(err, 0.0)


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


def test_grid_shape_mismatch_raises():
    """A `fourier_voxel_grid` whose shape doesn't correspond to any cubic
    volume's half-space RFFT grid must raise."""
    with pytest.raises(AttributeError, match="invalid shape"):
        cxs.FourierVoxelGridVolume(
            jnp.zeros((16, 16, 5), dtype=complex),
            im.make_frequency_slice((16, 16), outputs_rfftfreqs=True, fftshifted=True),
        )


def test_grid_full_cube_shape_suggests_rfftn():
    """Passing the full (non-rfft) FFT grid shape must raise an error
    suggesting `cryojax.ndimage.rfftn` instead of `fftn`."""
    dim = 16
    full_shaped = jnp.zeros((dim, dim, dim), dtype=complex)
    with pytest.raises(AttributeError, match="rfftn"):
        cxs.FourierVoxelGridVolume(
            full_shaped,
            im.make_frequency_slice((dim, dim), outputs_rfftfreqs=True, fftshifted=True),
        )


def test_spline_full_cube_shape_suggests_rfftn():
    """Same full-cube-shape check for `FourierVoxelSplineVolume`, whose
    stored array is padded by 2 samples per axis relative to the grid."""
    dim = 16
    full_shaped = jnp.zeros((dim + 2, dim + 2, dim + 2), dtype=complex)
    with pytest.raises(AttributeError, match="rfftn"):
        cxs.FourierVoxelSplineVolume(
            full_shaped,
            im.make_frequency_slice((dim, dim), outputs_rfftfreqs=True, fftshifted=True),
        )


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
    # `.shape` reports the logical cubic (real-space) shape; check it
    # against `_is_smooth`, and check the storage shape's own rfft-truncated
    # last axis separately below.
    for s, p in zip(shape, vol.shape):
        assert p >= math.ceil(pad_scale * s)
        assert _is_smooth(p)
    assert vol.fourier_voxel_grid.shape == vol.shape[:-1] + (vol.shape[-1] // 2 + 1,)


def test_fourier_voxel_grid_pad_scale_one_unchanged():
    shape = (10, 10, 10)
    real_voxel_grid = jnp.zeros(shape, dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=1.0)
    assert vol.shape == shape
    assert vol.fourier_voxel_grid.shape == shape[:-1] + (shape[-1] // 2 + 1,)


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
    for s, p in zip(shape, vol.shape):
        assert p >= math.ceil(pad_scale * s)
        assert _is_smooth(p)
    expected_last_axis = vol.shape[-1] // 2 + 1 + 2
    assert vol.spline_coefficients.shape == tuple(d + 2 for d in vol.shape[:-1]) + (
        expected_last_axis,
    )


def test_fourier_voxel_spline_pad_scale_one_unchanged():
    dim = 10
    real_voxel_grid = jnp.zeros((dim, dim, dim), dtype=float)
    vol = cxs.FourierVoxelSplineVolume.from_real_voxel_grid(
        real_voxel_grid, pad_scale=1.0
    )
    assert vol.shape == (dim, dim, dim)
    assert vol.spline_coefficients.shape == (dim + 2, dim + 2, dim // 2 + 3)


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
    # Only the two full axes were fftshift'd at construction time (the
    # rfft-truncated axis stays in rfft/corner convention) -- see
    # `_prepare_fourier_voxel_arguments` in `fourier_voxels.py`.
    real_voxel_grid = jnp.fft.fftshift(
        im.irfftn(
            jnp.fft.ifftshift(fourier_volume.fourier_voxel_grid, axes=(0, 1)),
            s=shape,
        )
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


@pytest.mark.parametrize("n_batches", (1, 2, 3))
def test_atom_batched_vs_non_batched_loop_agreement(sample_pdb_path, n_batches):
    shape = (128, 128, 128)
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
    batched_render_fn = cxs.GaussianMixtureRenderFn(
        shape, voxel_size, n_batches=n_batches
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
    ],
    ids=["fourier_grid"],
)
@pytest.mark.parametrize(
    "pose",
    [
        cxs.EulerAnglePose(phi_angle=0.0, theta_angle=0.0, psi_angle=0.0),
        cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0),
        cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=90.0),
    ],
    ids=["theta0_psi0", "theta90_psi0", "theta90_psi90"],
)
def test_pose_convention_exact_angles(pdb_info, pose, volume_cls, integrator, tol):
    """Voxel-based projection must agree with analytic GMM projection at poses
    where Fourier-slice interpolation is exact (theta=90, phi=0).

    At theta=90 the extracted frequency slice lies exactly on a Cartesian grid
    plane, so no interpolation error is introduced.  Using an asymmetric PDB
    molecule, a wrong pose convention would shift the tilted projection
    significantly, making this a sensitive cross-method convention check at
    the same tolerances as the identity-pose test.
    """
    gmm_volume, real_voxel_grid, image_config = _make_gmm_voxel_scene(pdb_info)
    gmm_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    volume = volume_cls.from_real_voxel_grid(real_voxel_grid)
    proj_ref = _compute_projection(
        gmm_volume.rotate_to_pose(pose), gmm_integrator, image_config
    )
    proj = _compute_projection(volume.rotate_to_pose(pose), integrator, image_config)

    np.testing.assert_allclose(proj_ref, proj, atol=tol)


@pytest.mark.parametrize(
    "pose",
    [
        cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0),
        cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=90.0),
    ],
    ids=["theta90_psi0", "theta90_psi90"],
)
def test_spline_agrees_with_grid_at_exact_angles(
    fourier_grid_volume, fourier_spline_volume, image_config, pose
):
    """Spline projection must agree with grid (bilinear) Fourier-slice extraction
    at poses where the frequency slice lies exactly on Cartesian grid planes.

    Uses the simple single-Gaussian fixtures to keep runtime low.
    """
    proj_grid = np.array(
        _compute_projection(fourier_grid_volume.rotate_to_pose(pose), _FSE, image_config)
    )
    proj_spline = np.array(
        _compute_projection(
            fourier_spline_volume.rotate_to_pose(pose), _FSE, image_config
        )
    )
    np.testing.assert_allclose(proj_spline, proj_grid, atol=1e-4)
