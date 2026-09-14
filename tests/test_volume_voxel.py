"""Tests for voxel-based volume representations and their integrators.

Covers FourierVoxelGridVolume (with each of its `interp` methods) and
RealVoxelGridVolume. GaussianMixtureVolume / GaussianMixtureProjection serve as
analytic ground truth throughout.

FourierSliceExtraction accuracy
--------------------------------
Both interp methods deconvolve their own interpolation kernel's transfer function
(sinc^2 for 'linear', sinc^4 for 'cubic') out of the voxel grid in advance, so the
only error left is aliasing -- which the cubic kernel suppresses far more strongly.

Measured on a 32³ grid with sigma=1 px (variance=1 Å²):
- 'cubic' is accurate at *every* pose, and at poses landing on grid nodes it is
  limited only by the voxel rendering (~2e-5).
- 'linear' is ~0.4-0.5 % of peak off-axis, and is *worse* at grid-node poses
  (identity, theta=90), where the trilinear kernel degenerates to a delta and so
  never re-applies the sinc^2 blur that the deconvolution removed.

Tests that need node-exactness therefore use interp='cubic', which gets there
honestly (it is accurate everywhere) rather than by kernel degeneracy.

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
from cryojax.ndimage._fourier_slice import _full_slice_from_half_slice
from cryojax.simulator._volume.fourier_voxels import _resolve_unroll
from jaxtyping import Array, Float


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


# ── module-level integrator singletons ───────────────────────────────────────

_FSE = cxs.FourierSliceExtraction()
_ESE = cxs.EwaldSphereExtraction()
_GMM_INTEGRATOR = cxs.GaussianMixtureProjection(sampling_mode="average")


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gmm_volume():
    """Single Gaussian at origin (amplitude=1, variance=1 Å², sigma=1 px)."""
    return cxs.GaussianMixtureVolume(np.zeros((1, 3)), amplitudes=1.0, variances=1.0)


@pytest.fixture(scope="module")
def fourier_grid_volume(gmm_volume):
    """Trilinear (sinc^2-deconvolved) volume, rendered from the GMM volume."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(gmm_volume), interp="linear"
    )


@pytest.fixture(scope="module")
def fourier_cubic_volume(gmm_volume):
    """Cubic B-spline volume with sinc^4 deconvolution, same GMM."""
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(gmm_volume), interp="cubic"
    )


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
        jnp.fft.irfftn(fourier_proj, s=image_config.padded_shape),
        image_config.shape,
    )


# ── render_voxel_volume ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "output_type",
    [
        cxs.FourierVoxelGridVolume,
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
            np.array(explicit.values),
            np.array(auto.values),
        )
    elif output_type is cxs.RealVoxelGridVolume:
        np.testing.assert_allclose(
            np.array(explicit.values),
            np.array(auto.values),
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
            cxs.RealVoxelGridVolume,
            cxs.RealVoxelGridVolume.from_real_voxel_grid(real_grid),
        ),
    ]:
        via_api = cxs.render_voxel_volume(gmm_volume, render_fn, output_type=output_type)
        np.testing.assert_allclose(np.array(via_api.values), np.array(direct.values))


def test_render_voxel_volume_projection_accuracy(gmm_volume, image_config):
    """Projecting a render_voxel_volume result must agree with GMM analytic projection.

    `render_voxel_volume` builds the volume with the default interp, 'linear'. At
    the identity pose the frequency slice lands exactly on grid nodes, where the
    trilinear kernel degenerates to a delta and so never re-applies the sinc^2
    blur that the deconvolution removed -- hence the loose tolerance here. Use
    interp='cubic' if node-pose accuracy matters.
    """
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    fourier_vol = cxs.render_voxel_volume(
        gmm_volume, render_fn, output_type=cxs.FourierVoxelGridVolume
    )
    proj_gmm = np.array(_project_real(gmm_volume, _GMM_INTEGRATOR, image_config))
    err = float(
        _max_abs_error(gmm_volume, fourier_vol, _GMM_INTEGRATOR, _FSE, image_config)
    )
    assert err < 0.005 * float(proj_gmm.max())


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
    """'linear' at identity. The frequency slice lands on grid nodes, where the
    trilinear kernel degenerates to a delta and never re-applies the sinc^2 blur
    that the deconvolution removed -- so this is 'linear''s *worst* case, not its
    best. Compare `test_accuracy_identity_cubic`, which holds 5e-5 here.
    """
    proj_gmm = np.array(_project_real(gmm_volume, _GMM_INTEGRATOR, image_config))
    err = float(
        _max_abs_error(
            gmm_volume, fourier_grid_volume, _GMM_INTEGRATOR, _FSE, image_config
        )
    )
    assert err < 0.005 * float(proj_gmm.max())


def test_accuracy_identity_cubic(gmm_volume, fourier_cubic_volume, image_config):
    """Cubic at identity must match GMM to within rendering discretization."""
    err = float(
        _max_abs_error(
            gmm_volume, fourier_cubic_volume, _GMM_INTEGRATOR, _FSE, image_config
        )
    )
    assert err < 5e-5


def test_accuracy_theta90_phi0_grid(gmm_volume, fourier_grid_volume, image_config):
    """theta=90, phi=0: another grid-node pose, so the same caveat as
    `test_accuracy_identity_grid` applies to 'linear'."""
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0)
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
    assert err < 0.005 * float(proj_gmm.max())


def test_accuracy_theta90_phi0_cubic(gmm_volume, fourier_cubic_volume, image_config):
    """Same exact case for the cubic volume."""
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=90.0, psi_angle=0.0)
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_cubic_volume.rotate_to_pose(pose),
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
    """Off-axis 'linear' accuracy: must agree with GMM to within 0.5 % of peak.

    Measured on a 32³ grid with variance=1 Å²:
      theta=30°: ~0.23 %   theta=45°: ~0.20 %   theta=60°: ~0.23 %

    These are the poses 'linear' is good at -- unlike the grid-node poses, where
    its kernel degenerates to a delta. Compare `test_accuracy_offaxis_cubic`.
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
    assert err < 0.005 * float(proj_gmm.max())


@pytest.mark.parametrize("theta_angle", [30.0, 45.0, 60.0])
def test_accuracy_offaxis_cubic(
    gmm_volume, fourier_cubic_volume, image_config, theta_angle
):
    """Off-axis cubic accuracy: must agree with GMM to within 0.5 % of peak.

    Measured on a 32³ grid with variance=1 Å²: ~0.10–0.15 % at these tilts.
    """
    pose = cxs.EulerAnglePose(phi_angle=0.0, theta_angle=theta_angle, psi_angle=0.0)
    proj_gmm = np.array(
        _project_real(gmm_volume.rotate_to_pose(pose), _GMM_INTEGRATOR, image_config)
    )
    err = float(
        _max_abs_error(
            gmm_volume.rotate_to_pose(pose),
            fourier_cubic_volume.rotate_to_pose(pose),
            _GMM_INTEGRATOR,
            _FSE,
            image_config,
        )
    )
    assert err < 0.005 * float(proj_gmm.max())


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


def test_ewald_sphere_high_voltage_limit_cubic(fourier_cubic_volume, image_config):
    """Same high-voltage limit check for the cubic-interpolated volume."""
    high_voltage_config = cxs.BasicImageConfig(
        image_config.shape,
        pixel_size=image_config.pixel_size,
        voltage_in_kilovolts=1e6,
    )
    err = float(
        _max_abs_error(
            fourier_cubic_volume,
            fourier_cubic_volume,
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
    N = volume.frequency_slice.shape[1]
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
    q_at_slice = _full_slice_from_half_slice(volume.frequency_slice)[0]
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
def two_atom_cubic_volume():
    volume = cxs.GaussianMixtureVolume(
        _TWO_ATOM_POSITIONS,
        amplitudes=_TWO_ATOM_AMPLITUDES,
        variances=_TWO_ATOM_VARIANCES,
    )
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)
    # interp='cubic': the Ewald surface hugs the q_z = 0 grid plane at low
    # frequency, where 'linear' is at its worst (its kernel degenerates to a
    # delta there and never re-applies the sinc^2 blur the deconvolution
    # removed, putting it ~6.7 % off -- well outside the 2 % tolerance below).
    # This test is about Ewald geometry, not interpolation, so use the method
    # that is accurate everywhere.
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        render_fn(volume), interp="cubic"
    )


def test_ewald_sphere_matches_analytic_ground_truth(two_atom_cubic_volume):
    """`EwaldSphereExtraction` must agree with the closed-form gaussian mixture
    fourier transform evaluated directly on the curved Ewald sphere surface.
    """
    voltage_in_kilovolts = 60.0  # low voltage: large, easily-detectable curvature
    config = cxs.BasicImageConfig(
        (32, 32), pixel_size=1.0, voltage_in_kilovolts=voltage_in_kilovolts
    )
    ese_fourier = _ESE.integrate(two_atom_cubic_volume, config, outputs_real_space=False)
    analytic = _ewald_sphere_analytic_ground_truth(
        two_atom_cubic_volume, voltage_in_kilovolts
    )
    err = float(jnp.max(jnp.abs(ese_fourier - analytic)))
    peak = float(jnp.max(jnp.abs(ese_fourier)))
    assert err < 0.02 * peak


# ── RFFT storage shape assertions ─────────────────────────────────────────────
#
# `FourierVoxelGridVolume` stores the 3D voxel grid as a half-space RFFT grid
# (halving memory relative to the full complex FFT cube), and
# `frequency_slice` as the half in-plane grid too. The storage shape is
# the same for every `interp`. Interpolation reflects each query point with
# `q_x < 0` through the origin and conjugates the result, and recovers taps that
# fall outside the stored half by Hermitian symmetry -- see
# `_sample_half_space_grid` in `cryojax/ndimage/_fourier_slice.py`.
# `EwaldSphereExtraction` reconstructs the full in-plane grid on demand
# (`_full_slice_from_half_slice`), since its curved output isn't
# Hermitian-symmetric as a whole.


def test_storage_shape_grid():
    dim = 16
    real_voxel_grid = jnp.zeros((dim, dim, dim), dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    assert vol.shape == (dim, dim, dim)
    assert vol.values.shape == (dim, dim, dim // 2 + 1)
    assert vol.frequency_slice.shape == (1, dim, dim // 2 + 1, 3)


@pytest.mark.parametrize("interp", ("linear", "cubic"))
def test_storage_shape_is_same_for_every_interp(interp):
    """Every `interp` stores the plain half-space RFFT grid: the methods differ
    only in the deconvolution applied before the transform, never in shape."""
    dim = 16
    real_voxel_grid = jnp.zeros((dim, dim, dim), dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, interp=interp)
    assert vol.interp == interp
    assert vol.shape == (dim, dim, dim)
    assert vol.values.shape == (dim, dim, dim // 2 + 1)
    assert vol.frequency_slice.shape == (1, dim, dim // 2 + 1, 3)


def test_invalid_interp_raises():
    with pytest.raises(ValueError, match="interp"):
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(
            jnp.zeros((16, 16, 16)),
            interp="quintic",  # type: ignore
        )


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


# ── FourierSliceExtraction: error handling ────────────────────────────────────


def test_odd_dimension_raises():
    """FourierVoxelGridVolume must reject odd-dimension input."""
    with pytest.raises(ValueError, match="odd"):
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(np.ones((31, 31, 31)))


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
    real_grid = cxs.RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)

    assert isinstance(
        fourier_grid.frequency_slice,
        Float[Array, "1 _ _ 3"],  # type: ignore
    )
    assert isinstance(fourier_grid.values, Array)
    assert isinstance(
        real_grid.coordinate_grid,
        Float[Array, "_ _ _ 3"],  # type: ignore
    )
    assert isinstance(real_grid.values, Array)


def test_render_voxels(sample_pdb_path):
    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
    )
    render_fn = cxs.AutoVolumeRenderFn((16, 16, 16), voxel_size=4.0)
    for cls in [
        cxs.FourierVoxelGridVolume,
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
    assert vol.values.shape == vol.shape[:-1] + (vol.shape[-1] // 2 + 1,)


def test_fourier_voxel_grid_pad_scale_one_unchanged():
    shape = (10, 10, 10)
    real_voxel_grid = jnp.zeros(shape, dtype=float)
    vol = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=1.0)
    assert vol.shape == shape
    assert vol.values.shape == shape[:-1] + (shape[-1] // 2 + 1,)


def test_fourier_voxel_grid_pad_scale_less_than_one_raises():
    real_voxel_grid = jnp.zeros((10, 10, 10), dtype=float)
    with pytest.raises(ValueError, match="pad_scale"):
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=0.5)


# ── Fourier vs real voxel agreement ──────────────────────────────────────────


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
            5e-2,
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
        cxs.EulerAnglePose(phi_angle=27.0, theta_angle=53.0, psi_angle=41.0),
    ],
    ids=["theta0_psi0", "theta90_psi0", "theta90_psi90", "off_axis"],
)
def test_pose_convention_exact_angles(pdb_info, pose, volume_cls, integrator, tol):
    """Voxel-based projection must agree with the analytic GMM projection, for an
    asymmetric PDB molecule, across grid-aligned *and* off-axis poses.

    This pins the pose convention, not the interpolation accuracy: a wrong
    convention displaces the molecule by ångströms and produces an error of order
    the projection peak (~6), which is ~100x the tolerance below.

    interp='cubic' is used because it is the only method accurate at grid-node
    poses. On this scene, at theta=0/90, 'cubic' is off by 0.38 % of peak while
    'linear' is off by 14 % -- there the trilinear kernel degenerates to a delta,
    so it never re-applies the sinc^2 blur that its deconvolution removed.

    `tol` is set from what cubic actually achieves (~0.022 absolute, 0.4 % of a
    ~6 peak). Note the old, non-deconvolving 'linear' path held a 10x tighter
    tolerance here -- but only because a no-op kernel reproduces the rendered
    grid bit-for-bit, which measures nothing about interpolation.
    """
    gmm_volume, real_voxel_grid, image_config = _make_gmm_voxel_scene(pdb_info)
    gmm_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    volume = volume_cls.from_real_voxel_grid(real_voxel_grid, interp="cubic")
    proj_ref = _compute_projection(
        gmm_volume.rotate_to_pose(pose), gmm_integrator, image_config
    )
    proj = _compute_projection(volume.rotate_to_pose(pose), integrator, image_config)

    np.testing.assert_allclose(proj_ref, proj, atol=tol)


# ── AbstractVoxelVolume.values / get() ───────────────────────────────────────


@pytest.mark.parametrize("interp", ("linear", "cubic"))
def test_get_returns_values(interp):
    """`get()` returns the voxel array, for both voxel volume flavours."""
    real_voxel_grid = jnp.zeros((16, 16, 16), dtype=float)
    fourier_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        real_voxel_grid, interp=interp
    )
    real_volume = cxs.RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)

    assert fourier_volume.get() is fourier_volume.values
    assert real_volume.get() is real_volume.values
    assert fourier_volume.get().shape == (16, 16, 9)
    assert real_volume.get().shape == (16, 16, 16)


# ── unroll='auto' ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("interp, expected", (("linear", False), ("cubic", True)))
def test_unroll_auto_resolves_from_interp(interp, expected):
    """`unroll='auto'` unrolls the gather only for the cubic kernel's 4^3
    neighborhood. The two strategies must be numerically identical -- only their
    memory/speed profiles differ.
    """
    assert _resolve_unroll("auto", interp) is expected

    real_voxel_grid = np.asarray(
        jnp.exp(-jnp.sum(im.make_coordinate_grid((16, 16, 16)) ** 2, axis=-1) / 4.0)
    )
    config = cxs.BasicImageConfig((16, 16), pixel_size=1.0, voltage_in_kilovolts=300.0)
    pose = cxs.EulerAnglePose(phi_angle=11.0, theta_angle=37.0, psi_angle=21.0)
    volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        real_voxel_grid, interp=interp
    ).rotate_to_pose(pose)

    auto = _project_real(volume, cxs.FourierSliceExtraction(), config)
    for unroll in (True, False):
        explicit = _project_real(
            volume, cxs.FourierSliceExtraction(unroll=unroll), config
        )
        np.testing.assert_allclose(np.array(auto), np.array(explicit), atol=1e-6)
