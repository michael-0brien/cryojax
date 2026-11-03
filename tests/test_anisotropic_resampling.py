"""Tests for resampling projections with the anisotropy matrix `D`, as `p(D⁻¹x)`."""

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.simulator._resample import resample_with_anisotropy_matrix


_ANISOTROPY_XY = (0.02 * np.cos(np.deg2rad(60.0)), 0.02 * np.sin(np.deg2rad(60.0)))


def _gaussians(shape, centers, sigma, matrix=np.eye(2)):
    """Point samples of `sum_j N(x; D r_j, σ² D Dᵀ)` (unnormalized) on the pixel grid,
    with `x` about the center `N//2`."""
    y, x = np.indices(shape)
    grid = np.stack([x - shape[1] // 2, y - shape[0] // 2], axis=-1).astype(float)
    precision = np.linalg.inv(sigma**2 * matrix @ matrix.T)
    image = np.zeros(shape)
    for center in centers @ matrix.T:
        r = grid - center
        image += np.exp(-0.5 * np.einsum("...i,ij,...j->...", r, precision, r))
    return image


def _shell_errors(actual, expected, shape, shells):
    """The error relative to `expected` in each shell of frequency magnitude, in
    cycles per pixel, for a full or half (rfft) plane at real-space `shape`."""
    is_rfft = expected.shape[1] != shape[1]
    kx = np.fft.rfftfreq(shape[1]) if is_rfft else np.fft.fftfreq(shape[1])
    ky = np.fft.fftfreq(shape[0])
    radius = np.hypot(ky[:, None], kx[None, :])
    return [
        np.linalg.norm((actual - expected)[(radius >= lo) & (radius < hi)])
        / np.linalg.norm(expected[(radius >= lo) & (radius < hi)])
        for lo, hi in shells
    ]


@pytest.mark.parametrize("shape", [(96, 96), (95, 88)])
@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
@pytest.mark.parametrize("is_complex", [False, True])
def test_resampling_matches_a_direct_anisotropic_gaussian_render(
    shape, anisotropy_xy, is_complex
):
    """Resampling gaussians is compared with rendering them directly with centers
    `D r` and covariance `σ² D Dᵀ`, per frequency shell. The bounds are those of cubic
    interpolation with its response divided out, for `σ = 1` pixel; the last shell
    before Nyquist is dominated by aliasing of the point samples themselves."""
    config = cxs.AnisotropicImageConfig(shape, 1.0, anisotropy_xy=anisotropy_xy)
    D = np.asarray(config.anisotropy_matrix)
    rng = np.random.default_rng(0)
    centers, sigma = rng.uniform(-20, 20, (40, 2)), 1.0
    image, expected = (
        _gaussians(shape, centers, sigma),
        _gaussians(shape, centers, sigma, D),
    )
    phase = 1.0 + 0.5j if is_complex else 1.0
    fourier_image = np.asarray(
        resample_with_anisotropy_matrix(
            jnp.asarray(phase * image), config, outputs_rfft=not is_complex
        )
    )
    fourier_expected = (
        np.fft.fftn(phase * expected) if is_complex else np.fft.rfftn(expected)
    )
    shells = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
    bounds = [3e-4, 5e-3, 3e-2, 1.5e-1]
    errors = _shell_errors(fourier_image, fourier_expected, shape, shells)
    assert all(e < b for e, b in zip(errors, bounds)), errors


@pytest.mark.parametrize("shape", [(32, 32), (31, 28)])
@pytest.mark.parametrize("is_complex", [False, True])
def test_resampling_with_zero_anisotropy_is_the_fft(shape, is_complex):
    config = cxs.AnisotropicImageConfig(shape, 1.3, anisotropy_xy=(0.0, 0.0))
    rng = np.random.default_rng(0)
    image = rng.normal(size=shape) + (1j * rng.normal(size=shape) if is_complex else 0)
    expected = np.fft.fftn(image) if is_complex else np.fft.rfftn(image)
    np.testing.assert_allclose(
        resample_with_anisotropy_matrix(
            jnp.asarray(image), config, outputs_rfft=not is_complex
        ),
        expected,
        atol=1e-10,
    )


def _gaussian_volume(amplitude=1.0):
    positions = np.array([[3.5, -2.2, 0.0], [-6.0, 4.0, 1.0], [8.0, 7.5, -2.0]])
    return cxs.GaussianMixtureVolume(positions, amplitudes=amplitude, variances=2.0)


def _models(config):
    integrator = cxs.GaussianMixtureProjection(sampling_mode="point")
    pose = cxs.EulerAnglePose(1.5, -2.0, 10.0, 20.0, 30.0)
    ctf = cxs.AstigmaticCTF(defocus_in_angstroms=5000.0, astigmatism_in_angstroms=300.0)
    return {
        "projection": cxs.ProjectionImageModel(
            _gaussian_volume(), pose, config, integrator
        ),
        "weak phase": cxs.LinearImageModel(
            _gaussian_volume(),
            pose,
            config,
            cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.1),
            integrator,
        ),
        "rytov": cxs.IntensityImageModel(
            _gaussian_volume(),
            pose,
            config,
            cxs.RytovScatteringTheory(integrator, cxs.WaveTransferTheory(ctf)),
        ),
    }


@pytest.mark.parametrize("name", ["projection", "weak phase", "rytov"])
def test_zero_anisotropy_reproduces_the_isotropic_simulation(name):
    shape = (48, 40)
    isotropic = _models(cxs.BasicImageConfig(shape, 1.1))[name].simulate()
    anisotropic = _models(cxs.AnisotropicImageConfig(shape, 1.1))[name].simulate()
    np.testing.assert_allclose(
        anisotropic, isotropic, atol=1e-10 * np.abs(isotropic).max()
    )


def test_rytov_in_vacuum_is_the_plane_wave():
    config = cxs.AnisotropicImageConfig((32, 32), 1.0, anisotropy_xy=(0.05, -0.03))
    model = cxs.IntensityImageModel(
        _gaussian_volume(amplitude=0.0),
        cxs.EulerAnglePose(),
        config,
        cxs.RytovScatteringTheory(
            cxs.GaussianMixtureProjection(), cxs.WaveTransferTheory(cxs.AstigmaticCTF())
        ),
    )
    np.testing.assert_allclose(model.simulate(), 1.0, atol=1e-12)


def _loss(anisotropy_xy):
    config = cxs.AnisotropicImageConfig((32, 32), 1.0, anisotropy_xy=anisotropy_xy)
    image = _models(config)["weak phase"].simulate()
    weights = jnp.asarray(np.random.default_rng(0).normal(size=(32, 32)))
    return jnp.sum(weights * image)


@pytest.mark.parametrize("anisotropy_xy", [(0.0, 0.0), (0.03, -0.02)])
def test_anisotropy_gradients_match_finite_differences(anisotropy_xy):
    anisotropy_xy = jnp.asarray(anisotropy_xy)
    gradient = jax.grad(_loss)(anisotropy_xy)
    h = 1e-6
    finite_differences = [
        (_loss(anisotropy_xy.at[i].add(h)) - _loss(anisotropy_xy.at[i].add(-h))) / (2 * h)
        for i in range(2)
    ]
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient, finite_differences, rtol=1e-6, atol=1e-9)


def test_anisotropy_vmaps_over_a_batch():
    batch = jnp.asarray([[0.0, 0.0], [0.02, 0.01], [-0.04, 0.03]])
    np.testing.assert_allclose(
        jax.vmap(jax.jit(_loss))(batch),
        jnp.stack([_loss(a) for a in batch]),
        rtol=1e-10,
    )


def _count_ffts(fn):
    def count(jaxpr):
        n = 0
        for eqn in jaxpr.eqns:
            n += eqn.primitive.name == "fft"
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else [param]:
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        n += count(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        n += count(sub)
        return n

    return count(jax.make_jaxpr(fn)().jaxpr)


@pytest.mark.parametrize("name", ["projection", "weak phase", "rytov"])
def test_resampling_adds_no_ffts_to_a_real_space_projection(name):
    """With `GaussianMixtureProjection`, which projects in real space, the resampling
    reuses the FFT that the isotropic simulation computes."""
    shape = (32, 32)
    isotropic = _models(cxs.BasicImageConfig(shape, 1.0))[name]
    anisotropic = _models(
        cxs.AnisotropicImageConfig(shape, 1.0, anisotropy_xy=(0.02, 0.01))
    )[name]
    assert _count_ffts(anisotropic.simulate) == _count_ffts(isotropic.simulate)
