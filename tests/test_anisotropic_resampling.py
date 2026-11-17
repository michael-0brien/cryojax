"""Tests for the anisotropic magnification, which takes a projection `p(x)` to
`p(D⁻¹x)` for the anisotropy matrix `D`."""

from typing import ClassVar

import cryojax.ndimage as im
import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.simulator._multislice import (
    FFTMultisliceIntegrator,
    MultisliceScatteringTheory,
)
from cryojax.simulator._volume.gaussian_mixture import _resample_with_anisotropy_matrix
from jax.scipy.special import erf


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
    cycles per pixel, for a half (rfft) plane at real-space `shape`."""
    kx = np.fft.rfftfreq(shape[1])
    ky = np.fft.fftfreq(shape[0])
    radius = np.hypot(ky[:, None], kx[None, :])
    return [
        np.linalg.norm((actual - expected)[(radius >= lo) & (radius < hi)])
        / np.linalg.norm(expected[(radius >= lo) & (radius < hi)])
        for lo, hi in shells
    ]


@pytest.mark.parametrize("shape", [(96, 96), (95, 88)])
@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
def test_resampling_matches_a_direct_anisotropic_gaussian_render(shape, anisotropy_xy):
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
    fourier_image = np.asarray(
        _resample_with_anisotropy_matrix(jnp.asarray(image), config)
    )
    fourier_expected = np.fft.rfftn(expected)
    shells = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
    bounds = [3e-4, 5e-3, 3e-2, 1.5e-1]
    errors = _shell_errors(fourier_image, fourier_expected, shape, shells)
    assert all(e < b for e, b in zip(errors, bounds)), errors


@pytest.mark.parametrize("shape", [(32, 32), (31, 28)])
def test_resampling_with_zero_anisotropy_is_the_fft(shape):
    config = cxs.AnisotropicImageConfig(shape, 1.3, anisotropy_xy=(0.0, 0.0))
    image = np.random.default_rng(0).normal(size=shape)
    np.testing.assert_allclose(
        _resample_with_anisotropy_matrix(jnp.asarray(image), config),
        np.fft.rfftn(image),
        atol=1e-10,
    )


_POSITIONS = np.array([[3.5, -2.2, 0.0], [-6.0, 4.0, 1.0], [8.0, 7.5, -2.0]])
_POSE = cxs.EulerAnglePose(1.5, -2.0, 10.0, 20.0, 30.0)
_SIGMA = 3.0


def _gaussian_volume(amplitude=1.0, variance=2.0):
    return cxs.GaussianMixtureVolume(_POSITIONS, amplitudes=amplitude, variances=variance)


@pytest.fixture(scope="module")
def volumes():
    gaussians = _gaussian_volume(variance=_SIGMA**2)
    return {
        "gaussians": gaussians,
        "voxels": cxs.FourierVoxelGridVolume.from_real_voxel_grid(
            cxs.GaussianMixtureRenderFn((64, 64, 64), voxel_size=1.0)(gaussians),
            interp="cubic",
            pad_scale=2.0,
        ),
        "fourier gaussians": cxs.GaussianFourierVolume(_POSITIONS, im.FourierGaussian()),
    }


def _models(
    config,
    volume=None,
    # A projection made in real space, so that resampling adds no FFTs
    integrator=cxs.GaussianMixtureProjection(sampling_mode="average"),
):
    volume = _gaussian_volume() if volume is None else volume
    ctf = cxs.AstigmaticCTF(defocus_in_angstroms=5000.0, astigmatism_in_angstroms=300.0)
    transfer_theory = cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.1)
    models = {
        "weak phase": cxs.LinearImageModel(
            volume, _POSE, config, transfer_theory, integrator
        ),
        "weak phase theory": cxs.ContrastImageModel(
            volume,
            _POSE,
            config,
            cxs.WeakPhaseScatteringTheory(integrator, transfer_theory),
        ),
        "rytov": cxs.IntensityImageModel(
            volume,
            _POSE,
            config,
            cxs.RytovScatteringTheory(integrator, cxs.WaveTransferTheory(ctf)),
        ),
    }
    # `ProjectionImageModel` refuses Ewald integrators
    if not integrator.outputs_ewald_sphere:
        models["projection"] = cxs.ProjectionImageModel(volume, _POSE, config, integrator)
    return models


def _analytic_projection(config, is_translated=True):
    """The pixel-averaged projection of the gaussians `_POSITIONS` (unit amplitude,
    standard deviation `_SIGMA`) in `_POSE`, translated if `is_translated`, at the
    points `D⁻¹x`."""
    volume = _gaussian_volume(variance=_SIGMA**2).rotate_to_pose(_POSE)
    centers = np.asarray(volume.positions)[:, :2]
    if is_translated:
        centers = centers + np.asarray(_POSE.offset_in_angstroms)
    x = np.asarray(config.get_coordinate_grid(padding=True))[..., None, :] - centers
    d, s = float(config.pixel_size), np.sqrt(2) * _SIGMA
    box = np.asarray(erf((x + d / 2) / s) - erf((x - d / 2) / s))
    return np.sum(np.prod(box, axis=-1), axis=-1) / (2 * d) ** 2


@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
@pytest.mark.parametrize(
    "integrator, volume, outcome, is_direct",
    [
        # Cubic resampling; exact isotropically
        (cxs.GaussianMixtureProjection(), "gaussians", 3e-4, False),
        # Isotropically 5e-6 to 1e-4 over poses, 7e-6 at `_POSE`
        (cxs.FourierSliceExtraction(), "voxels", 1e-4, False),
        (
            cxs.GaussianFourierProjection(),
            "fourier gaussians",
            "GaussianFourierProjection",
            False,
        ),
        (
            cxs.GaussianFourierProjection(),
            "fourier gaussians",
            "GaussianFourierProjection",
            True,
        ),
        # The real part of its field; 7e-6 to 1.3e-5, as `FourierSliceExtraction`
        (cxs.EwaldSphereExtraction(), "voxels", 1e-4, True),
        (cxs.AutoVolumeProjection(), "gaussians", 3e-4, False),
        (cxs.AutoVolumeProjection(), "voxels", 1e-4, False),
        (
            cxs.AutoVolumeProjection(),
            "fourier gaussians",
            "GaussianFourierProjection",
            False,
        ),
    ],
)
def test_integrators_apply_the_anisotropic_magnification(
    volumes, integrator, volume, outcome, is_direct, anisotropy_xy
):
    """Each integrator matches the analytic `p(D⁻¹x)` (Ewald by the real part of its
    field) to `outcome` of its peak, with its real-space output the `irfftn` of its
    fourier-space output, or raises for the integrator named by `outcome`, through an
    image model or when `integrate`, which does not translate, is called directly."""
    config = cxs.AnisotropicImageConfig((64, 64), 1.0, anisotropy_xy=anisotropy_xy)
    volume_representation = volumes[volume].rotate_to_pose(_POSE)
    if is_direct:
        simulate = lambda: integrator.integrate(
            volume_representation, config, outputs_real_space=True
        )
    else:
        model = cxs.ProjectionImageModel(volumes[volume], _POSE, config, integrator)
        simulate = model.simulate
    if isinstance(outcome, str):
        with pytest.raises(NotImplementedError, match=f"`{outcome}` does not support"):
            simulate()
    else:
        expected = _analytic_projection(config, is_translated=not is_direct)
        np.testing.assert_allclose(
            np.real(simulate()), expected, atol=outcome * expected.max()
        )
        parts = np.fft.irfftn(
            integrator.integrate(volume_representation, config),
            s=config.padded_shape,
            axes=(-2, -1),
        )
        if integrator.outputs_ewald_sphere:
            parts = parts[0] + 1j * parts[1]
        np.testing.assert_allclose(
            integrator.integrate(volume_representation, config, outputs_real_space=True),
            parts,
            atol=1e-12 * expected.max(),
        )


@pytest.mark.parametrize(
    "integrator, volume, name",
    [
        (integrator, volume, name)
        for integrator, volume in [
            (cxs.GaussianMixtureProjection(sampling_mode="average"), None),
            (cxs.FourierSliceExtraction(), "voxels"),
        ]
        for name in ["projection", "weak phase", "weak phase theory", "rytov"]
    ]
    + [
        (cxs.EwaldSphereExtraction(), "voxels", name)
        for name in ["weak phase", "weak phase theory", "rytov"]
    ],
)
def test_zero_anisotropy_reproduces_the_isotropic_simulation(
    volumes, integrator, volume, name
):
    shape = (48, 40)
    volume = None if volume is None else volumes[volume]
    simulate = lambda config: _models(config, volume, integrator)[name].simulate()
    isotropic = simulate(cxs.BasicImageConfig(shape, 1.1))
    anisotropic = simulate(cxs.AnisotropicImageConfig(shape, 1.1))
    np.testing.assert_allclose(
        anisotropic, isotropic, atol=1e-10 * np.abs(isotropic).max()
    )


@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
def test_ewald_sphere_is_sampled_at_the_distorted_frequencies(volumes, anisotropy_xy):
    """The sphere `q - λ|q|²/2 ẑ` and its mirror at `q = Dᵀk`, times `|det D|`, here
    curved before the pose rotates it, at a voltage where the curvature matters."""
    volume = volumes["voxels"]
    config = cxs.AnisotropicImageConfig(
        (128, 128), 1.0, voltage_in_kilovolts=60.0, anisotropy_xy=anisotropy_xy
    )
    D = config.anisotropy_matrix
    k = im.make_frequency_slice((128, 128), fftshifted=True)
    q = k.at[..., :2].set(k[..., :2] @ D)
    curvature = config.wavelength_in_angstroms * jnp.sum(q**2, axis=-1) / 2
    on_sphere, on_mirror = (
        im.sample_fft_slice(
            volume.values,
            _POSE.rotate_coordinates(q.at[..., 2].add(sign * curvature), inverse=True),
            interp="cubic",
        )
        for sign in (-1.0, 1.0)
    )
    expected = jnp.stack([on_sphere + on_mirror, (on_sphere - on_mirror) / 1j]) / 2
    expected *= jnp.abs(jnp.linalg.det(D))
    np.testing.assert_allclose(
        cxs.EwaldSphereExtraction().integrate(volume.rotate_to_pose(_POSE), config),
        expected,
        atol=1e-12 * np.abs(expected).max(),
    )


@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
def test_anisotropic_ewald_sphere_at_high_voltage_is_the_fourier_slice(
    volumes, anisotropy_xy
):
    """The field's real part is `FourierSliceExtraction`'s projection and its imaginary
    part vanishes, up to second and first order in the wavelength (the imaginary part
    is 1.4e-3 of the peak at 300 kV)."""
    config = cxs.AnisotropicImageConfig(
        (64, 64), 1.0, voltage_in_kilovolts=1e6, anisotropy_xy=anisotropy_xy
    )
    volume = volumes["voxels"].rotate_to_pose(_POSE)
    field = cxs.EwaldSphereExtraction().integrate(volume, config, outputs_real_space=True)
    projection = cxs.FourierSliceExtraction().integrate(
        volume, config, outputs_real_space=True
    )
    peak = np.abs(projection).max()
    # Measured 1.2e-12 and 8.9e-7
    np.testing.assert_allclose(field.real, projection, atol=1e-11 * peak)
    np.testing.assert_allclose(field.imag, 0.0, atol=2e-6 * peak)


class _NoAnisotropyIntegrator(cxs.AbstractVolumeIntegrator, strict=True):
    outputs_ewald_sphere: ClassVar[bool] = False
    supports_anisotropy: ClassVar[bool] = False

    def integrate(self, volume_representation, image_config, outputs_real_space=False):
        y_dim, x_dim = image_config.padded_shape
        if outputs_real_space:
            return jnp.zeros((y_dim, x_dim))
        return jnp.zeros((y_dim, x_dim // 2 + 1), dtype=complex)


@pytest.mark.parametrize(
    "name", ["projection", "weak phase", "weak phase theory", "rytov"]
)
def test_an_integrator_without_anisotropy_raises_through_the_image_models(name):
    shape, integrator = (32, 32), _NoAnisotropyIntegrator()
    _models(cxs.BasicImageConfig(shape, 1.0), integrator=integrator)[name].simulate()
    config = cxs.AnisotropicImageConfig(shape, 1.0, anisotropy_xy=_ANISOTROPY_XY)
    model = _models(config, integrator=integrator)[name]
    with pytest.raises(
        NotImplementedError, match="`_NoAnisotropyIntegrator` does not support"
    ):
        model.simulate()


def test_an_integrator_must_declare_whether_it_supports_anisotropy():
    class Undeclared(cxs.AbstractVolumeIntegrator, strict=True):
        outputs_ewald_sphere: ClassVar[bool] = False

        def integrate(
            self, volume_representation, image_config, outputs_real_space=False
        ):
            raise NotImplementedError

    with pytest.raises(TypeError, match="supports_anisotropy"):
        Undeclared()


def test_multislice_raises():
    config = cxs.AnisotropicImageConfig((8, 8), 1.0, anisotropy_xy=_ANISOTROPY_XY)
    model = cxs.IntensityImageModel(
        cxs.RealVoxelGridVolume.from_real_voxel_grid(np.zeros((8, 8, 8))),
        cxs.EulerAnglePose(),
        config,
        MultisliceScatteringTheory(
            FFTMultisliceIntegrator(), cxs.WaveTransferTheory(cxs.AstigmaticCTF())
        ),
    )
    with pytest.raises(
        NotImplementedError, match="`MultisliceScatteringTheory` does not support"
    ):
        model.simulate()


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


def _loss(
    anisotropy_xy,
    volume=None,
    integrator=cxs.GaussianMixtureProjection(sampling_mode="point"),
):
    config = cxs.AnisotropicImageConfig((32, 32), 1.0, anisotropy_xy=anisotropy_xy)
    image = _models(config, volume, integrator)["weak phase"].simulate()
    weights = jnp.asarray(np.random.default_rng(0).normal(size=(32, 32)))
    return jnp.sum(weights * image)


@pytest.mark.parametrize("anisotropy_xy", [(0.0, 0.0), (0.03, -0.02)])
@pytest.mark.parametrize(
    "integrator, volume",
    [
        (cxs.GaussianMixtureProjection(sampling_mode="point"), None),
        (cxs.FourierSliceExtraction(), "voxels"),
        (cxs.EwaldSphereExtraction(), "voxels"),
    ],
)
def test_anisotropy_gradients_match_finite_differences(
    volumes, integrator, volume, anisotropy_xy
):
    loss = lambda a: _loss(a, None if volume is None else volumes[volume], integrator)
    anisotropy_xy = jnp.asarray(anisotropy_xy)
    gradient = jax.grad(loss)(anisotropy_xy)
    h = 1e-6
    finite_differences = [
        (loss(anisotropy_xy.at[i].add(h)) - loss(anisotropy_xy.at[i].add(-h))) / (2 * h)
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


@pytest.mark.parametrize(
    "name, extra_ffts", [("projection", 0), ("weak phase", 0), ("rytov", 2)]
)
def test_resampling_adds_no_ffts_to_a_real_space_projection(name, extra_ffts):
    """With `GaussianMixtureProjection`, which projects in real space, the resampling
    reuses the FFT that the isotropic simulation computes. Rytov takes the projection
    in real space, so there the resampling adds a pair."""
    shape = (32, 32)
    isotropic = _models(cxs.BasicImageConfig(shape, 1.0))[name]
    anisotropic = _models(
        cxs.AnisotropicImageConfig(shape, 1.0, anisotropy_xy=(0.02, 0.01))
    )[name]
    assert (
        _count_ffts(anisotropic.simulate) == _count_ffts(isotropic.simulate) + extra_ffts
    )
