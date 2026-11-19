"""Tests for the anisotropic magnification, which takes a projection `p(x)` to
`p(D⁻¹x)` for the anisotropy matrix `D`."""

from typing import ClassVar

import cryojax.ndimage as im
import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage._nufft import api as _nufft
from cryojax.simulator._multislice import (
    FFTMultisliceIntegrator,
    MultisliceScatteringTheory,
)
from cryojax.simulator._volume.gaussian_mixture import _resample_with_anisotropy_matrix
from jax.scipy.special import erf


try:
    import jax_finufft
except ModuleNotFoundError:
    jax_finufft = None

_skip_without_jax_finufft = pytest.mark.skipif(
    jax_finufft is None, reason="jax-finufft not installed"
)
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
        _resample_with_anisotropy_matrix(jnp.asarray(image), config.anisotropy_matrix)
    )
    fourier_expected = np.fft.rfftn(expected)
    shells = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
    bounds = [3e-4, 5e-3, 3e-2, 1.5e-1]
    errors = _shell_errors(fourier_image, fourier_expected, shape, shells)
    assert all(e < b for e, b in zip(errors, bounds)), errors


@pytest.mark.parametrize(
    "shape, cropped_shape",
    [((32, 32), None), ((31, 28), None), ((64, 56), (32, 28)), ((63, 57), (21, 19))],
)
def test_resampling_with_zero_anisotropy_is_the_fft(shape, cropped_shape):
    config = cxs.AnisotropicImageConfig(shape, 1.3, anisotropy_xy=(0.0, 0.0))
    image = jnp.asarray(np.random.default_rng(0).normal(size=shape))
    expected = (
        jnp.fft.rfftn(image)
        if cropped_shape is None
        else im.fourier_crop_to_shape(
            image, cropped_shape, outputs_real_space=False, preserve_mean=True
        )
    )
    np.testing.assert_allclose(
        _resample_with_anisotropy_matrix(image, config.anisotropy_matrix, cropped_shape),
        expected,
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
        "fourier gaussians": cxs.GaussianFourierVolume(
            _POSITIONS, im.FourierGaussian(b_factor=8 * np.pi**2 * _SIGMA**2)
        ),
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


def _analytic_projection(
    config, is_translated=True, sampling_mode="average", amplitudes=1.0, sigma=_SIGMA
):
    """The projection of the gaussians `_POSITIONS` (standard deviation `sigma`, of
    `amplitudes`) in `_POSE`, translated if `is_translated`, at the points `D⁻¹x`,
    averaged over each pixel or, for `sampling_mode = 'point'`, sampled."""
    volume = _gaussian_volume(variance=_SIGMA**2).rotate_to_pose(_POSE)
    centers = np.asarray(volume.positions)[:, :2]
    if is_translated:
        centers = centers + np.asarray(_POSE.offset_in_angstroms)
    x = np.asarray(config.get_coordinate_grid(padding=True))[..., None, :] - centers
    if sampling_mode == "point":
        r_squared = np.sum(x**2, axis=-1)
        gaussians = np.exp(-r_squared / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    else:
        d, s = float(config.pixel_size), np.sqrt(2) * sigma
        box = np.asarray(erf((x + d / 2) / s) - erf((x - d / 2) / s))
        gaussians = np.prod(box, axis=-1) / (2 * d) ** 2
    return np.sum(amplitudes * gaussians, axis=-1)


@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
@pytest.mark.parametrize(
    "integrator, volume, outcome, is_direct",
    [
        # Cubic resampling at the image's pixels, 6.8e-5 to 2.1e-4; exact isotropically
        (cxs.GaussianMixtureProjection(), "gaussians", 3e-4, False),
        (cxs.GaussianMixtureProjection(sampling_mode="point"), "gaussians", 3e-4, False),
        # On the grid twice as fine, before the crop: 8.7e-6 to 1.1e-5
        (
            cxs.GaussianMixtureProjection(sampling_mode="point", upsample_factor=2),
            "gaussians",
            2e-5,
            False,
        ),
        (
            cxs.GaussianMixtureProjection(sampling_mode="point", upsample_factor=2),
            "gaussians",
            2e-5,
            True,
        ),
        # The blur divided out at `Dᵀk`: 8.9e-6 to 1.6e-5
        (
            cxs.GaussianMixtureProjection(
                sampling_mode="point", upsample_factor=2, blur_variance=1.0
            ),
            "gaussians",
            3e-5,
            False,
        ),
        # Isotropically 5e-6 to 1e-4 over poses, 7e-6 at `_POSE`
        (cxs.FourierSliceExtraction(), "voxels", 1e-4, False),
        # Exact to its `eps`, 1e-6; 2.2e-8 and 3.0e-8
        (cxs.GaussianFourierProjection(), "fourier gaussians", 1e-6, False),
        (cxs.GaussianFourierProjection(), "fourier gaussians", 1e-6, True),
        # The real part of its field; 7e-6 to 1.3e-5, as `FourierSliceExtraction`
        (cxs.EwaldSphereExtraction(), "voxels", 1e-4, True),
        (cxs.AutoVolumeProjection(), "gaussians", 3e-4, False),
        (cxs.AutoVolumeProjection(), "voxels", 1e-4, False),
        (cxs.AutoVolumeProjection(), "fourier gaussians", 1e-6, False),
    ],
)
def test_integrators_apply_the_anisotropic_magnification(
    volumes, integrator, volume, outcome, is_direct, anisotropy_xy
):
    """Each integrator matches the analytic `p(D⁻¹x)` (Ewald by the real part of its
    field; point-sampled for the band-limited fourier gaussians) to `outcome` of its
    peak, through an image model or when `integrate`, which does not translate, is
    called directly, with its real-space output the `irfftn` of its fourier-space
    output."""
    config = cxs.AnisotropicImageConfig((64, 64), 1.0, anisotropy_xy=anisotropy_xy)
    volume_representation = volumes[volume].rotate_to_pose(_POSE)
    if is_direct:
        image = integrator.integrate(
            volume_representation, config, outputs_real_space=True
        )
    else:
        model = cxs.ProjectionImageModel(volumes[volume], _POSE, config, integrator)
        image = model.simulate()
    expected = _analytic_projection(
        config,
        is_translated=not is_direct,
        sampling_mode=(
            "point"
            if volume == "fourier gaussians"
            else getattr(integrator, "sampling_mode", "average")
        ),
    )
    np.testing.assert_allclose(np.real(image), expected, atol=outcome * expected.max())
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
    "backend",
    ["native", pytest.param("jax-finufft", marks=_skip_without_jax_finufft)],
)
@pytest.mark.parametrize(
    "shape, projection_shape", [((64, 64), None), ((63, 58), None), ((63, 58), (80, 75))]
)
def test_gaussian_fourier_projection_is_exact(
    monkeypatch, backend, shape, projection_shape
):
    """The band-limited `p(D⁻¹x)` to the NUFFT's `eps`, about the center of odd grids
    and of a larger `shape` cropped to the image, with each gaussian of a kernel at its
    own width and amplitude, and the atoms' amplitudes, each applied once."""
    monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", backend)
    amplitudes, sigmas = np.array([0.5, 1.0, 1.5]), (_SIGMA, 4.0)
    kernel_fn = im.FourierGaussian(
        amplitude=np.array([1.5, 0.5]), b_factor=8 * np.pi**2 * np.square(sigmas)
    )
    volume = cxs.GaussianFourierVolume(_POSITIONS, kernel_fn, amplitudes)
    config = cxs.AnisotropicImageConfig(shape, 1.3, anisotropy_xy=(-0.05, 0.03))
    integrator = cxs.GaussianFourierProjection(eps=1e-9, shape=projection_shape)
    expected = sum(
        _analytic_projection(
            config,
            is_translated=False,
            sampling_mode="point",
            amplitudes=a * amplitudes,
            sigma=sigma,
        )
        for a, sigma in zip((1.5, 0.5), sigmas)
    )
    # Measured 4.4e-12 to 6.8e-12 (native) and 2.8e-10 to 3.8e-10 (jax-finufft); one
    # width for both gaussians would give 1.2e-1
    np.testing.assert_allclose(
        integrator.integrate(
            volume.rotate_to_pose(_POSE), config, outputs_real_space=True
        ),
        expected,
        atol=1e-9 * expected.max(),
    )


@pytest.mark.parametrize("anisotropy_xy", [_ANISOTROPY_XY, (-0.05, 0.03)])
@pytest.mark.parametrize(
    "integrator, outcome",
    [
        # 2.1e-6 to 2.2e-6
        (cxs.GaussianMixtureProjection(sampling_mode="point", upsample_factor=3), 5e-6),
        # Cropped from a larger render: 8.7e-6 to 1.1e-5
        (
            cxs.GaussianMixtureProjection(
                sampling_mode="point", upsample_factor=2, shape=(70, 66)
            ),
            2e-5,
        ),
        # Padded from a smaller render, periodic at its edges, which hold 3e-5 of the
        # peak: 4.6e-5 to 1.6e-4
        (
            cxs.GaussianMixtureProjection(
                sampling_mode="point", upsample_factor=2, shape=(48, 45)
            ),
            3e-4,
        ),
    ],
)
def test_point_sampling_resamples_odd_grids_about_their_center(
    volumes, integrator, outcome, anisotropy_xy
):
    """With `sampling_mode = 'point'`, the finer grid is resampled about its center
    pixel, which the crop keeps on the image's, here at odd sizes."""
    config = cxs.AnisotropicImageConfig((63, 57), 1.0, anisotropy_xy=anisotropy_xy)
    image = integrator.integrate(
        volumes["gaussians"].rotate_to_pose(_POSE), config, outputs_real_space=True
    )
    expected = _analytic_projection(config, is_translated=False, sampling_mode="point")
    np.testing.assert_allclose(image, expected, atol=outcome * expected.max())


@pytest.mark.parametrize(
    "integrator, volume, name",
    [
        (integrator, volume, name)
        for integrator, volume in [
            (cxs.GaussianMixtureProjection(sampling_mode="point"), None),
            (
                cxs.GaussianMixtureProjection(
                    sampling_mode="point", upsample_factor=2, blur_variance=1.0
                ),
                None,
            ),
            (cxs.FourierSliceExtraction(), "voxels"),
            (cxs.GaussianFourierProjection(), "fourier gaussians"),
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


_ONE_POSITION = np.array([[5.0, -3.0, 4.0]])
_ONE_POSE = cxs.EulerAnglePose(4.0, -6.0, 30.0, 50.0, -20.0)
_ONE_SIGMA = 4.0


@pytest.fixture(scope="module")
def one_gaussian():
    """One wide gaussian off the volume's center, point-sampled in every
    representation."""
    z, y, x = np.indices((64, 64, 64)) - 32
    r = np.stack([x, y, z], axis=-1) - _ONE_POSITION[0]
    density = np.exp(-np.sum(r**2, axis=-1) / (2 * _ONE_SIGMA**2))
    return {
        "gaussians": cxs.GaussianMixtureVolume(
            _ONE_POSITION, amplitudes=1.0, variances=_ONE_SIGMA**2
        ),
        "voxels": cxs.FourierVoxelGridVolume.from_real_voxel_grid(
            density / (2 * np.pi * _ONE_SIGMA**2) ** 1.5, interp="cubic", pad_scale=2.0
        ),
        "fourier gaussians": cxs.GaussianFourierVolume(
            _ONE_POSITION, im.FourierGaussian(b_factor=8 * np.pi**2 * _ONE_SIGMA**2)
        ),
    }


@pytest.mark.parametrize("anisotropy_xy", [(0.12, 0.08), (-0.05, 0.03)])
@pytest.mark.parametrize(
    "integrator, volume, outcome",
    [
        # 3.4e-6 to 3.6e-6
        (
            cxs.GaussianMixtureProjection(sampling_mode="point", upsample_factor=2),
            "gaussians",
            1e-5,
        ),
        # 1.1e-5, as isotropically
        (cxs.FourierSliceExtraction(), "voxels", 3e-5),
        # 1.5e-8 to 2.1e-8
        (cxs.GaussianFourierProjection(), "fourier gaussians", 1e-7),
    ],
)
def test_integrators_match_one_distorted_gaussian(
    one_gaussian, integrator, volume, outcome, anisotropy_xy
):
    """The gaussian in a pose with an offset, under a large and a typical anisotropy,
    against the point-sampled `p(D⁻¹x)` written out here from
    `D = I + [[-e0, e1], [e1, e0]]`, periodic like the simulations."""
    e0, e1 = anisotropy_xy
    D = np.eye(2) + np.array([[-e0, e1], [e1, e0]])
    config = cxs.AnisotropicImageConfig((64, 64), 1.0, anisotropy_xy=anisotropy_xy)
    center = np.asarray(one_gaussian["gaussians"].rotate_to_pose(_ONE_POSE).positions)[
        :, :2
    ] + np.asarray(_ONE_POSE.offset_in_angstroms)
    shifts = 64 * np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    expected = _gaussians(
        (64, 64), center + shifts @ np.linalg.inv(D).T, _ONE_SIGMA, D
    ) / (2 * np.pi * _ONE_SIGMA**2)
    image = cxs.ProjectionImageModel(
        one_gaussian[volume], _ONE_POSE, config, integrator
    ).simulate()
    np.testing.assert_allclose(image, expected, atol=outcome * expected.max())


@pytest.mark.parametrize("anisotropy_xy", [(0.12, 0.08), (-0.05, 0.03)])
def test_integrators_agree_through_a_ctf(one_gaussian, anisotropy_xy):
    """Through a CTF, each integrator matches `GaussianFourierProjection`, and
    `EwaldSphereExtraction` at zero curvature matches `FourierSliceExtraction`."""
    transfer_theory = cxs.ContrastTransferTheory(
        cxs.AstigmaticCTF(
            defocus_in_angstroms=8000.0,
            astigmatism_in_angstroms=600.0,
            astigmatism_angle=20.0,
        ),
        amplitude_contrast_ratio=0.1,
    )

    def simulate(integrator, volume, voltage_in_kilovolts=300.0):
        config = cxs.AnisotropicImageConfig(
            (64, 64),
            1.0,
            voltage_in_kilovolts=voltage_in_kilovolts,
            anisotropy_xy=anisotropy_xy,
        )
        return cxs.LinearImageModel(
            one_gaussian[volume], _ONE_POSE, config, transfer_theory, integrator
        ).simulate()

    expected = simulate(cxs.GaussianFourierProjection(), "fourier gaussians")
    # 6.9e-7 to 9.4e-7, and 9.4e-6 to 9.5e-6
    for integrator, volume, outcome in [
        (
            cxs.GaussianMixtureProjection(sampling_mode="point", upsample_factor=2),
            "gaussians",
            3e-6,
        ),
        (cxs.FourierSliceExtraction(), "voxels", 3e-5),
    ]:
        np.testing.assert_allclose(
            simulate(integrator, volume),
            expected,
            atol=outcome * np.abs(expected).max(),
        )
    # At 1e9 kV the curvature `λ|q|²/2` is below 2e-9: 2.3e-9
    expected = simulate(cxs.FourierSliceExtraction(), "voxels", 1e9)
    np.testing.assert_allclose(
        simulate(cxs.EwaldSphereExtraction(), "voxels", 1e9),
        expected,
        atol=1e-8 * np.abs(expected).max(),
    )


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
    "integrator, volume, backend",
    [
        (cxs.GaussianMixtureProjection(sampling_mode="point"), None, None),
        (
            cxs.GaussianMixtureProjection(
                sampling_mode="point", upsample_factor=2, blur_variance=0.5
            ),
            None,
            None,
        ),
        (cxs.FourierSliceExtraction(), "voxels", None),
        (cxs.EwaldSphereExtraction(), "voxels", None),
        (cxs.GaussianFourierProjection(eps=1e-9), "fourier gaussians", "native"),
        pytest.param(
            cxs.GaussianFourierProjection(eps=1e-9),
            "fourier gaussians",
            "jax-finufft",
            marks=_skip_without_jax_finufft,
        ),
    ],
)
def test_anisotropy_gradients_match_finite_differences(
    monkeypatch, volumes, integrator, volume, backend, anisotropy_xy
):
    if backend is not None:
        monkeypatch.setattr(_nufft, "CRYOJAX_FINUFFT_BACKEND", backend)
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
    "sampling_mode, name, extra_ffts",
    [
        ("average", "projection", 0),
        ("average", "weak phase", 0),
        ("average", "rytov", 2),
        ("point", "projection", 0),
        ("point", "weak phase", 0),
        ("point", "rytov", 0),
    ],
)
def test_resampling_adds_no_ffts_to_a_real_space_projection(
    sampling_mode, name, extra_ffts
):
    """With `GaussianMixtureProjection`, which projects in real space, the resampling
    reuses the FFT that the isotropic simulation computes: of the image with
    `sampling_mode = 'average'`, of the finer grid before the crop with `'point'`.
    Rytov takes the projection in real space, so there `'average'` adds a pair."""
    shape = (32, 32)
    integrator = cxs.GaussianMixtureProjection(
        sampling_mode=sampling_mode, upsample_factor=2
    )
    isotropic = _models(cxs.BasicImageConfig(shape, 1.0), integrator=integrator)[name]
    anisotropic = _models(
        cxs.AnisotropicImageConfig(shape, 1.0, anisotropy_xy=(0.02, 0.01)),
        integrator=integrator,
    )[name]
    assert (
        _count_ffts(anisotropic.simulate) == _count_ffts(isotropic.simulate) + extra_ffts
    )
