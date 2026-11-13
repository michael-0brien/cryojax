import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "pixel_size, shape, ctf_params",
    (
        (
            1.0,
            (75, 75),
            (0.1, 300.0, 2500.0, -100.0, 10.0),
        ),
        (
            1.0,
            (75, 75),
            (0.1, 300.0, 2500.0, -100.0, 10.0),
        ),
        (
            1.0,
            (75, 75),
            (0.1, 300.0, 2500.0, -100.0, 10.0),
        ),
    ),
)
def test_scattering_theories_no_pose(
    sample_pdb_path,
    pixel_size,
    shape,
    ctf_params,
):
    (
        ac,
        voltage_in_kilovolts,
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
    ) = ctf_params

    atom_potential = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
        selection_string="not element H",
    )
    instrument_config = cxs.BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage_in_kilovolts,
    )
    pose = cxs.EulerAnglePose()

    ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=defocus_in_angstroms,
        astigmatism_in_angstroms=astigmatism_in_angstroms,
        astigmatism_angle=astigmatism_angle,
    )
    sp = cxs.IntensityImageModel(
        atom_potential,
        pose,
        instrument_config,
        cxs.RytovScatteringTheory(
            cxs.GaussianMixtureProjection(sampling_mode="average"),
            cxs.WaveTransferTheory(ctf),
            amplitude_contrast_ratio=ac,
        ),
    )
    wp = cxs.IntensityImageModel(
        atom_potential,
        pose,
        instrument_config,
        cxs.WeakPhaseScatteringTheory(
            cxs.GaussianMixtureProjection(sampling_mode="average"),
            cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=ac),
        ),
    )
    # TODO: use jax.linearize for exact agreement
    np.testing.assert_allclose(simulate_fn(sp), simulate_fn(wp), atol=1e-2)


@pytest.mark.parametrize(
    "pixel_size, shape, euler_pose_params, ctf_params",
    (
        (
            1.0,
            (75, 75),
            (2.5, -5.0, 0.0, 0.0, 0.0),
            (0.1, 300.0, 2500.0, -100.0, 10.0),
        ),
        (
            1.0,
            (75, 75),
            (0.0, 0.0, 10.0, -30.0, 60.0),
            (0.1, 300.0, 2500.0, -100.0, 10.0),
        ),
        (
            1.0,
            (75, 75),
            (2.5, -5.0, 10.0, -30.0, 60.0),
            (0.1, 300.0, 2500.0, -100.0, 10.0),
        ),
    ),
)
def test_scattering_theories_pose(
    sample_pdb_path,
    pixel_size,
    shape,
    euler_pose_params,
    ctf_params,
):
    (
        ac,
        voltage_in_kilovolts,
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
    ) = ctf_params

    atom_potential = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
        selection_string="name CA ",
    )
    instrument_config = cxs.BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage_in_kilovolts,
    )
    pose = cxs.EulerAnglePose(*euler_pose_params)

    ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=defocus_in_angstroms,
        astigmatism_in_angstroms=astigmatism_in_angstroms,
        astigmatism_angle=astigmatism_angle,
    )

    sp = cxs.IntensityImageModel(
        atom_potential,
        pose,
        instrument_config,
        cxs.RytovScatteringTheory(
            cxs.GaussianMixtureProjection(sampling_mode="average"),
            cxs.WaveTransferTheory(ctf),
            amplitude_contrast_ratio=ac,
        ),
    )
    wp = cxs.IntensityImageModel(
        atom_potential,
        pose,
        instrument_config,
        cxs.WeakPhaseScatteringTheory(
            cxs.GaussianMixtureProjection(sampling_mode="average"),
            cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=ac),
        ),
    )
    # TODO: use jax.linearize for exact agreement
    np.testing.assert_allclose(simulate_fn(sp), simulate_fn(wp), atol=1e-2)


@eqx.filter_jit
def simulate_fn(model):
    return model.simulate()


# ── EwaldSphereExtraction vs FourierSliceExtraction ──────────────────────────

_TWO_ATOM_VOLUME = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
    cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)(
        cxs.GaussianMixtureVolume(
            np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 1.0]]),
            amplitudes=1.0,
            variances=1.0,
        )
    )
)
_ASTIGMATIC_CTF = cxs.AstigmaticCTF(
    defocus_in_angstroms=2500.0,
    astigmatism_in_angstroms=-100.0,
    astigmatism_angle=10.0,
)


def _make_weak_phase_model(integrator, voltage_in_kilovolts):
    instrument_config = cxs.BasicImageConfig(
        (32, 32), pixel_size=1.0, voltage_in_kilovolts=voltage_in_kilovolts
    )
    return cxs.IntensityImageModel(
        _TWO_ATOM_VOLUME,
        cxs.EulerAnglePose(),
        instrument_config,
        cxs.WeakPhaseScatteringTheory(
            integrator,
            cxs.ContrastTransferTheory(_ASTIGMATIC_CTF, amplitude_contrast_ratio=0.1),
        ),
    )


def test_ewald_sphere_matches_fourier_slice_at_high_voltage():
    """At high voltage, the Ewald sphere curvature vanishes, so
    `EwaldSphereExtraction` and `FourierSliceExtraction` must give the
    same image through `ContrastTransferTheory`.
    """
    high_voltage_in_kilovolts = 1e6
    fse_image = simulate_fn(
        _make_weak_phase_model(cxs.FourierSliceExtraction(), high_voltage_in_kilovolts)
    )
    ese_image = simulate_fn(
        _make_weak_phase_model(cxs.EwaldSphereExtraction(), high_voltage_in_kilovolts)
    )
    np.testing.assert_allclose(fse_image, ese_image, atol=1e-6)


def test_ewald_sphere_differs_from_fourier_slice_at_intermediate_voltage():
    """At a typical microscope voltage, Ewald sphere curvature is
    non-negligible, so images from the two extraction methods should not
    agree to the same tolerance as the high voltage limit.
    """
    intermediate_voltage_in_kilovolts = 100.0
    fse_image = simulate_fn(
        _make_weak_phase_model(
            cxs.FourierSliceExtraction(), intermediate_voltage_in_kilovolts
        )
    )
    ese_image = simulate_fn(
        _make_weak_phase_model(
            cxs.EwaldSphereExtraction(), intermediate_voltage_in_kilovolts
        )
    )
    assert not np.allclose(fse_image, ese_image, atol=1e-6)


def test_rytov_scattering_theory_with_ewald_sphere_extraction():
    """`RytovScatteringTheory` should support `EwaldSphereExtraction` as its
    volume integrator, converging to the `FourierSliceExtraction` result
    in the high voltage limit.
    """
    high_voltage_in_kilovolts = 1e6
    instrument_config = cxs.BasicImageConfig(
        (32, 32), pixel_size=1.0, voltage_in_kilovolts=high_voltage_in_kilovolts
    )

    def make_model(integrator):
        return cxs.IntensityImageModel(
            _TWO_ATOM_VOLUME,
            cxs.EulerAnglePose(),
            instrument_config,
            cxs.RytovScatteringTheory(
                integrator,
                cxs.WaveTransferTheory(_ASTIGMATIC_CTF),
                amplitude_contrast_ratio=0.1,
            ),
        )

    fse_image = simulate_fn(make_model(cxs.FourierSliceExtraction()))
    ese_image = simulate_fn(make_model(cxs.EwaldSphereExtraction()))
    np.testing.assert_allclose(fse_image, ese_image, atol=1e-6)


# ── The Ewald branch of `ContrastTransferTheory` ─────────────────────────────

_ODD_ABERRATIONS = pytest.mark.parametrize(
    "coma_xy_in_um, trefoil_xy_in_um",
    [(None, None), ((2.0, -1.0), None), ((2.0, -1.0), (0.5, 0.3))],
    ids=["none", "coma", "coma+trefoil"],
)
_SHAPES = pytest.mark.parametrize("shape", [(32, 32), (31, 31), (32, 31)])


def _aberrated_ctf(coma_xy_in_um, trefoil_xy_in_um):
    return cxs.AberratedCTF(
        defocus_in_um=1.0,
        astigmatism_xy_in_um=(0.03, 0.02),
        coma_xy_in_um=coma_xy_in_um,
        trefoil_xy_in_um=trefoil_xy_in_um,
    )


def _off_nyquist(shape):
    """The half-plane bins off the Nyquist row and column, where the half-plane and
    full-grid paths take opposite signs of the Nyquist frequency."""
    y_dim, x_dim = shape
    mask = np.ones((y_dim, x_dim // 2 + 1), dtype=bool)
    if y_dim % 2 == 0:
        mask[y_dim // 2, :] = False
    if x_dim % 2 == 0:
        mask[:, -1] = False
    return mask


def _relative_error(actual, expected, mask=Ellipsis):
    actual, expected = np.asarray(actual)[mask], np.asarray(expected)[mask]
    return np.abs(actual - expected).max() / np.abs(expected).max()


def _weak_complex_field(shape):
    key_re, key_im = jax.random.split(jax.random.key(0))
    field = 1e-4 * (
        jax.random.normal(key_re, shape) + 1.0j * jax.random.normal(key_im, shape)
    )
    spectrum = jnp.fft.fftn(field)
    y_dim, x_dim = shape
    if y_dim % 2 == 0:
        spectrum = spectrum.at[y_dim // 2, :].set(0.0)
    if x_dim % 2 == 0:
        spectrum = spectrum.at[:, x_dim // 2].set(0.0)
    return jnp.fft.ifftn(spectrum)


def _rytov_weak_limit(
    field, config, ctf, amplitude_contrast_ratio, offset_z_in_angstroms=None
):
    """The contrast spectrum of the exit wave `exp(i w field)`, to first order in the
    field."""
    ac = amplitude_contrast_ratio
    w = jnp.sqrt(1.0 - ac**2) + 1.0j * ac

    def intensity(scale):
        wave = jnp.fft.fftn(jnp.exp(1.0j * scale * w * field))
        wave = cxs.WaveTransferTheory(ctf).propagate_exit_wave(
            wave, config, offset_z_in_angstroms=offset_z_in_angstroms
        )
        return jnp.abs(jnp.fft.ifftn(wave)) ** 2

    _, linear_term = jax.jvp(intensity, (0.0,), (1.0,))
    return jnp.fft.rfftn(linear_term / 2)


def _check_ewald_branch_matches_rytov_weak_limit(
    shape, ctf, amplitude_contrast_ratio, rtol, offset_z_in_angstroms=None
):
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    field = _weak_complex_field(shape)
    ewald = cxs.ContrastTransferTheory(
        ctf, amplitude_contrast_ratio=amplitude_contrast_ratio
    ).propagate_object(
        jnp.fft.fftn(field),
        config,
        is_ewald_sphere=True,
        offset_z_in_angstroms=offset_z_in_angstroms,
    )
    rytov = _rytov_weak_limit(
        field, config, ctf, amplitude_contrast_ratio, offset_z_in_angstroms
    )
    assert _relative_error(ewald, rytov, _off_nyquist(shape)) <= rtol


@_SHAPES
@_ODD_ABERRATIONS
@pytest.mark.parametrize("amplitude_contrast_ratio", [0.0, 0.1])
@pytest.mark.parametrize("offset_z_in_angstroms", [None, 600.0])
def test_ewald_branch_matches_rytov_weak_limit(
    shape,
    coma_xy_in_um,
    trefoil_xy_in_um,
    amplitude_contrast_ratio,
    offset_z_in_angstroms,
):
    _check_ewald_branch_matches_rytov_weak_limit(
        shape,
        _aberrated_ctf(coma_xy_in_um, trefoil_xy_in_um),
        amplitude_contrast_ratio,
        rtol=1e-6,
        offset_z_in_angstroms=offset_z_in_angstroms,
    )


def test_ewald_branch_matches_rytov_weak_limit_in_float32():
    with jax.enable_x64(False):
        _check_ewald_branch_matches_rytov_weak_limit(
            (32, 31), _aberrated_ctf((2.0, -1.0), (0.5, 0.3)), 0.1, rtol=1e-4
        )


@_SHAPES
@_ODD_ABERRATIONS
def test_ewald_real_field_equals_projection(shape, coma_xy_in_um, trefoil_xy_in_um):
    config = cxs.BasicImageConfig(shape, pixel_size=1.0, voltage_in_kilovolts=300.0)
    field = jax.random.normal(jax.random.key(1), shape)
    transfer_theory = cxs.ContrastTransferTheory(
        _aberrated_ctf(coma_xy_in_um, trefoil_xy_in_um), amplitude_contrast_ratio=0.1
    )
    ewald = transfer_theory.propagate_object(
        jnp.fft.fftn(field), config, is_ewald_sphere=True
    )
    projection = transfer_theory.propagate_object(jnp.fft.rfftn(field), config)
    assert _relative_error(ewald, projection) <= 1e-12


@pytest.mark.parametrize("coma_xy_in_um", [None, (3.0, 1.0)], ids=["no-coma", "coma"])
def test_ewald_height_equals_defocus(coma_xy_in_um):
    """An atom at height `d` images as the same atom at 0 with
    `offset_z_in_angstroms = d`, up to interpolation error."""
    d, dim = 6.0, 64
    config = cxs.BasicImageConfig((dim, dim), pixel_size=1.0, voltage_in_kilovolts=20.0)
    transfer_theory = cxs.ContrastTransferTheory(
        cxs.AberratedCTF(defocus_in_um=0.3, coma_xy_in_um=coma_xy_in_um),
        amplitude_contrast_ratio=0.1,
    )

    def image(integrator, z, offset_z_in_angstroms=None):
        volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
            cxs.GaussianMixtureRenderFn((dim, dim, dim), voxel_size=1.0)(
                cxs.GaussianMixtureVolume(
                    np.array([[0.0, 0.0, z]]), amplitudes=1.0, variances=0.5
                )
            ),
            interp="cubic",
        )
        return simulate_fn(
            cxs.IntensityImageModel(
                volume,
                cxs.EulerAnglePose(offset_z_in_angstroms=offset_z_in_angstroms),
                config,
                cxs.WeakPhaseScatteringTheory(integrator, transfer_theory),
            )
        )

    ewald = cxs.EwaldSphereExtraction()
    ewald_change = image(ewald, d) - image(ewald, 0.0)
    projection = cxs.FourierSliceExtraction()
    projection_at_0 = image(projection, 0.0)

    def relative_l2(offset):
        expected = image(projection, 0.0, offset) - projection_at_0
        return np.linalg.norm(ewald_change - expected) / np.linalg.norm(expected)

    assert relative_l2(+d) < 0.05
    assert relative_l2(-d) > 0.5


def test_ewald_rytov_agree_on_extraction():
    volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size=1.0)(
            cxs.GaussianMixtureVolume(
                np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 6.0]]),
                amplitudes=1.0,
                variances=1.0,
            )
        ),
        interp="cubic",
    )
    config = cxs.BasicImageConfig((32, 32), pixel_size=1.0, voltage_in_kilovolts=20.0)
    weak_phase = cxs.WeakPhaseScatteringTheory(
        cxs.EwaldSphereExtraction(),
        cxs.ContrastTransferTheory(_ASTIGMATIC_CTF, amplitude_contrast_ratio=0.1),
    )
    rytov = cxs.RytovScatteringTheory(
        cxs.EwaldSphereExtraction(),
        cxs.WaveTransferTheory(_ASTIGMATIC_CTF),
        amplitude_contrast_ratio=0.1,
    )

    def weak_limit(theory):
        def contrast(scale):
            scaled = eqx.tree_at(lambda v: v.values, volume, scale * volume.values)
            return theory.compute_contrast_spectrum(scaled, config)

        return jax.jvp(contrast, (0.0,), (1.0,))[1]

    error = _relative_error(
        weak_limit(rytov), weak_limit(weak_phase), _off_nyquist((32, 32))
    )
    assert error <= 1e-4
