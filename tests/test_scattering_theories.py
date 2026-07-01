import cryojax.simulator as cxs
import equinox as eqx
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
