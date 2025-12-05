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
        cxs.StrongPhaseScatteringTheory(
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
        cxs.StrongPhaseScatteringTheory(
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
