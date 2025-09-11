import numpy as np
import pytest

import cryojax.experimental as cxe
import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb


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

    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
        loads_b_factors=True,
    )
    atom_potential = cxs.PengIndependentAtomPotential.from_tabulated_parameters(
        atom_positions,
        parameters=cxs.PengScatteringFactorParameters(atom_types),
        extra_b_factors=b_factors,
    )

    instrument_config = cxs.BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage_in_kilovolts,
    )
    # dim = shape[0]
    # voxel_potential = cxs.RealVoxelGridVolume.from_real_voxel_grid(
    #     atom_potential.to_real_voxel_grid((dim, dim, dim), pixel_size),
    # )

    # multislice_integrator = cxe.FFTMultisliceIntegrator(
    #     slice_thickness_in_voxels=3,
    # )
    pose = cxs.EulerAnglePose()
    # pose_inv = pose.to_inverse_rotation()

    ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=defocus_in_angstroms,
        astigmatism_in_angstroms=astigmatism_in_angstroms,
        astigmatism_angle=astigmatism_angle,
    )

    # multislice_scattering_theory = cxe.MultisliceScatteringTheory(
    #     multislice_integrator,
    #     cxe.WaveTransferTheory(ctf),
    #     amplitude_contrast_ratio=ac,
    # )
    high_energy_scattering_theory = cxe.HighEnergyScatteringTheory(
        cxs.GaussianMixtureProjection(use_error_functions=True),
        cxe.WaveTransferTheory(ctf),
        amplitude_contrast_ratio=ac,
    )
    weak_phase_scattering_theory = cxs.WeakPhaseScatteringTheory(
        cxs.GaussianMixtureProjection(use_error_functions=True),
        cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=ac),
    )

    # multislice_image_model_voxel = cxs.IntensityImageModel(
    #     voxel_potential, pose_inv, instrument_config, multislice_scattering_theory
    # )
    high_energy_image_model = cxs.IntensityImageModel(
        atom_potential, pose, instrument_config, high_energy_scattering_theory
    )
    weak_phase_image_model = cxs.IntensityImageModel(
        atom_potential, pose, instrument_config, weak_phase_scattering_theory
    )

    # ms = multislice_image_model_voxel.simulate()
    he = high_energy_image_model.simulate()
    wp = weak_phase_image_model.simulate()

    np.testing.assert_allclose(he, wp, atol=1e-2)

    # normalize_image = lambda image: (image - image.mean()) / image.std()

    # from matplotlib import pyplot as plt

    # vmin, vmax = min(he.min(), wp.min(), ms.min()), max(he.max(), wp.max(), ms.max())
    # fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
    # _ = axes[0].imshow(wp, vmin=vmin, vmax=vmax)
    # _ = axes[1].imshow(he, vmin=vmin, vmax=vmax)
    # im3 = axes[2].imshow(ms, vmin=vmin, vmax=vmax)
    # fig.colorbar(im3)
    # plt.show()

    # np.testing.assert_allclose(normalize_image(he), normalize_image(ms), atol=atol)
    # np.testing.assert_allclose(normalize_image(ms), normalize_image(wp), atol=atol)


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

    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="name CA ",
        loads_b_factors=True,
    )

    atom_potential = cxs.PengIndependentAtomPotential.from_tabulated_parameters(
        atom_positions,
        parameters=cxs.PengScatteringFactorParameters(atom_types),
        extra_b_factors=b_factors,
    )
    instrument_config = cxs.BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage_in_kilovolts,
    )
    # dim = shape[0]

    pose = cxs.EulerAnglePose(*euler_pose_params)
    # pose_inv = pose.to_inverse_rotation()

    # voxel_potential = cxs.RealVoxelGridVolume.from_real_voxel_grid(
    #     atom_potential.to_real_voxel_grid((dim, dim, dim), pixel_size),
    # )
    # multislice_integrator = cxe.FFTMultisliceIntegrator(
    #     slice_thickness_in_voxels=3,
    # )

    ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=defocus_in_angstroms,
        astigmatism_in_angstroms=astigmatism_in_angstroms,
        astigmatism_angle=astigmatism_angle,
    )

    # multislice_scattering_theory = cxe.MultisliceScatteringTheory(
    #     multislice_integrator,
    #     cxe.WaveTransferTheory(ctf),
    #     amplitude_contrast_ratio=ac,
    # )
    high_energy_scattering_theory = cxe.HighEnergyScatteringTheory(
        cxs.GaussianMixtureProjection(use_error_functions=True),
        cxe.WaveTransferTheory(ctf),
        amplitude_contrast_ratio=ac,
    )
    weak_phase_scattering_theory = cxs.WeakPhaseScatteringTheory(
        cxs.GaussianMixtureProjection(use_error_functions=True),
        cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=ac),
    )
    # multislice_image_model_voxel = cxs.IntensityImageModel(
    #     voxel_potential, pose_inv, instrument_config, multislice_scattering_theory
    # )
    high_energy_image_model = cxs.IntensityImageModel(
        atom_potential, pose, instrument_config, high_energy_scattering_theory
    )
    weak_phase_image_model = cxs.IntensityImageModel(
        atom_potential, pose, instrument_config, weak_phase_scattering_theory
    )
    # ms = multislice_image_model_voxel.simulate()
    he = high_energy_image_model.simulate()
    wp = weak_phase_image_model.simulate()

    np.testing.assert_allclose(he, wp, atol=1e-3)

    # normalize_image = lambda image: (image - image.mean()) / image.std()

    # np.testing.assert_allclose(normalize_image(wp), normalize_image(he), atol=atol)
    # np.testing.assert_allclose(normalize_image(he), normalize_image(ms), atol=atol)
    # np.testing.assert_allclose(normalize_image(ms), normalize_image(wp), atol=atol)

    # from matplotlib import pyplot as plt

    # vmin, vmax = min(he.min(), wp.min(), ms.min()), max(he.max(), wp.max(), ms.max())
    # fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
    # _ = axes[0].imshow(wp), vmin=vmin, vmax=vmax)
    # _ = axes[1].imshow(he), vmin=vmin, vmax=vmax)
    # im3 = axes[2].imshow(ms, vmin=vmin, vmax=vmax)
    # fig.colorbar(im3)
    # plt.show()

    # close_fraction = 0.95
    # atol = 1.0
    # assert (
    #     np.isclose(normalize_image(ms), normalize_image(wp), atol=atol)
    #     .astype(float)
    #     .mean()
    #     > close_fraction
    # )
