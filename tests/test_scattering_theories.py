import pytest

import numpy as np

import cryojax.experimental as cxe
import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb
from time import time

import logging

# logging with time stamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# @pytest.mark.parametrize(
#     "pixel_size, shape, ctf_params",
#     (
#         (
#             2.0,
#             (150, 150),
#             (0.1, 300.0, 10000.0, -100.0, 10.0),
#         ),
#         (
#             2.0,
#             (150, 150),
#             (0.1, 300.0, 10000.0, -100.0, 10.0),
#         ),
#         (
#             2.0,
#             (150, 150),
#             (0.1, 300.0, 10000.0, -100.0, 10.0),
#         ),
#     ),
# )
# def test_scattering_theories_no_pose(
#     sample_pdb_path,
#     pixel_size,
#     shape,
#     ctf_params,
# ):
#     (
#         ac,
#         voltage_in_kilovolts,
#         defocus_in_angstroms,
#         astigmatism_in_angstroms,
#         astigmatism_angle,
#     ) = ctf_params

#     atom_positions, atom_types, b_factors = read_atoms_from_pdb(
#         sample_pdb_path,
#         center=True,
#         selection_string="not element H",
#         loads_b_factors=True,
#     )
#     atom_potential = cxs.PengIndependentAtomPotential.from_tabulated_parameters(
#         atom_positions,
#         parameters=cxs.PengScatteringFactorParameters(atom_types),
#         extra_b_factors=b_factors,
#     )

#     instrument_config = cxs.BasicImageConfig(
#         shape=shape,
#         pixel_size=pixel_size,
#         voltage_in_kilovolts=voltage_in_kilovolts,
#     )
#     dim = shape[0]
#     voxel_potential = cxs.RealVoxelGridVolume.from_real_voxel_grid(
#         atom_potential.to_real_voxel_grid((dim, dim, dim), pixel_size),
#     )

#     multislice_integrator = cxe.FFTMultisliceIntegrator(
#         slice_thickness_in_voxels=3,
#     )
#     pose = cxs.EulerAnglePose()
#     pose_inv = pose.to_inverse_rotation()

#     ctf = cxs.AberratedAstigmaticCTF(
#         defocus_in_angstroms=defocus_in_angstroms,
#         astigmatism_in_angstroms=astigmatism_in_angstroms,
#         astigmatism_angle=astigmatism_angle,
#     )

#     multislice_scattering_theory = cxe.MultisliceScatteringTheory(
#         multislice_integrator,
#         cxe.WaveTransferTheory(ctf),
#         amplitude_contrast_ratio=ac,
#     )
#     high_energy_scattering_theory = cxe.HighEnergyScatteringTheory(
#         cxs.GaussianMixtureProjection(use_error_functions=True),
#         cxe.WaveTransferTheory(ctf),
#         amplitude_contrast_ratio=ac,
#     )
#     weak_phase_scattering_theory = cxs.WeakPhaseScatteringTheory(
#         cxs.GaussianMixtureProjection(use_error_functions=True),
#         cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=ac),
#     )

#     multislice_image_model_voxel = cxs.IntensityImageModel(
#         voxel_potential, pose_inv, instrument_config, multislice_scattering_theory
#     )
#     high_energy_image_model = cxs.IntensityImageModel(
#         atom_potential, pose, instrument_config, high_energy_scattering_theory
#     )
#     weak_phase_image_model = cxs.IntensityImageModel(
#         atom_potential, pose, instrument_config, weak_phase_scattering_theory
#     )

#     ms = multislice_image_model_voxel.simulate()
#     he = high_energy_image_model.simulate()
#     wp = weak_phase_image_model.simulate()

#     normalize_image = lambda image: (image - image.mean()) / image.std()

#     atol = 0.1
#     from matplotlib import pyplot as plt

#     # fig, axes = plt.subplots(ncols=3)
#     # vmin, vmax = min(np.amin(he), np.amin(wp), np.amin(ms)), max(
#     #     np.amax(he), np.amax(wp), np.amax(ms)
#     # )
#     # im = axes[0].imshow(he, vmin=vmin, vmax=vmax)
#     # axes[1].imshow(wp, vmin=vmin, vmax=vmax)
#     # axes[2].imshow(ms, vmin=vmin, vmax=vmax)
#     # plt.colorbar(im)
#     # plt.show()

#     np.testing.assert_allclose(normalize_image(he), normalize_image(wp), atol=atol)
#     np.testing.assert_allclose(normalize_image(he), normalize_image(ms), atol=atol)
#     np.testing.assert_allclose(normalize_image(ms), normalize_image(wp), atol=atol)


pixel_size = 8.0
shape_0 = 32
@pytest.mark.parametrize(
    "pixel_size, shape, euler_pose_params, ctf_params",
    (
        (
            pixel_size,
            (shape_0, shape_0),
            (2.5, -5.0, 0.0, 0.0, 0.0),
            (0.1, 300.0, 10000.0, -100.0, 10.0),
        ),
        (
            pixel_size,
            (shape_0, shape_0),
            (0.0, 0.0, 10.0, -30.0, 60.0),
            (0.1, 300.0, 10000.0, -100.0, 10.0),
        ),
        (
            pixel_size,
            (shape_0, shape_0),
            (2.5, -5.0, 10.0, -30.0, 60.0),
            (0.1, 300.0, 10000.0, -100.0, 10.0),
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

    logger.info(f"Testing with amplitude contrast ratio: {ac}")
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="name CA ",
        loads_b_factors=True,
    )
    logger.info(f"Loaded atom positions: {atom_positions.shape[0]}")

    logger.info(f'Loading atom potential...')
    atom_potential = cxs.PengIndependentAtomPotential.from_tabulated_parameters(
        atom_positions,
        parameters=cxs.PengScatteringFactorParameters(atom_types),
        extra_b_factors=b_factors,
    )
    logger.info(f'Creating instrument configuration...')
    instrument_config = cxs.BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage_in_kilovolts,
    )
    dim = shape[0]

    logger.info(f'Creating voxel potential...')
    voxel_potential = cxs.RealVoxelGridVolume.from_real_voxel_grid(
        atom_potential.to_real_voxel_grid((dim, dim, dim), pixel_size),
    )
    logger.info(f'Creating multislice integrator...')
    multislice_integrator = cxe.FFTMultisliceIntegrator(
        slice_thickness_in_voxels=3,
    )
    logger.info(f'Creating pose...')
    pose = cxs.EulerAnglePose(*euler_pose_params)
    pose_inv = pose.to_inverse_rotation()
    logger.info(f'Creating CTF...')
    ctf = cxs.AberratedAstigmaticCTF(
        defocus_in_angstroms=defocus_in_angstroms,
        astigmatism_in_angstroms=astigmatism_in_angstroms,
        astigmatism_angle=astigmatism_angle,
    )

    logger.info(f'Creating multislice scattering theory...')
    multislice_scattering_theory = cxe.MultisliceScatteringTheory(
        multislice_integrator,
        cxe.WaveTransferTheory(ctf),
        amplitude_contrast_ratio=ac,
    )
    logger.info(f'Creating high energy scattering theory...')
    high_energy_scattering_theory = cxe.HighEnergyScatteringTheory(
        cxs.GaussianMixtureProjection(use_error_functions=True),
        cxe.WaveTransferTheory(ctf),
        amplitude_contrast_ratio=ac,
    )
    logger.info(f'Creating weak phase scattering theory...')
    weak_phase_scattering_theory = cxs.WeakPhaseScatteringTheory(
        cxs.GaussianMixtureProjection(use_error_functions=True),
        cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=ac),
    )
    logger.info(f'Creating multislice image model (voxel)...')
    multislice_image_model_voxel = cxs.IntensityImageModel(
        voxel_potential, pose_inv, instrument_config, multislice_scattering_theory
    )
    logger.info(f'Creating high energy image model...')
    high_energy_image_model = cxs.IntensityImageModel(
        atom_potential, pose, instrument_config, high_energy_scattering_theory
    )
    logger.info(f'Creating weak phase image model...')
    weak_phase_image_model = cxs.IntensityImageModel(
        atom_potential, pose, instrument_config, weak_phase_scattering_theory
    )
    logger.info(f'Simulating multislice image model (voxel)...')
    ms = multislice_image_model_voxel.simulate()
    logger.info(f'Simulating high energy image model...')
    he = high_energy_image_model.simulate()
    logger.info(f'Simulating weak phase image model...')
    wp = weak_phase_image_model.simulate()

    logger.info(f'Normalizing images...')
    normalize_image = lambda image: (image - image.mean()) / image.std()

    logger.info(f'Comparing images...')
    atol = 9.0
    np.testing.assert_allclose(normalize_image(he), normalize_image(wp), atol=atol)
    np.testing.assert_allclose(normalize_image(he), normalize_image(ms), atol=atol)
    np.testing.assert_allclose(normalize_image(ms), normalize_image(wp), atol=atol)

    close_fraction = 0.95
    atol = 1.0
    assert np.isclose(normalize_image(ms), normalize_image(wp), atol=atol).astype(float).mean() > close_fraction
    logger.info('Done')

if __name__ == "__main__":
    test_scattering_theories_pose(
        sample_pdb_path="/Users/gw/repos/cryojax/docs/examples/data/groel_chainA.pdb",
        pixel_size=4.0,
        shape=(74, 74),
        euler_pose_params=(2.5, -5.0, 0.0, 0.0, 0.0),
        ctf_params=(0.1, 300.0, 10000.0, -100.0, 10.0),
    )