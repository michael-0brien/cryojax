import cryojax.simulator as cxs
import equinox as eqx
import jax
import numpy as np
import pytest
from cryojax.atom_util import split_atoms_by_element
from cryojax.constants import PengScatteringFactorParameters
from cryojax.io import read_atoms_from_pdb
from cryojax.ndimage import crop_to_shape, irfftn, operators as op
from jaxtyping import Array


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def pdb_info(sample_pdb_path):
    return read_atoms_from_pdb(sample_pdb_path, center=True, loads_properties=True)


def test_fft_atom_bad_instantiation():
    with pytest.raises(ValueError):
        _ = cxs.IndependentAtomVolume(
            position_pytree=np.zeros((10, 3)),
            scattering_factor_pytree=(op.FourierGaussian(),),
        )


def test_fft_atom_projection_exact(pdb_info):
    atom_positions, _, _ = pdb_info
    pixel_size, shape = 0.5, (64, 64)
    pad_options = dict(shape=(128, 128))
    image_config = cxs.BasicImageConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, pad_options=pad_options
    )
    amplitude, b_factor = 1.0, 100.0
    gaussian_volume, gaussian_integrator = (
        cxs.GaussianMixtureVolume(
            atom_positions,
            amplitudes=amplitude,
            variances=b_factor / (8 * np.pi**2),
        ),
        cxs.GaussianMixtureProjection(use_error_functions=False),
    )
    atom_volume, fft_integrator = (
        cxs.IndependentAtomVolume(
            position_pytree=atom_positions,
            scattering_factor_pytree=op.FourierGaussian(
                amplitude=amplitude, b_factor=b_factor
            ),
        ),
        cxs.FFTAtomProjection(antialias=False, eps=1e-16),
    )
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_fft = compute_projection(atom_volume, fft_integrator, image_config)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_fft, atol=1e-8)


@pytest.mark.parametrize(
    "width, pixel_size, shape",
    ((5.0, 0.5, (64, 64)), (1.0, 0.5, (64, 64)), (2.0, 1.0, (32, 32))),
)
def test_fft_atom_projection_antialias(pdb_info, width, pixel_size, shape):
    atom_positions, _, _ = pdb_info
    gaussian_volume = cxs.GaussianMixtureVolume(
        atom_positions,
        amplitudes=1.0,
        variances=width**2,
    )
    atom_volume = cxs.IndependentAtomVolume(
        position_pytree=atom_positions,
        scattering_factor_pytree=op.FourierGaussian(
            amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)
        ),
    )
    gaussian_integrator = cxs.GaussianMixtureProjection(use_error_functions=True)
    fft_integrator = cxs.FFTAtomProjection(eps=1e-16)
    pad_options = dict(shape=(2 * shape[0], 2 * shape[1]))
    image_config = cxs.BasicImageConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, pad_options=pad_options
    )
    fft_integrator = cxs.FFTAtomProjection(eps=1e-16)
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_fft = compute_projection(atom_volume, fft_integrator, image_config)

    np.testing.assert_allclose(proj_by_gaussians, proj_by_fft, atol=1e-8)


@pytest.mark.parametrize(
    "pixel_size, shape",
    ((0.125, (256, 256)),),
)
def test_fft_atom_projection_peng(pdb_info, pixel_size, shape):
    atom_positions, atom_ids, _ = pdb_info
    positions_by_id, unique_atom_ids = split_atoms_by_element(atom_ids, atom_positions)
    peng_parameters, peng_parameters_by_id = (
        PengScatteringFactorParameters(atom_ids),
        PengScatteringFactorParameters(unique_atom_ids),
    )
    gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        peng_parameters,
    )
    atom_volume = cxs.IndependentAtomVolume.from_tabulated_parameters(
        positions_by_id,
        peng_parameters_by_id,
    )
    pad_options = dict(shape=(2 * shape[0], 2 * shape[1]))
    image_config = cxs.BasicImageConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, pad_options=pad_options
    )
    # Check to make sure the implementations are identical, up to the
    # nufft (don't include anti-aliasing)
    gaussian_integrator = cxs.GaussianMixtureProjection(use_error_functions=False)
    fft_integrator = cxs.FFTAtomProjection(antialias=False, eps=1e-16)
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_fft = compute_projection(atom_volume, fft_integrator, image_config)

    plot_images(proj_by_gaussians, proj_by_fft)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_fft, atol=1e-8)

    # Check antialiasing
    # gaussian_integrator = cxs.GaussianMixtureProjection(use_error_functions=True)
    # fft_integrator = cxs.FFTAtomProjection(antialias=True, eps=1e-16)
    # proj_by_gaussians = compute_projection(
    #     gaussian_volume, gaussian_integrator, image_config
    # )
    # proj_by_fft = compute_projection(atom_volume, fft_integrator, image_config)

    # plot_images(proj_by_gaussians, proj_by_fft)


# np.testing.assert_allclose(proj_by_gaussians, proj_by_fft, atol=1e-8)


# @pytest.mark.parametrize(
#     "pixel_size, shape",
#     (
#         (1.0, (32, 32)),
#         (1.0, (31, 31)),
#         (1.0, (31, 32)),
#         (1.0, (32, 31)),
#     ),
# )
# def test_projection_methods_no_pose(pdb_info, pixel_size, shape):
#     """
#     Test that computing a projection in real
#     space agrees with real-space, with no rotation. This mostly
#     makes sure there are no numerical artifacts in fourier space
#     interpolation and that volumes are read in real vs. fourier
#     at the same orientation.
#     """
#     # Unpack PDB info
#     atom_positions, atom_types, atom_properties = pdb_info
#     # Objects for imaging
#     image_config = cxs.BasicImageConfig(
#         shape,
#         pixel_size,
#         voltage_in_kilovolts=300.0,
#     )
#     # Real vs fourier volumes
#     dim = max(*shape)  # Make sure to use `padded_shape` here
#     positions_by_id, atom_id = split_atoms_by_element(atom_types, atom_positions)

#     peng_parameters = PengScatteringFactorParameters(atom_types)
#     peng_parameters_by_id = PengScatteringFactorParameters(atom_id)
#     base_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
#         atom_positions,
#         peng_parameters,
#         extra_b_factors=atom_properties["b_factors"],
#     )
#     base_method = cxs.GaussianMixtureProjection(use_error_functions=True)
#     render_volume_fn = cxs.GaussianMixtureRenderFn((dim, dim, dim), pixel_size)
#     real_voxel_grid = render_volume_fn(base_volume)
#     other_volumes = [
#         cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid),
#         make_spline(real_voxel_grid),
#     ]
#     other_projection_methods = [
#         cxs.FourierSliceExtraction(),
#         cxs.FourierSliceExtraction(),
#     ]
#     try:
#         other_projection_methods.extend(
#             [
#                 cxs.FFTAtomProjection(antialias=True, eps=1e-16),
#                 cxs.RealVoxelProjection(eps=1e-16),
#             ]  # type: ignore
#         )
#         other_volumes.extend(
#             [
#                 cxs.IndependentAtomVolume.from_tabulated_parameters(
#                     positions_by_id, peng_parameters_by_id
#                 ),
#                 cxs.RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid),
#             ]
#         )
#     except RuntimeError as err:
#         warnings.warn(
#             "Could not test projection method `NufftProjection`, "
#             "most likely because `jax_finufft` is not installed. "
#             f"Error traceback is:\n{err}"
#         )

#     projection_by_gaussian_integration = compute_projection(
#         base_volume, base_method, image_config
#     )
#     for volume, projection_method in zip(other_volumes, other_projection_methods):
#         projection_by_other_method = compute_projection(
#             volume, projection_method, image_config
#         )
#         np.testing.assert_allclose(
#             projection_by_gaussian_integration, projection_by_other_method, atol=1e-12
#         )


# @pytest.mark.parametrize(
#     "pixel_size, shape, euler_pose_params",
#     (
#         (1.0, (32, 32), (2.5, -5.0, 0.0, 0.0, 0.0)),
#         (1.0, (32, 32), (0.0, 0.0, 10.0, -30.0, 60.0)),
#         (1.0, (32, 32), (2.5, -5.0, 10.0, -30.0, 60.0)),
#     ),
# )
# def test_projection_methods_with_pose(
#     sample_pdb_path, pixel_size, shape, euler_pose_params
# ):
#     """Test that computing a projection across different
#     methods agrees. This tests pose convention and accuracy
#     for real vs fourier, atoms vs voxels, etc.
#     """
#     # Objects for imaging
#     instrument_config = cxs.BasicImageConfig(
#         shape,
#         pixel_size,
#         voltage_in_kilovolts=300.0,
#     )
#     euler_pose = cxs.EulerAnglePose(*euler_pose_params)
#     # Real vs fourier potentials
#     dim = max(*shape)
#     atom_positions, atom_types, b_factors = read_atoms_from_pdb(
#         sample_pdb_path, center=True, loads_b_factors=True
#     )
#     scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
#         atom_types, read_peng_element_scattering_factor_parameter_table()
#     )
#     base_potential = cxs.PengAtomicPotential(
#         atom_positions,
#         scattering_factor_a=scattering_factor_parameters["a"],
#         scattering_factor_b=scattering_factor_parameters["b"],
#         b_factors=b_factors,
#     )
#     base_method = cxs.GaussianMixtureProjection(use_error_functions=True)

#     real_voxel_grid = base_potential.as_real_voxel_grid((dim, dim, dim), pixel_size)
#     other_potentials = [
#         cxs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
#         make_spline_potential(real_voxel_grid, pixel_size),
#         cxs.GaussianMixtureAtomicPotential(
#             atom_positions,
#             scattering_factor_parameters["a"],
#             (scattering_factor_parameters["b"] + b_factors[:, None]) / (8 * jnp.pi**2),
#         ),
#     ]
#     #     cxs.RealVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
#     #     cxs.RealVoxelCloudPotential.from_real_voxel_grid(real_voxel_grid, pixel_size),
#     # ]
#     other_projection_methods = [
#         cxs.FourierSliceExtraction(),
#         cxs.FourierSliceExtraction(),
#         base_method,
#     ]
#     #     cxs.NufftProjection(),
#     #     cxs.NufftProjection(),
#     # ]

#     projection_by_gaussian_integration = compute_projection_at_pose(
#         base_potential, base_method, euler_pose, instrument_config
#     )
#     for idx, (potential, projection_method) in enumerate(
#         zip(other_potentials, other_projection_methods)
#     ):
#         if isinstance(projection_method, cxs.NufftProjection):
#             try:
#                 projection_by_other_method = compute_projection_at_pose(
#                     potential, projection_method, euler_pose, instrument_config
#                 )
#             except Exception as err:
#                 warnings.warn(
#                     "Could not test projection method `NufftProjection` "
#                     "This is most likely because `jax_finufft` is not installed. "
#                     f"Error traceback is:\n{err}"
#                 )
#                 continue
#         else:
#             projection_by_other_method = compute_projection_at_pose(
#                 potential, projection_method, euler_pose, instrument_config
#             )
#         np.testing.assert_allclose(
#             np.sum(
#                 (projection_by_gaussian_integration - projection_by_other_method) ** 2
#             ),
#             0.0,
#             atol=1e-8,
#         )


def plot_images(proj1, proj2):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    vmin, vmax = min(proj1.min(), proj2.min()), max(proj1.max(), proj2.max())
    fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
    im1 = axes[0].imshow(proj1, vmin=vmin, vmax=vmax, cmap="gray")
    im2 = axes[1].imshow(proj2, vmin=vmin, vmax=vmax, cmap="gray")
    for im, ax in zip([im1, im2], axes):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    im3 = axes[2].imshow(np.abs(proj2 - proj1), cmap="gray")
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax)
    plt.show()


@eqx.filter_jit
def compute_projection(
    volume: cxs.AbstractVolumeRepresentation,
    integrator: cxs.AbstractVolumeIntegrator,
    image_config: cxs.BasicImageConfig,
) -> Array:
    fourier_projection = integrator.integrate(
        volume, image_config, outputs_real_space=False
    )
    return crop_to_shape(
        irfftn(
            fourier_projection,
            s=image_config.padded_shape,
        ),
        image_config.shape,
    )


@eqx.filter_jit
def compute_projection_at_pose(
    volume: cxs.AbstractVolumeRepresentation,
    integrator: cxs.AbstractVolumeIntegrator,
    pose: cxs.AbstractPose,
    image_config: cxs.BasicImageConfig,
) -> Array:
    rotated_volume = volume.rotate_to_pose(pose)
    fourier_projection = integrator.integrate(
        rotated_volume, image_config, outputs_real_space=False
    )
    translation_operator = pose.compute_translation_operator(
        image_config.padded_frequency_grid_in_angstroms
    )
    return crop_to_shape(
        irfftn(
            pose.translate_image(
                fourier_projection,
                translation_operator,
                image_config.padded_shape,
            ),
            s=image_config.padded_shape,
        ),
        image_config.shape,
    )


@eqx.filter_jit
def make_spline(real_voxel_grid):
    return cxs.FourierVoxelSplineVolume.from_real_voxel_grid(
        real_voxel_grid,
    )
