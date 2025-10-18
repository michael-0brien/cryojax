import warnings

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.io import read_atoms_from_pdb
from cryojax.ndimage import crop_to_shape, irfftn
from jaxtyping import Array


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (1.0, (32, 32)),
        (1.0, (31, 31)),
        (1.0, (31, 32)),
        (1.0, (32, 31)),
    ),
)
def test_projection_methods_no_pose(sample_pdb_path, pixel_size, shape):
    """
    Test that computing a projection in real
    space agrees with real-space, with no rotation. This mostly
    makes sure there are no numerical artifacts in fourier space
    interpolation and that volumes are read in real vs. fourier
    at the same orientation.
    """
    # Objects for imaging
    image_config = cxs.BasicImageConfig(
        shape,
        pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # Real vs fourier volumes
    dim = max(*shape)  # Make sure to use `padded_shape` here
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        sample_pdb_path, center=True, loads_b_factors=True
    )
    scattering_factor_parameters = cxs.PengScatteringFactorParameters(atom_types)
    base_volume = cxs.PengAtomicVolume.from_tabulated_parameters(
        atom_positions,
        scattering_factor_parameters,
        extra_b_factors=b_factors,
    )
    base_method = cxs.GaussianMixtureProjection(use_error_functions=True)

    real_voxel_grid = base_volume.to_real_voxel_grid((dim, dim, dim), pixel_size)
    other_volumes = [
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid),
        make_spline(real_voxel_grid),
        cxs.GaussianMixtureVolume(
            atom_positions,
            4 * jnp.pi * scattering_factor_parameters.a,
            (scattering_factor_parameters.b + b_factors[:, None]) / (8 * jnp.pi**2),
        ),
        cxs.RealVoxelGridVolume.from_real_voxel_grid(real_voxel_grid),
    ]
    other_projection_methods = [
        cxs.FourierSliceExtraction(),
        cxs.FourierSliceExtraction(),
        base_method,
        cxs.NufftProjection(eps=1e-16),
    ]

    projection_by_gaussian_integration = compute_projection(
        base_volume, base_method, image_config
    )
    for volume, projection_method in zip(other_volumes, other_projection_methods):
        if isinstance(projection_method, cxs.NufftProjection):
            try:
                projection_by_other_method = compute_projection(
                    volume, projection_method, image_config
                )
            except RuntimeError as err:
                warnings.warn(
                    "Could not test projection method `NufftProjection`, "
                    "most likely because `jax_finufft` is not installed. "
                    f"Error traceback is:\n{err}"
                )
                continue
        else:
            projection_by_other_method = compute_projection(
                volume, projection_method, image_config
            )
        np.testing.assert_allclose(
            projection_by_gaussian_integration, projection_by_other_method, atol=1e-12
        )


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
