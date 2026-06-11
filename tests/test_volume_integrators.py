import warnings

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import numpy as np
import pytest
from cryojax.atom_util import split_atoms_by_element
from cryojax.constants import PengScatteringFactorParameters
from cryojax.io import read_atoms_from_pdb
from jaxtyping import Array


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


@pytest.fixture
def pdb_info(sample_pdb_path):
    return read_atoms_from_pdb(sample_pdb_path, center=True, loads_properties=True)


@pytest.mark.parametrize("shape", ((64, 64), (63, 63), (63, 64), (64, 63)))
def test_gmm_integrator_shape(sample_pdb_path, shape):
    atom_volume = cxs.load_tabulated_volume(
        sample_pdb_path,
        output_type=cxs.GaussianMixtureVolume,
        include_b_factors=True,
        selection_string="not element H",
    )
    pixel_size = 0.5

    integrator = cxs.GaussianMixtureProjection(shape=(2 * shape[0], 2 * shape[1]))
    # # ... and the configuration of the imaging instrument
    image_config = cxs.BasicImageConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # ... compute the integrated volume
    fourier_integrated_volume = integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )

    assert fourier_integrated_volume.shape == (shape[0], shape[1] // 2 + 1)


def test_fft_atom_bad_instantiation():
    with pytest.raises(ValueError):
        _ = cxs.IndependentAtomVolume(
            positions=np.zeros((10, 3)),
            kernel_fns=(im.FourierGaussian(),),
        )


@pytest.mark.parametrize(
    "pixel_size, shape",
    ((1.0, (32, 32)), (1.0, (32, 31)), (1.0, (31, 32)), (1.0, (31, 31))),
)
def test_fft_atom_projection_exact(pdb_info, pixel_size, shape):
    for backend in ["jax-finufft", "nufftax"]:
        if jnufft is None and backend == "jax-finufft":
            warnings.warn(
                "Could not test projection method `IndependentAtomProjection`, "
                "most likely because `jax_finufft` is not installed. "
                f"Error traceback is:\n{JAX_FINUFFT_IMPORT_ERROR}"
            )
            continue
        atom_positions, _, _ = pdb_info
        pixel_size, shape = 0.5, (64, 64)
        image_config = cxs.BasicImageConfig(
            shape, pixel_size, voltage_in_kilovolts=300.0, padded_shape=(128, 128)
        )
        amplitude, b_factor = 1.0, 100.0
        gaussian_volume, gaussian_integrator = (
            cxs.GaussianMixtureVolume(
                atom_positions,
                amplitudes=amplitude,
                variances=b_factor / (8 * np.pi**2),
            ),
            cxs.GaussianMixtureProjection(sampling_mode="point"),
        )
        atom_volume, atom_integrator = (
            cxs.IndependentAtomVolume(
                positions=atom_positions,
                kernel_fns=im.FourierGaussian(amplitude=amplitude, b_factor=b_factor),
            ),
            cxs.IndependentAtomProjection(
                backend=backend,  # type: ignore
                sampling_mode="point",
                eps=1e-10,
            ),
        )
        proj_by_gaussians = compute_projection(
            gaussian_volume, gaussian_integrator, image_config
        )
        proj_by_atom = compute_projection(atom_volume, atom_integrator, image_config)
        np.testing.assert_allclose(proj_by_gaussians, proj_by_atom, atol=1e-8)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (0.5, (63, 63)),
        (0.5, (64, 64)),
    ),
)
def test_real_atom_projection_exact(pdb_info, pixel_size, shape):
    atom_positions, _, _ = pdb_info
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    amplitude, variance = 0.75, 20.0 / (8 * np.pi**2)
    gaussian_volume, gaussian_integrator = (
        cxs.GaussianMixtureVolume(
            atom_positions, amplitudes=amplitude, variances=variance
        ),
        cxs.GaussianMixtureProjection(sampling_mode="average"),
    )
    atom_volume, atom_integrator = (
        cxs.IndependentAtomVolume(
            positions=atom_positions,
            kernel_fns=im.RealGaussian(amplitude=amplitude, variance=variance),
        ),
        cxs.IndependentAtomProjection(sampling_mode="average", eps=1e-12),
    )
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atom = compute_projection(atom_volume, atom_integrator, image_config)

    np.testing.assert_allclose(proj_by_gaussians, proj_by_atom, atol=1e-5)


@pytest.mark.parametrize(
    "width, pixel_size, shape",
    ((5.0, 0.5, (64, 64)), (1.0, 0.5, (64, 64)), (2.0, 1.0, (32, 32))),
)
def test_fft_atom_projection_antialias(pdb_info, width, pixel_size, shape):
    for backend in ["jax-finufft", "nufftax"]:
        if jnufft is None and backend == "jax-finufft":
            continue
        atom_positions, _, _ = pdb_info
        gaussian_volume = cxs.GaussianMixtureVolume(
            atom_positions,
            amplitudes=1.0,
            variances=width**2,
        )
        atom_volume = cxs.IndependentAtomVolume(
            positions=atom_positions,
            kernel_fns=im.FourierGaussian(
                amplitude=1.0, b_factor=width**2 * (8 * np.pi**2)
            ),
        )
        gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
        atom_integrator = cxs.IndependentAtomProjection(
            eps=1e-10,
            backend=backend,  # type: ignore
            upsample_factor=2.0,
        )
        padded_shape = (2 * shape[0], 2 * shape[1])
        image_config = cxs.BasicImageConfig(
            shape, pixel_size, voltage_in_kilovolts=300.0, padded_shape=padded_shape
        )
        proj_by_gaussians = compute_projection(
            gaussian_volume, gaussian_integrator, image_config
        )
        proj_by_atoms = compute_projection(atom_volume, atom_integrator, image_config)

        np.testing.assert_allclose(proj_by_gaussians, proj_by_atoms, atol=1e-8)


@pytest.mark.parametrize(
    "pixel_size, shape, upsampfac",
    (
        (0.25, (134, 134), 2.62),
        (0.25, (133, 133), 2),
        (0.25, (134, 133), 3),
        (0.25, (133, 133), 3),
    ),
)
def test_fft_atom_projection_peng(pdb_info, pixel_size, shape, upsampfac):
    for backend in ["jax-finufft", "nufftax"]:
        if jnufft is None and backend == "jax-finufft":
            continue
        atom_positions, atom_ids, _ = pdb_info
        positions_by_id, unique_atom_ids = split_atoms_by_element(
            atom_ids, atom_positions
        )
        peng_parameters, peng_parameters_by_id = (
            PengScatteringFactorParameters(atom_ids),
            PengScatteringFactorParameters(unique_atom_ids),
        )
        gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
            atom_positions,
            peng_parameters,
            extra_b_factors=4.0,
        )
        atom_volume = cxs.IndependentAtomVolume.from_tabulated_parameters(
            positions_by_id,
            peng_parameters_by_id,
            b_factor_by_element=4.0,
            use_real_space=False,
        )
        image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
        # Check to make sure the implementations are identical, up to the
        # nufft (don't include anti-aliasing)
        gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
        atom_integrator = cxs.IndependentAtomProjection(
            backend=backend,  # type: ignore
            sampling_mode="average",
            upsample_factor=upsampfac,
            eps=1e-10,
        )
        proj_by_gaussians = compute_projection(
            gaussian_volume, gaussian_integrator, image_config
        )
        proj_by_atoms = compute_projection(atom_volume, atom_integrator, image_config)
        np.testing.assert_allclose(proj_by_gaussians, proj_by_atoms, atol=5e-3)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (0.5, (64, 64)),
        (0.5, (63, 63)),
    ),
)
def test_real_atom_projection_peng(pdb_info, pixel_size, shape):
    atom_positions, atom_ids, _ = pdb_info
    positions_by_id, unique_atom_ids = split_atoms_by_element(atom_ids, atom_positions)
    peng_parameters, peng_parameters_by_id = (
        PengScatteringFactorParameters(atom_ids),
        PengScatteringFactorParameters(unique_atom_ids),
    )
    gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        peng_parameters,
        extra_b_factors=10.0,
    )
    atom_volume = cxs.IndependentAtomVolume.from_tabulated_parameters(
        positions_by_id,
        peng_parameters_by_id,
        b_factor_by_element=10.0,
        use_real_space=True,
    )
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    # Check to make sure the implementations are identical, up to the
    # nufft (don't include anti-aliasing)
    gaussian_integrator = cxs.GaussianMixtureProjection(sampling_mode="average")
    atom_integrator = cxs.IndependentAtomProjection(sampling_mode="average", eps=1e-10)
    proj_by_gaussians = compute_projection(
        gaussian_volume, gaussian_integrator, image_config
    )
    proj_by_atoms = compute_projection(atom_volume, atom_integrator, image_config)
    # _plot_image_compare(proj_by_gaussians, proj_by_atoms)
    np.testing.assert_allclose(proj_by_gaussians, proj_by_atoms, atol=5e-3)


@pytest.mark.parametrize(
    "pixel_size, shape",
    (
        (1.0, (32, 32)),
        (1.0, (31, 31)),
        (1.0, (31, 32)),
        (1.0, (32, 31)),
    ),
)
def test_analytic_vs_voxels_nopose(pdb_info, pixel_size, shape):
    """
    Test that computing a projection analytically with gaussians
    agrees with numerical results for voxel-based volumes.
    """
    # Unpack PDB info
    atom_positions, atom_types, atom_properties = pdb_info
    # Objects for imaging
    image_config = cxs.BasicImageConfig(
        shape,
        pixel_size,
        voltage_in_kilovolts=300.0,
    )
    # Real vs fourier volumes
    dim = max(*shape)  # Make sure to use `padded_shape` here

    peng_parameters = PengScatteringFactorParameters(atom_types)
    base_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        peng_parameters,
        extra_b_factors=atom_properties["b_factors"],
    )
    base_method = cxs.GaussianMixtureProjection(sampling_mode="average")
    render_volume_fn = cxs.GaussianMixtureRenderFn((dim, dim, dim), pixel_size)
    real_voxel_grid = render_volume_fn(base_volume)
    other_volumes = [
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid),
        make_spline(real_voxel_grid),
        cxs.RealVoxelCloudVolume.from_real_voxel_grid(real_voxel_grid),
    ]
    other_projection_methods = [
        cxs.FourierSliceExtraction(),
        cxs.FourierSliceExtraction(),
        cxs.RealVoxelProjection(eps=1e-15, backend="nufftax"),
    ]
    if jnufft is not None:
        other_volumes.append(other_volumes[-1])
        other_projection_methods.append(
            cxs.RealVoxelProjection(eps=1e-15, backend="jax-finufft")  # type: ignore
        )

    projection_by_gaussian_integration = compute_projection(
        base_volume, base_method, image_config
    )
    # from matplotlib import pyplot as plt

    for volume, projection_method in zip(other_volumes, other_projection_methods):
        projection_by_other_method = compute_projection(
            volume, projection_method, image_config
        )
        # fig, axes = plt.subplots(figsize=(12, 4), ncols=3)
        # axes[0].imshow(projection_by_gaussian_integration)
        # axes[1].imshow(projection_by_other_method)
        # axes[2].imshow(projection_by_other_method - projection_by_gaussian_integration)
        # plt.show()
        np.testing.assert_allclose(
            projection_by_gaussian_integration, projection_by_other_method, atol=1e-8
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
#     base_method = cxs.GaussianMixtureProjection(sampling_mode="average")

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


# def plot_images(proj1, proj2):
#     from matplotlib import pyplot as plt
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

#     vmin, vmax = min(proj1.min(), proj2.min()), max(proj1.max(), proj2.max())
#     fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
#     im1 = axes[0].imshow(proj1, vmin=vmin, vmax=vmax, cmap="gray")
#     im2 = axes[1].imshow(proj2, vmin=vmin, vmax=vmax, cmap="gray")
#     for im, ax in zip([im1, im2], axes):
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         fig.colorbar(im, cax=cax)
#     im3 = axes[2].imshow(np.abs(proj2 - proj1), cmap="gray")
#     divider = make_axes_locatable(axes[2])
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(im3, cax=cax)
#     plt.show()


# @eqx.filter_jit
def compute_projection(
    volume: cxs.AbstractVolumeRepresentation,
    integrator: cxs.AbstractVolumeIntegrator,
    image_config: cxs.BasicImageConfig,
) -> Array:
    fourier_projection = integrator.integrate(
        volume, image_config, outputs_real_space=False
    )
    return im.crop_to_shape(
        im.irfftn(
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
        image_config.get_frequency_grid(padding=True, physical=True)
    )
    return im.crop_to_shape(
        im.irfftn(
            pose.translate_image(
                fourier_projection,
                translation_operator,
                image_config.padded_shape,
            ),
            s=image_config.padded_shape,
        ),
        image_config.shape,
    )


@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_projection_custom_upsampfac(upsampfac):
    """Smoke test: passing a custom upsampfac via options runs without error."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, pixel_size = (8, 8), 1.0
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_volume = cxs.IndependentAtomVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    atom_integrator = cxs.IndependentAtomProjection(
        backend="nufftax",
        sampling_mode="point",
        eps=1e-6,
        options={"upsampfac": upsampfac},
    )
    result = atom_integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )
    assert result.shape == (shape[0], shape[1] // 2 + 1)


@pytest.mark.parametrize("upsampfac", (1.25, 2.0))
def test_fft_atom_projection_custom_upsampfac_jax_finufft(upsampfac):
    """Smoke test: passing custom opts via options to jax-finufft runs without error."""
    if jnufft is None:
        pytest.skip("jax-finufft not installed")
    from jax_finufft.options import NestedOpts, Opts

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]])
    shape, pixel_size = (8, 8), 1.0
    image_config = cxs.BasicImageConfig(shape, pixel_size, voltage_in_kilovolts=300.0)
    atom_volume = cxs.IndependentAtomVolume(
        positions=positions,
        kernel_fns=im.FourierGaussian(amplitude=1.0, b_factor=100.0),
    )
    opts = NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )
    atom_integrator = cxs.IndependentAtomProjection(
        backend="jax-finufft",
        sampling_mode="point",
        eps=1e-6,
        options={"opts": opts},
    )
    result = atom_integrator.integrate(
        atom_volume, image_config, outputs_real_space=False
    )
    assert result.shape == (shape[0], shape[1] // 2 + 1)


@eqx.filter_jit
def make_spline(real_voxel_grid):
    return cxs.FourierVoxelSplineVolume.from_real_voxel_grid(
        real_voxel_grid,
    )


# def _plot_image_compare(im1, im2):
#     from matplotlib import pyplot as plt

#     fig, axes = plt.subplots(figsize=(9, 4), ncols=2, constrained_layout=True)
#     mappable1 = axes[0].imshow(im1)
#     mappable2 = axes[1].imshow(im2)
#     fig.colorbar(mappable1)
#     fig.colorbar(mappable2)
#     plt.show()
