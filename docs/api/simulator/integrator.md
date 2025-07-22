# Scattering potential integration methods

`cryojax` provides different methods for integrating [scattering potentials](./potential.md#scattering-potential-representations) onto a plane.

???+ abstract "`cryojax.simulator.AbstractPotentialIntegrator`"
    ::: cryojax.simulator.AbstractPotentialIntegrator
        options:
            members:
                - compute_fourier_integrated_potential

## Integration methods for voxel-based potentials

??? abstract "`cryojax.simulator.AbstractVoxelPotentialIntegrator`"
    ::: cryojax.simulator.AbstractVoxelPotentialIntegrator
        options:
            members:
                - pixel_rescaling_method

::: cryojax.simulator.FourierSliceExtraction
        options:
            members:
                - __init__
                - compute_fourier_integrated_potential
                - extract_fourier_slice_from_spline_coefficients
                - extract_fourier_slice_from_grid_points

---

::: cryojax.simulator.NufftProjection
        options:
            members:
                - __init__
                - compute_fourier_integrated_potential
                - project_voxel_cloud_with_nufft

## Integration methods for atom-based potentials

::: cryojax.simulator.GaussianMixtureProjection
        options:
            members:
                - __init__
                - compute_fourier_integrated_potential
