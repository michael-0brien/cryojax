# Scattering potential integration methods

`cryojax` provides different methods for integrating [scattering potentials](./potential.md#scattering-potential-representations) onto a plane.

???+ abstract "`cryojax.simulator.AbstractDirectIntegrator`"
    ::: cryojax.simulator.AbstractDirectIntegrator
        options:
            members:
                - integrate

## Integration methods for voxel-based potentials

::: cryojax.simulator.FourierSliceExtraction
        options:
            members:
                - __init__
                - integrate
                - extract_fourier_slice_from_spline_coefficients
                - extract_fourier_slice_from_grid_points

---

::: cryojax.simulator.NufftProjection
        options:
            members:
                - __init__
                - integrate
                - project_voxel_cloud_with_nufft

## Integration methods for atom-based potentials

::: cryojax.simulator.GaussianMixtureProjection
        options:
            members:
                - __init__
                - integrate
