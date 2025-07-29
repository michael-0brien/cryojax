# Structure integration methods

`cryojax` provides different methods for integrating [structures](./structure.md#structure-representations) onto a plane to generate an image.

???+ abstract "`cryojax.simulator.AbstractDirectIntegrator`"
    ::: cryojax.simulator.AbstractDirectIntegrator
        options:
            members:
                - integrate

## Integration methods for voxel-based structures

???+ abstract "`cryojax.simulator.AbstractDirectVoxelIntegrator`"
    ::: cryojax.simulator.AbstractDirectIntegrator
        options:
            members:
                - outputs_integral

::: cryojax.simulator.FourierSliceExtraction
        options:
            members:
                - __init__
                - integrate
                - extract_fourier_slice_from_spline
                - extract_fourier_slice_from_grid

---

::: cryojax.simulator.NufftProjection
        options:
            members:
                - __init__
                - integrate
                - project_voxel_cloud_with_nufft

## Integration methods for point-cloud based structures

::: cryojax.simulator.GaussianMixtureProjection
        options:
            members:
                - __init__
                - integrate
