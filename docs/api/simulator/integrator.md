# Volume integration methods

`cryojax` provides different methods for integrating [volumes](./volume.md#volume-representations) onto a plane to generate an image.

???+ abstract "`cryojax.simulator.AbstractVolumeIntegrator`"
    ::: cryojax.simulator.AbstractVolumeIntegrator
        options:
            members:
                - integrate

## Integration methods for voxel-based structures

???+ abstract "`cryojax.simulator.AbstractVoxelVolumeIntegrator`"
    ::: cryojax.simulator.AbstractVoxelVolumeIntegrator
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
