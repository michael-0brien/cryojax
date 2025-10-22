# Volume integration methods

`cryojax` provides different methods for integrating [volumes](./volume.md#volume-representations) onto a plane to generate an image.

???+ abstract "`cryojax.simulator.AbstractVolumeIntegrator`"
    ::: cryojax.simulator.AbstractVolumeIntegrator
        options:
            members:
                - integrate

## Integration methods for voxel-based structures

::: cryojax.simulator.FourierSliceExtraction
        options:
            members:
                - __init__
                - integrate
                - extract_fourier_slice_from_spline
                - extract_fourier_slice_from_grid

---

::: cryojax.simulator.RealVoxelProjection
        options:
            members:
                - __init__
                - integrate
                - project_voxel_cloud_with_nufft

## Integration methods for atom-based based structures

::: cryojax.simulator.GaussianMixtureProjection
        options:
            members:
                - __init__
                - integrate

---

::: cryojax.simulator.FFTAtomProjection
        options:
            members:
                - __init__
                - integrate
