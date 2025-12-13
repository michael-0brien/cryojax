# Volume projection and integration onto the plane

`cryojax` provides different methods for integrating [volumes](./volume.md#volume-representations) onto a plane to generate an image.

???+ abstract "`cryojax.simulator.AbstractVolumeIntegrator`"
    ::: cryojax.simulator.AbstractVolumeIntegrator
        options:
            members:
                - integrate


::: cryojax.simulator.AutoVolumeProjection
        options:
            members:
                - __init__
                - integrate

## Integration methods for voxel-based structures

::: cryojax.simulator.FourierSliceExtraction
        options:
            members:
                - __init__
                - integrate

---

::: cryojax.simulator.RealVoxelProjection
        options:
            members:
                - __init__
                - integrate

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
