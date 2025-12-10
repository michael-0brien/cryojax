# Modeling cryo-EM volumes

There are many different volume representations of biological structures for cryo-EM, including atomic models, voxel maps, and neural network representations. Further, there are many ways to generate these volumes, such as from protein generative modeling and molecular dynamics. The optimal implementation to use depends on the user's needs. Therefore, CryoJAX supports a variety of these representations as well as a modeling interface for usage downstream. This page discusses how to use this interface and documents the volumes included in the library.

## Core base classes

???+ abstract "`cryojax.simulator.AbstractVolumeParametrization`"
    ::: cryojax.simulator.AbstractVolumeParametrization
        options:
            members:
                - to_representation


???+ abstract "`cryojax.simulator.AbstractVolumeRepresentation`"
    ::: cryojax.simulator.AbstractVolumeRepresentation
        options:
            members:
                - rotate_to_pose

## Volume representations

### Atom-based volumes

??? abstract "`cryojax.simulator.AbstractAtomVolume`"
    ::: cryojax.simulator.AbstractAtomVolume
        options:
            members:
                - translate_to_pose

::: cryojax.simulator.GaussianMixtureVolume
    options:
        members:
            - __init__
            - from_tabulated_parameters
            - to_representation
            - rotate_to_pose
            - translate_to_pose

---

::: cryojax.simulator.IndependentAtomVolume
    options:
        members:
            - __init__
            - from_tabulated_parameters
            - to_representation
            - rotate_to_pose
            - translate_to_pose

### Voxel-based volumes

#### Fourier-space

???+ abstract "`cryojax.simulator.AbstractVoxelVolume`"
    ::: cryojax.simulator.AbstractVoxelVolume
        options:
            members:
                - from_real_voxel_grid
                - shape

!!! info "Fourier-space conventions"
    The `fourier_voxel_grid` and `frequency_slice` arguments to
    `FourierVoxelGridVolume.__init__` should be loaded with the zero frequency
    component in the center of the box.

::: cryojax.simulator.FourierVoxelGridVolume
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - to_representation
                - rotate_to_pose
                - frequency_slice_in_pixels
                - shape

---

::: cryojax.simulator.FourierVoxelSplineVolume
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - to_representation
                - rotate_to_pose
                - frequency_slice_in_pixels
                - shape


#### Real-space

::: cryojax.simulator.RealVoxelGridVolume
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - to_representation
                - rotate_to_pose
                - coordinate_grid_in_pixels
                - shape

## Volume rendering

???+ abstract "`cryojax.simulator.AbstractVolumeRenderFn`"
    ::: cryojax.simulator.AbstractVolumeRenderFn
        options:
            members:
                - shape
                - voxel_size
                - __call__

---

::: cryojax.simulator.AutoVolumeRenderFn
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.simulator.GaussianMixtureRenderFn
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.simulator.FFTAtomRenderFn
        options:
            members:
                - __init__
                - __call__
