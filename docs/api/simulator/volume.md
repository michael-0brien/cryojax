# Modeling cryo-EM volumes

There are many different volume representations of biological structures for cryo-EM, including atomic models, voxel maps, and neural network representations. Further, there are many ways to generate these volumes, such as from protein generative modeling and molecular dynamics. The optimal implementation to use depends on the user's needs. Therefore, CryoJAX supports a variety of these representations as well as a modeling interface for usage downstream. This page discusses how to use this interface and documents the volumes included in the library.

## Core base classes

???+ abstract "`cryojax.simulator.AbstractVolumeParametrisation`"
    ::: cryojax.simulator.AbstractVolumeParametrisation
        options:
            members:
                - compute_representation


???+ abstract "`cryojax.simulator.AbstractVolumeRepresentation`"
    ::: cryojax.simulator.AbstractVolumeRepresentation
        options:
            members:
                - rotate_to_pose

## Volume representations

### Point clouds

??? abstract "`cryojax.simulator.AbstractPointCloudVolume`"
    ::: cryojax.simulator.AbstractPointCloudVolume
        options:
            members:
                - translate_to_pose

::: cryojax.simulator.GaussianMixtureVolume
    options:
        members:
            - __init__
            - compute_representation
            - rotate_to_pose
            - translate_to_pose
            - to_real_voxel_grid

#### Atomic models

??? abstract "`cryojax.simulator.AbstractTabulatedAtomicVolume`"
    ::: cryojax.simulator.AbstractTabulatedAtomicVolume
        options:
            members:
                - atom_positions
                - from_tabulated_parameters

::: cryojax.simulator.PengAtomicVolume
    options:
        members:
            - __init__
            - from_tabulated_parameters
            - compute_representation
            - rotate_to_pose
            - translate_to_pose
            - to_real_voxel_grid

### Voxel-based

#### Fourier-space

!!! info "Fourier-space conventions"
    - The `fourier_voxel_grid` and `frequency_slice` arguments to
    `FourierVoxelGridVolume.__init__` should be loaded with the zero frequency
    component in the center of the box. This is returned by the
    - The parameters in an `AbstractPose` represent a rotation in real-space. This means that when calling `FourierVoxelGridVolume.rotate_to_pose`,
    frequencies are rotated by the inverse rotation as stored in the pose.

::: cryojax.simulator.FourierVoxelGridVolume
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - compute_representation
                - rotate_to_pose
                - frequency_slice_in_pixels
                - shape

---

::: cryojax.simulator.FourierVoxelSplineVolume
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - compute_representation
                - rotate_to_pose
                - frequency_slice_in_pixels
                - shape


#### Real-space

::: cryojax.simulator.RealVoxelGridVolume
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - compute_representation
                - rotate_to_pose
                - coordinate_grid_in_pixels
                - shape
