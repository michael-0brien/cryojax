# Modeling cryo-EM structures

There are many different data representations of biological structures for cryo-EM, including atomic models, voxel maps, and neural network representations. Further, there are many ways to generate these structures, such as from protein generative modeling and molecular dynamics, and there are also different ways of parametrising an electrostatic potential once a structure is generated. The optimal implementation to use depends on the user's needs. Therefore, CryoJAX supports a variety of these structure representations as well as a modeling interface for creating new representations downstream. This page discusses how to use this interface and documents the structures included in the library.

## Core base classes

???+ abstract "`cryojax.simulator.AbstractStructureParameterisation`"
    ::: cryojax.simulator.AbstractStructureParameterisation
        options:
            members:
                - to_volume_parametrisation


???+ abstract "`cryojax.simulator.AbstractVolumeParametrisation`"
    ::: cryojax.simulator.AbstractVolumeParametrisation
        options:
            members:
                - rotate_to_pose

???+ abstract "`cryojax.simulator.AbstractEnsembleParametrisation`"
    ::: cryojax.simulator.AbstractEnsembleParametrisation
        options:
            members:
                - rotate_to_pose

???+ abstract "`cryojax.simulator.AbstractPotentialParametrisation`"
    ::: cryojax.simulator.AbstractPotentialParametrisation
        options:
            members:

## Volume parametrisations

### Point clouds

::: cryojax.simulator.GaussianMixtureVolume
    options:
        members:
            - __init__
            - to_volume_parametrisation
            - rotate_to_pose
            - translate_to_pose
            - to_real_voxel_grid

---

::: cryojax.simulator.PengIndependentAtomVolume
    options:
        members:
            - __init__
            - from_scattering_factor_parameters
            - to_volume_parametrisation
            - rotate_to_pose
            - translate_to_pose
            - to_real_voxel_grid

---

::: cryojax.simulator.PengScatteringFactorParameters
        options:
            members:
                - __init__

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


## Ensemble parametrisations

::: cryojax.simulator.DiscreteStructuralEnsemble
        options:
            members:
                - __init__
                - to_volume_parametrisation
