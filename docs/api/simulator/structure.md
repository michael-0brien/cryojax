# Modeling cryo-EM structures

There are many different data representations of biological structures for cryo-EM, including atomic models, voxel maps, and neural network representations. The optimal representation to use depends on the user's needs. Therefore, CryoJAX supports a variety of structure representations as well as a modeling interface for creating new representations downstream. More generally, cryoJAX implements *functions* of structures via the core `AbstractStructureParameterisation` class. This page discusses how to use this interface and documents the structures included in the library.

## Core base classes

???+ abstract "`cryojax.simulator.AbstractStructureParameterisation`"
    ::: cryojax.simulator.AbstractStructureParameterisation
        options:
            members:
                - evaluate


???+ abstract "`cryojax.simulator.AbstractStructureRepresentation`"
    ::: cryojax.simulator.AbstractStructureRepresentation
        options:
            members:
                - rotate_to_pose

## Structure representations

### Point clouds

??? abstract "`cryojax.simulator.AbstractPointCloudStructure`"

    ::: cryojax.simulator.AbstractPointCloudStructure
        options:
            members:
                - translate_to_pose

::: cryojax.simulator.GaussianMixtureStructure
    options:
        members:
            - __init__
            - evaluate
            - rotate_to_pose
            - translate_to_pose

#### Atom-based

??? abstract "`cryojax.simulator.AbstractIndependentAtomStructure`"

    ::: cryojax.simulator.AbstractIndependentAtomStructure
        options:
            members:
                - atom_positions

??? abstract "`cryojax.simulator.AbstractTabulatedScatteringPotential`"

    ::: cryojax.simulator.AbstractTabulatedScatteringPotential
        options:
            members:
                - from_scattering_factor_parameters

::: cryojax.simulator.PengIndependentAtomPotential
        options:
            members:
                - __init__
                - from_scattering_factor_parameters
                - evaluate
                - rotate_to_pose
                - translate_to_pose
                - to_real_voxel_grid

---

::: cryojax.simulator.PengScatteringFactorParameters
        options:
            members:
                - __init__


### Voxel-based

??? abstract "`cryojax.simulator.AbstractVoxelStructure`"

    ::: cryojax.simulator.AbstractVoxelStructure
        options:
            members:
                - rotate_to_pose
                - shape
                - from_real_voxel_grid


#### Fourier-space

!!! info "Fourier-space conventions"
    - The `fourier_voxel_grid` and `frequency_slice` arguments to
    `FourierVoxelGridStructure.__init__` should be loaded with the zero frequency
    component in the center of the box. This is returned by the
    - The parameters in an `AbstractPose` represent a rotation in real-space. This means that when calling `FourierVoxelGridStructure.rotate_to_pose`,
    frequencies are rotated by the inverse rotation as stored in the pose.

??? abstract "`cryojax.simulator.AbstractFourierVoxelStructure`"

    ::: cryojax.simulator.AbstractFourierVoxelStructure
        options:
            members:
                - from_real_voxel_grid
                - frequency_slice_in_pixels
                - shape

::: cryojax.simulator.FourierVoxelGridStructure
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - evaluate
                - rotate_to_pose
                - frequency_slice_in_pixels
                - shape

---

::: cryojax.simulator.FourierVoxelSplineStructure
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - evaluate
                - rotate_to_pose
                - frequency_slice_in_pixels
                - shape


#### Real-space

??? abstract "`cryojax.simulator.AbstractRealVoxelStructure`"

    ::: cryojax.simulator.AbstractRealVoxelStructure
        options:
            members:
                - from_real_voxel_grid
                - coordinate_grid_in_pixels
                - shape


::: cryojax.simulator.RealVoxelGridStructure
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - evaluate
                - rotate_to_pose
                - coordinate_grid_in_pixels
                - shape


## Mappings to structures

??? abstract "`cryojax.simulator.AbstractStructuralEnsemble`"

    ::: cryojax.simulator.AbstractStructuralEnsemble
        options:
            members:
                - conformation
                - evaluate

::: cryojax.simulator.DiscreteStructuralEnsemble
        options:
            members:
                - __init__
                - evaluate
