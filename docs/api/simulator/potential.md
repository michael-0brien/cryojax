# Scattering potential representations

`cryojax` provides different options for how to represent spatial potential energy distributions in cryo-EM.

???+ abstract "`cryojax.simulator.AbstractPotentialRepresentation`"
    ::: cryojax.simulator.AbstractPotentialRepresentation
        options:
            members:
                - rotate_to_pose

## Atom-based scattering potentials

???+ abstract "`cryojax.simulator.AbstractAtomicPotential`"

    ::: cryojax.simulator.AbstractAtomicPotential
        options:
            members:
                - atom_positions
                - as_real_voxel_grid

::: cryojax.simulator.GaussianMixtureAtomicPotential
        options:
            members:
                - __init__
                - rotate_to_pose
                - translate_to_pose
                - as_real_voxel_grid


::: cryojax.simulator.PengAtomicPotential
        options:
            members:
                - __init__
                - rotate_to_pose
                - translate_to_pose
                - as_real_voxel_grid


## Voxel-based scattering potentials

??? abstract "`cryojax.simulator.AbstractVoxelPotential`"

    ::: cryojax.simulator.AbstractVoxelPotential
        options:
            members:
                - voxel_size
                - is_real_space
                - shape
                - from_real_voxel_grid

### Fourier-space voxel representations

!!! info "Fourier-space conventions"
    - The `fourier_voxel_grid` and `frequency_slice` arguments to
    `FourierVoxelGridPotential.__init__` should be loaded with the zero frequency
    component in the center of the box.
    - The parameters in an `AbstractPose` represent a rotation in real-space. This means that when calling `FourierVoxelGridPotential.rotate_to_pose`,
    frequencies are rotated by the inverse rotation as stored in the pose.

::: cryojax.simulator.FourierVoxelGridPotential
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - rotate_to_pose
                - frequency_slice_in_angstroms
                - shape

---

::: cryojax.simulator.FourierVoxelSplinePotential
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - rotate_to_pose
                - frequency_slice_in_angstroms
                - shape


### Real-space voxel representations

::: cryojax.simulator.RealVoxelGridPotential
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - rotate_to_pose
                - coordinate_grid_in_angstroms
                - shape

---

::: cryojax.simulator.RealVoxelCloudPotential
        options:
            members:
                - __init__
                - from_real_voxel_grid
                - rotate_to_pose
                - coordinate_list_in_angstroms
                - shape
