import equinox as eqx

import cryojax.simulator as cxs
from cryojax.io import read_array_from_mrc


@profile  # noqa: F821
def main():
    # Scattering potential stored in MRC format
    filename = "./data/groel_5w0s_scattering_potential.mrc"
    # ... read into a FourierVoxelGridPotential
    real_voxel_grid, voxel_size = read_array_from_mrc(filename, loads_grid_spacing=True)
    volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=2)
    # Now, instantiate the pose. Angles are given in degrees
    pose = cxs.EulerAnglePose(
        offset_x_in_angstroms=5.0,
        offset_y_in_angstroms=-3.0,
        phi_angle=20.0,
        theta_angle=80.0,
        psi_angle=-5.0,
    )
    ctf = cxs.AstigmaticCTF(
        defocus_in_angstroms=10000.0,
        astigmatism_in_angstroms=-100.0,
        astigmatism_angle=10.0,
    )
    transfer_theory = cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.1)
    # Then the configuration. Add padding with respect to the final image shape.
    pad_options = dict(shape=volume.shape[0:2])
    image_config = cxs.BasicImageConfig(
        shape=(80, 80),
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
        pad_options=pad_options,
    )
    image_model = cxs.make_image_model(
        volume,
        image_config,
        pose,
        transfer_theory,
        normalizes_signal=True,
    )

    @eqx.filter_jit
    def simulate_fn(image_model):
        return image_model.simulate()

    simulate_fn(image_model)


if __name__ == "__main__":
    main()
