import cryojax.simulator as cxs
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.constants import PengScatteringFactorParameters, wavelength_from_kilovolts
from cryojax.io import read_atoms_from_pdb, write_volume_to_mrc
from cryojax.ndimage import (
    cartesian_to_polar,
    compute_binned_powerspectrum,
    irfftn,
    make_frequency_grid,
)
from cryojax.simulator import AstigmaticCTF, EulerAnglePose


try:
    from pycistem.core import CTF as cistemCTF, AnglesAndShifts, Image  # pyright: ignore
except ModuleNotFoundError:
    cistemCTF, AnglesAndShifts, Image = None, None, None

_PYCISTEM_SKIP_REASON = (
    "pycistem not installed (Linux-only wheels; "
    "install the version pinned in pyproject.toml to run these tests)"
)

_skip_without_pycistem = pytest.mark.skipif(
    AnglesAndShifts is None, reason=_PYCISTEM_SKIP_REASON
)


@_skip_without_pycistem
@pytest.mark.parametrize(
    "defocus1,defocus2,asti_angle,kV,cs,ac,pixel_size",
    [
        (12000.0, 12000.0, 0.0, 300.0, 2.7, 0.07, 1.0),
        (12000.0, 12000.0, 0.0, 200.0, 0.01, 0.12, 1.3),
        (1200.0, 1200.0, 0.0, 300.0, 2.7, 0.07, 1.5),
        (24000.0, 12000.0, 30.0, 300.0, 2.7, 0.07, 0.9),
        (24000.0, 24000.0, 0.0, 300.0, 2.7, 0.07, 2.0),
        (9000.0, 7000.0, 180.0, 300.0, 2.7, 0.07, 1.0),
        (12000.0, 9000.0, 0.0, 200.0, 2.7, 0.07, 0.9),
        (12000.0, 12000.0, 60.0, 200.0, 2.7, 0.02, 0.75),
        (12000.0, 3895.0, 45.0, 200.0, 2.7, 0.07, 2.2),
    ],
)
def test_ctf_with_cistem(defocus1, defocus2, asti_angle, kV, cs, ac, pixel_size):
    """Test CTF model against cisTEM.

    Modified from https://github.com/jojoelfe/contrasttransferfunction"""
    shape = (512, 512)
    freqs = make_frequency_grid(shape, pixel_size)
    k_sqr, theta = cartesian_to_polar(freqs, square=True)
    # Compute cryojax CTF
    optics = AstigmaticCTF(
        defocus_in_angstroms=(defocus1 + defocus2) / 2,
        astigmatism_in_angstroms=defocus1 - defocus2,
        astigmatism_angle=asti_angle,
        spherical_aberration_in_mm=cs,
    )
    ctf = jnp.array(
        optics(
            freqs,
            wavelength_in_angstroms=wavelength_from_kilovolts(kV),
            amplitude_contrast_ratio=ac,
        )
    )
    # Compute cisTEM CTF
    assert cistemCTF is not None
    cisTEM_optics = cistemCTF(
        kV=kV,
        cs=cs,
        ac=ac,
        defocus1=defocus1,
        defocus2=defocus2,
        astig_angle=asti_angle,
        pixel_size=pixel_size,
    )
    cisTEM_ctf = np.vectorize(lambda k_sqr, theta: cisTEM_optics.Evaluate(k_sqr, theta))(
        k_sqr.ravel() * pixel_size**2, theta.ravel()
    ).reshape(freqs.shape[0:2])
    # cisTEM_ctf[0, 0] = 0.0

    # Compute cryojax and cisTEM power spectrum
    radial_freqs = jnp.linalg.norm(freqs, axis=-1)
    spectrum1D, _ = compute_binned_powerspectrum(
        ctf, radial_freqs, pixel_size, maximum_frequency=1 / (2 * pixel_size)
    )
    cisTEM_spectrum1D, _ = compute_binned_powerspectrum(
        cisTEM_ctf, radial_freqs, pixel_size, maximum_frequency=1 / (2 * pixel_size)
    )

    np.testing.assert_allclose(ctf, cisTEM_ctf, atol=5e-2)
    np.testing.assert_allclose(spectrum1D, cisTEM_spectrum1D, atol=5e-3)


@pytest.mark.parametrize(
    "phi, theta, psi",
    [(10.0, 90.0, 170.0), (10.0, 80.0, -20.0), (-1.2, 90.5, 67.0), (-50.0, 62.0, -21.0)],
)
def test_euler_matrix_with_cistem(phi, theta, psi):
    """Test zyz rotation matrix"""
    # Hard code zyz rotation matrix from cisTEM convention
    phi_in_rad, theta_in_rad, psi_in_rad = [
        np.deg2rad(angle) for angle in [phi, theta, psi]
    ]
    matrix = np.zeros((3, 3))
    cos_phi = np.cos(phi_in_rad)
    sin_phi = np.sin(phi_in_rad)
    cos_theta = np.cos(theta_in_rad)
    sin_theta = np.sin(theta_in_rad)
    cos_psi = np.cos(psi_in_rad)
    sin_psi = np.sin(psi_in_rad)
    matrix[0, 0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi
    matrix[0, 1] = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi
    matrix[0, 2] = -sin_theta * cos_psi
    matrix[1, 0] = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi
    matrix[1, 1] = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi
    matrix[1, 2] = sin_theta * sin_psi
    matrix[2, 0] = sin_theta * cos_phi
    matrix[2, 1] = sin_theta * sin_phi
    matrix[2, 2] = cos_theta
    # Generate rotation that matches this rotation matrix
    pose = EulerAnglePose(phi_angle=-phi, theta_angle=-theta, psi_angle=-psi)
    np.testing.assert_allclose(pose.rotation.as_matrix(), matrix, atol=1e-12)


@_skip_without_pycistem
@pytest.mark.parametrize(
    "phi, theta, psi, box_size, voxel_size",
    [
        # multiple Euler angles at one grid size
        (0.0, 0.0, 0.0, 128, 0.5),
        (10.0, 90.0, 170.0, 128, 0.5),
        (10.0, 80.0, -20.0, 128, 0.5),
        (-1.2, 90.5, 67.0, 128, 0.5),
        (-50.0, 62.0, -21.0, 128, 0.5),
        # same angle, different grid sizes rendered on-the-fly
        # (10.0, 90.0, 170.0, 48, ),
        # (10.0, 90.0, 170.0, 32, 1.5),
    ],
)
def test_compute_projection_with_cistem(
    phi,
    theta,
    psi,
    box_size,
    voxel_size,
    tmp_path,
    sample_pdb_path,
):
    # Render volume on-the-fly from atom positions at the requested grid shape
    atom_positions, atomic_numbers = read_atoms_from_pdb(sample_pdb_path, center=True)
    gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atomic_numbers),
        extra_b_factors=10.0,
    )
    render_fn = cxs.GaussianMixtureRenderFn((box_size, box_size, box_size), voxel_size)
    real_voxel_grid = render_fn(gaussian_volume)

    # cryojax projection
    volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
    pose = cxs.EulerAnglePose(phi_angle=-phi, theta_angle=-theta, psi_angle=-psi)
    projection_method = cxs.FourierSliceExtraction()
    image_config = cxs.BasicImageConfig((box_size, box_size), voxel_size, 300.0)

    def project_impl(volume, pose, image_config):
        return irfftn(
            (
                projection_method.integrate(
                    volume.rotate_to_pose(pose), image_config, outputs_real_space=False
                )
                / image_config.pixel_size
            )
            .at[0, 0]
            .set(0.0 + 0.0j)
            / np.sqrt(np.prod(image_config.shape)),
            s=image_config.padded_shape,
        )

    cryojax_projection = project_impl(volume, pose, image_config)

    # pycistem projection — write rendered grid to a temp MRC for loading
    mrc_path = str(tmp_path / "volume.mrc")
    write_volume_to_mrc(
        np.array(real_voxel_grid, dtype=np.float32),
        voxel_size,
        mrc_path,
        overwrite=True,
    )
    pycistem_volume = _load_pycistem_template(mrc_path, box_size)
    assert AnglesAndShifts is not None
    pycistem_angles = AnglesAndShifts()
    pycistem_angles.Init(phi, theta, psi, 0.0, 0.0)
    pycistem_model = _compute_projection(pycistem_volume, pycistem_angles, box_size)
    pycistem_projection = np.asarray(pycistem_model.real_values)

    # from matplotlib import pyplot as plt
    # fig, axes = plt.subplots(ncols=2)
    # im1 = axes[0].imshow(cryojax_projection)
    # im2 = axes[1].imshow(pycistem_projection)
    # fig.savefig("temp.png")

    np.testing.assert_allclose(cryojax_projection, pycistem_projection, atol=1e-3)


def _load_pycistem_template(filename, box_size):
    """Load pycistem template in fourier space."""
    volume = Image()  # type: ignore
    volume.QuickAndDirtyReadSlices(filename, 1, box_size)
    volume.ForwardFFT(True)
    volume.ZeroCentralPixel()
    volume.SwapRealSpaceQuadrants()

    return volume


def _compute_projection(volume, angles, box_size):
    """Compute scattering projection of pycistem volume."""
    projection = Image()  # type: ignore
    projection.Allocate(box_size, box_size, False)

    volume.ExtractSlice(projection, angles, 1.0, False)
    projection.PhaseShift(angles.ReturnShiftX(), angles.ReturnShiftY(), 0.0)

    projection.SwapRealSpaceQuadrants()
    projection.BackwardFFT()

    return projection
