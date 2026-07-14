"""Compare `cryojax` against reference values computed with cisTEM.

The reference values are frozen in `tests/data/cistem_*.npz` rather than being
computed on the fly, so `pycistem` is not a test dependency (its wheels are
Linux-only and capped at Python < 3.12). Regenerate them with

    python tests/generate_cistem_references.py

See that script for what is stored and why.
"""

import os

import cryojax.simulator as cxs
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.constants import wavelength_from_kilovolts
from cryojax.ndimage import compute_binned_powerspectrum, make_frequency_grid
from cryojax.simulator import AstigmaticCTF, EulerAnglePose


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CTF_REFERENCE = np.load(os.path.join(_DATA_DIR, "cistem_ctf.npz"))
_PROJECTION_REFERENCE = np.load(os.path.join(_DATA_DIR, "cistem_projection.npz"))


@pytest.mark.parametrize("index", range(len(_CTF_REFERENCE["parameters"])))
def test_ctf_with_cistem(index):
    """Test CTF model against cisTEM.

    Modified from https://github.com/jojoelfe/contrasttransferfunction"""
    defocus1, defocus2, astig_angle, kV, cs, ac, pixel_size = _CTF_REFERENCE[
        "parameters"
    ][index]
    cistem_ctf = _CTF_REFERENCE["ctf"][index]
    grid_size = int(_CTF_REFERENCE["grid_size"])
    frequency_grid = make_frequency_grid((grid_size, grid_size), pixel_size)
    # Compute cryojax CTF
    optics = AstigmaticCTF(
        defocus_in_angstroms=(defocus1 + defocus2) / 2,
        astigmatism_in_angstroms=defocus1 - defocus2,
        astigmatism_angle=astig_angle,
        spherical_aberration_in_mm=cs,
    )
    ctf = jnp.asarray(
        optics(
            frequency_grid,
            wavelength_in_angstroms=wavelength_from_kilovolts(kV),
            amplitude_contrast_ratio=ac,
        )
    )
    # Compare the radially averaged power spectrum too, which is more sensitive
    # to a systematic bias than the pointwise comparison
    radial_frequency_grid = jnp.linalg.norm(frequency_grid, axis=-1)
    spectrum1D, _ = compute_binned_powerspectrum(
        ctf,
        radial_frequency_grid,
        pixel_size,
        maximum_frequency=1 / (2 * pixel_size),
    )
    cisTEM_spectrum1D, _ = compute_binned_powerspectrum(
        jnp.asarray(cistem_ctf),
        radial_frequency_grid,
        pixel_size,
        maximum_frequency=1 / (2 * pixel_size),
    )

    np.testing.assert_allclose(ctf, cistem_ctf, atol=1e-3)
    np.testing.assert_allclose(spectrum1D, cisTEM_spectrum1D, atol=1e-4)


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


@pytest.mark.parametrize("index", range(len(_PROJECTION_REFERENCE["angles"])))
def test_compute_projection_with_cistem(index):
    """Test fourier slice extraction, and the euler angle convention it is used
    with, against cisTEM.

    The volume is loaded from the reference data rather than rendered from a PDB,
    so that this isolates the projection and the pose convention from the
    renderer, the PDB reader, and the scattering factor parameters.
    """
    phi, theta, psi = _PROJECTION_REFERENCE["angles"][index]
    cistem_projection = _PROJECTION_REFERENCE["projections"][index]
    box_size = int(_PROJECTION_REFERENCE["box_size"])
    voxel_size = float(_PROJECTION_REFERENCE["voxel_size"])

    volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        jnp.asarray(_PROJECTION_REFERENCE["volume"])
    )
    pose = cxs.EulerAnglePose(phi_angle=-phi, theta_angle=-theta, psi_angle=-psi)
    projection_method = cxs.FourierSliceExtraction()
    image_config = cxs.BasicImageConfig((box_size, box_size), voxel_size, 300.0)
    projection = jnp.fft.irfftn(
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

    # A wrong euler angle convention gives a difference of order the projection's
    # peak amplitude (~6e-2), so this sits far below such an error while staying
    # comfortably above the numerical agreement with cisTEM (~1.3e-3).
    np.testing.assert_allclose(projection, cistem_projection, atol=5e-3)
