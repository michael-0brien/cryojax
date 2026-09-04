"""Regenerate the cisTEM reference data used by `test_agree_with_cistem.py`.

`pycistem` is *not* a test dependency. Instead, the values cisTEM produces are
frozen into `tests/data/cistem_ctf.npz` and `tests/data/cistem_projection.npz`,
so that the tests run everywhere without it (its wheels are Linux-only and
capped at Python < 3.12).

Run this script to re-validate `cryojax` against cisTEM, for example after
changing a convention that these tests pin down:

    pip install 'pycistem==0.4.1'
    python tests/generate_cistem_references.py

It rewrites both `.npz` files in place. Commit the result together with the
change that motivated it.
"""

import os
import tempfile

import cryojax.simulator as cxs
import numpy as np
from cryojax.constants import PengScatteringFactorParameters
from cryojax.io import read_atoms_from_pdb, write_volume_to_mrc
from cryojax.ndimage import cartesian_to_polar, make_frequency_grid
from cryojax.ndimage._interpolation import deconvolve_interpolation_kernel


try:
    from pycistem.core import CTF as CistemCTF, AnglesAndShifts, Image  # pyright: ignore
except ModuleNotFoundError as err:  # pragma: no cover
    raise SystemExit(
        "`pycistem` is required to regenerate the reference data, but is not "
        "installed. Install it with `pip install 'pycistem==0.4.1'` (Linux only, "
        "Python < 3.12)."
    ) from err


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# CTF: `(defocus1, defocus2, astigmatism_angle, kV, cs, ac, pixel_size)`. The grid
# is small on purpose: cisTEM's CTF is evaluated pointwise, so a coarse grid still
# samples the full frequency range and the whole range of astigmatism angles.
CTF_GRID_SIZE = 128
CTF_PARAMETERS = [
    (12000.0, 12000.0, 0.0, 300.0, 2.7, 0.07, 1.0),
    (12000.0, 12000.0, 0.0, 200.0, 0.01, 0.12, 1.3),
    (1200.0, 1200.0, 0.0, 300.0, 2.7, 0.07, 1.5),
    (24000.0, 12000.0, 30.0, 300.0, 2.7, 0.07, 0.9),
    (24000.0, 24000.0, 0.0, 300.0, 2.7, 0.07, 2.0),
    (9000.0, 7000.0, 180.0, 300.0, 2.7, 0.07, 1.0),
    (12000.0, 9000.0, 0.0, 200.0, 2.7, 0.07, 0.9),
    (12000.0, 12000.0, 60.0, 200.0, 2.7, 0.02, 0.75),
    (12000.0, 3895.0, 45.0, 200.0, 2.7, 0.07, 2.2),
]

# Projection: this test pins down the pose (Euler angle) convention, so it spends
# its resolution budget on angular coverage rather than on a large box. Includes
# the `theta = 0` and `theta = 180` gimbal cases, negative angles, and wrap-around.
PROJECTION_BOX_SIZE = 64
PROJECTION_VOXEL_SIZE = 1.0
PROJECTION_ANGLES = [
    (0.0, 0.0, 0.0),
    (10.0, 90.0, 170.0),
    (10.0, 80.0, -20.0),
    (-1.2, 90.5, 67.0),
    (-50.0, 62.0, -21.0),
    (0.0, 180.0, 0.0),
    (45.0, 0.0, 45.0),
    (90.0, 45.0, -90.0),
    (-120.0, 135.0, 30.0),
    (180.0, 90.0, 180.0),
    (30.0, 0.5, -30.0),
    (-75.0, 179.5, 120.0),
]


def generate_ctf_reference():
    """Evaluate cisTEM's CTF for each parameter set on a fixed frequency grid."""
    reference = []
    for defocus1, defocus2, astig_angle, kV, cs, ac, pixel_size in CTF_PARAMETERS:
        frequency_grid = make_frequency_grid((CTF_GRID_SIZE, CTF_GRID_SIZE), pixel_size)
        k_sqr, theta = cartesian_to_polar(frequency_grid, square=True)
        cistem_ctf = CistemCTF(
            kV=kV,
            cs=cs,
            ac=ac,
            defocus1=defocus1,
            defocus2=defocus2,
            astig_angle=astig_angle,
            pixel_size=pixel_size,
        )
        evaluated = np.vectorize(lambda a, b: cistem_ctf.Evaluate(a, b))(
            np.asarray(k_sqr).ravel() * pixel_size**2, np.asarray(theta).ravel()
        )
        reference.append(evaluated.reshape(frequency_grid.shape[0:2]))

    return np.stack(reference).astype(np.float32)


def generate_projection_reference():
    """Project a rendered volume at each pose using cisTEM's slice extraction.

    The volume is returned alongside the projections. Freezing it means the test
    does not re-render from a PDB, so it isolates the slice extraction and the
    pose convention from the renderer, the PDB reader, and the scattering
    factor parameters.
    """
    box_size, voxel_size = PROJECTION_BOX_SIZE, PROJECTION_VOXEL_SIZE
    atom_positions, atomic_numbers = read_atoms_from_pdb(
        os.path.join(DATA_DIR, "1uao.pdb"), center=True
    )
    gaussian_volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(
        atom_positions,
        parameters=PengScatteringFactorParameters(atomic_numbers),
        extra_b_factors=10.0,
    )
    render_fn = cxs.GaussianMixtureRenderFn((box_size, box_size, box_size), voxel_size)
    real_voxel_grid = np.asarray(render_fn(gaussian_volume), dtype=np.float32)

    projections = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        # ... cisTEM reads from an MRC. The volume is deconvolved first to
        # compensate for the interpolation kernel that `cryojax` applies when it
        # extracts a fourier slice, so that the two are compared like for like.
        mrc_path = os.path.join(tmp_dir, "volume.mrc")
        write_volume_to_mrc(
            np.asarray(
                deconvolve_interpolation_kernel(real_voxel_grid, sinc_power=2),
                dtype=np.float32,
            ),
            voxel_size,
            mrc_path,
            overwrite=True,
        )
        cistem_volume = Image()
        cistem_volume.QuickAndDirtyReadSlices(mrc_path, 1, box_size)
        cistem_volume.ForwardFFT(True)
        cistem_volume.ZeroCentralPixel()
        cistem_volume.SwapRealSpaceQuadrants()

        for phi, theta, psi in PROJECTION_ANGLES:
            angles = AnglesAndShifts()
            angles.Init(phi, theta, psi, 0.0, 0.0)
            projection = Image()
            projection.Allocate(box_size, box_size, False)
            cistem_volume.ExtractSlice(projection, angles, 1.0, False)
            projection.PhaseShift(angles.ReturnShiftX(), angles.ReturnShiftY(), 0.0)
            projection.SwapRealSpaceQuadrants()
            projection.BackwardFFT()
            projections.append(np.asarray(projection.real_values).copy())

    return real_voxel_grid, np.stack(projections).astype(np.float32)


if __name__ == "__main__":
    ctf = generate_ctf_reference()
    ctf_path = os.path.join(DATA_DIR, "cistem_ctf.npz")
    np.savez_compressed(
        ctf_path,
        grid_size=np.asarray(CTF_GRID_SIZE),
        parameters=np.asarray(CTF_PARAMETERS, dtype=np.float64),
        ctf=ctf,
    )

    volume, projections = generate_projection_reference()
    projection_path = os.path.join(DATA_DIR, "cistem_projection.npz")
    np.savez_compressed(
        projection_path,
        box_size=np.asarray(PROJECTION_BOX_SIZE),
        voxel_size=np.asarray(PROJECTION_VOXEL_SIZE),
        angles=np.asarray(PROJECTION_ANGLES, dtype=np.float64),
        volume=volume,
        projections=projections,
    )

    for path in (ctf_path, projection_path):
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.3f} MB)")
