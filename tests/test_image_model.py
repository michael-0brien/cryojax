import cryojax.simulator as cxs
import equinox as eqx
import numpy as np
import pytest
from cryojax.io import read_array_from_mrc, read_atoms_from_pdb
from cryojax.ndimage import (
    CircularCosineMask,
    ImageScaling,
    LowpassFilter,
    crop_to_shape,
    pad_to_shape,
)


@pytest.fixture
def pdb_info(sample_pdb_path):
    return read_atoms_from_pdb(sample_pdb_path, center=True, loads_properties=True)


@pytest.fixture
def voxel_info(sample_mrc_path):
    return read_array_from_mrc(sample_mrc_path, loads_grid_spacing=True)


@pytest.fixture
def voxel_volume(voxel_info):
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_info[0], pad_scale=1.3)


@pytest.fixture
def voxel_size(voxel_info):
    return voxel_info[1]


@pytest.fixture
def basic_config(voxel_volume, voxel_size):
    shape = voxel_volume.shape[0:2]
    return cxs.BasicImageConfig(
        shape=(int(0.9 * shape[0]), int(0.9 * shape[1])),
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
        pad_options=dict(shape=shape),
    )


@pytest.fixture
def image_model(voxel_volume, basic_config):
    return cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF()),
    )


# Test correct image shape
@pytest.mark.parametrize("model", ["image_model"])
def test_real_shape(model, request):
    """Make sure shapes are as expected in real space."""
    model = request.getfixturevalue(model)
    image = model.simulate(outputs_real_space=True)
    padded_image = model.raw_simulate(outputs_real_space=True)
    assert image.shape == model.image_config.shape
    assert padded_image.shape == model.image_config.padded_shape


@pytest.mark.parametrize("model", ["image_model"])
def test_fourier_shape(model, request):
    """Make sure shapes are as expected in fourier space."""
    model = request.getfixturevalue(model)
    image = model.simulate(outputs_real_space=False)
    padded_image = model.raw_simulate(outputs_real_space=False)
    assert image.shape == model.image_config.frequency_grid_in_pixels.shape[0:2]
    assert (
        padded_image.shape
        == model.image_config.padded_frequency_grid_in_pixels.shape[0:2]
    )


@pytest.mark.parametrize("extra_dim_y, extra_dim_x", [(1, 1), (1, 0), (0, 1)])
def test_even_vs_odd_image_shape(extra_dim_y, extra_dim_x, voxel_volume, voxel_size):
    control_shape = voxel_volume.shape[0:2]
    test_shape = (control_shape[0] + extra_dim_y, control_shape[1] + extra_dim_x)
    config_control = cxs.BasicImageConfig(
        control_shape, pixel_size=voxel_size, voltage_in_kilovolts=300.0
    )
    config_test = cxs.BasicImageConfig(
        test_shape, pixel_size=voxel_size, voltage_in_kilovolts=300.0
    )
    pose = cxs.EulerAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(cxs.AstigmaticCTF())
    model_control = cxs.make_image_model(
        voxel_volume, config_control, pose=pose, transfer_theory=transfer_theory
    )
    model_test = cxs.make_image_model(
        voxel_volume, config_test, pose=pose, transfer_theory=transfer_theory
    )
    np.testing.assert_allclose(
        crop_to_shape(model_test.simulate(), control_shape),
        model_control.simulate(),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "offset_xy, pixel_size, shape, pad_scale",
    (
        ((1, 1), 1.0, (31, 31), 2),
        ((-1, -1), 1.0, (32, 32), 2),
        ((1, -1), 1.0, (31, 32), 2),
        ((-1, 1), 1.0, (32, 31), 2),
    ),
)
def test_translate_mode(pdb_info, offset_xy, pixel_size, shape, pad_scale):
    atom_pos, _, _ = pdb_info
    image_config = cxs.BasicImageConfig(
        shape,
        pixel_size,
        voltage_in_kilovolts=300.0,
        pad_options=dict(shape=(pad_scale * shape[0], pad_scale * shape[1])),
    )
    gaussian_width = 2 * pixel_size
    volume, integrator = (
        cxs.GaussianMixtureVolume(atom_pos, amplitudes=1.0, variances=gaussian_width**2),
        cxs.GaussianMixtureProjection(),
    )
    pose = cxs.EulerAnglePose.from_translation(np.asarray(offset_xy))
    # Projections
    fft_proj_model = cxs.ProjectionImageModel(
        volume,
        pose,
        image_config,
        integrator,
        translate_mode="fft",
    )
    atom_proj_model = cxs.ProjectionImageModel(
        volume,
        pose,
        image_config,
        integrator,
        translate_mode="atom",
    )
    atom_translate_proj = compute_image(atom_proj_model)
    fft_translate_proj = compute_image(fft_proj_model)

    np.testing.assert_allclose(atom_translate_proj, fft_translate_proj, atol=1e-8)
    # Images
    transfer_theory = cxs.ContrastTransferTheory(cxs.AstigmaticCTF())
    fft_im_model = cxs.LinearImageModel(
        volume,
        pose,
        image_config,
        transfer_theory,
        integrator,
        translate_mode="fft",
    )
    atom_im_model = cxs.LinearImageModel(
        volume,
        pose,
        image_config,
        transfer_theory,
        integrator,
        translate_mode="atom",
    )
    atom_translate_im = compute_image(atom_im_model)
    fft_translate_im = compute_image(fft_im_model)

    np.testing.assert_allclose(atom_translate_im, fft_translate_im, atol=1e-8)


def test_bad_translate_mode(voxel_info, basic_config):
    real_voxels, _ = voxel_info
    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxels)
    with pytest.raises(ValueError):
        model = cxs.make_image_model(
            voxel_volume, basic_config, pose=cxs.EulerAnglePose(), translate_mode="atom"
        )
        _ = model.simulate()


@pytest.mark.parametrize(
    "std, signal_centering",
    (
        (2.0, "mean"),
        (4.0, "mean"),
        (2.0, "bg"),
        (4.0, "bg"),
    ),
)
def test_normalize_and_transform(std, signal_centering, voxel_info, basic_config):
    real_voxels, _ = voxel_info
    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxels)
    image_model = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transform=ImageScaling(scale=std),
        normalizes_signal=True,
        signal_centering="bg",
    )
    image = compute_image(image_model)
    np.testing.assert_approx_equal(np.std(image), std)


@pytest.mark.parametrize("use_transform", (True, False))
def test_mask_zeros_edges(use_transform, voxel_info, basic_config):
    real_voxels, _ = voxel_info
    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxels)
    image_model = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transform=(ImageScaling(scale=1.0) if use_transform else None),
        normalizes_signal=True,
    )
    image = image_model.simulate(
        mask=CircularCosineMask(
            basic_config.coordinate_grid_in_pixels, radius=10, rolloff_width=0
        )
    )
    ny, nx = image.shape
    np.testing.assert_allclose(image[0, 0], 0.0)
    np.testing.assert_allclose(image[ny, 0], 0.0)
    np.testing.assert_allclose(image[0, nx], 0.0)
    np.testing.assert_allclose(image[ny, nx], 0.0)


def test_filter_padded_shape(voxel_info, basic_config):
    real_voxels, _ = voxel_info
    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxels)
    image_model = cxs.make_image_model(
        voxel_volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
    )
    _ = image_model.simulate(
        filter=LowpassFilter(
            basic_config.padded_frequency_grid_in_pixels,
            grid_spacing=basic_config.pixel_size,
        )
    )
    with pytest.raises(ValueError):
        _ = image_model.simulate(
            filter=LowpassFilter(
                basic_config.frequency_grid_in_pixels,
                grid_spacing=basic_config.pixel_size,
            )
        )


def test_bg_subtract(voxel_info):
    real_voxels, voxel_size = voxel_info
    dim = real_voxels.shape[0]
    padded_dim = 2 * dim
    real_voxels = pad_to_shape(real_voxels, tuple(3 * [padded_dim]))
    image_config = cxs.BasicImageConfig(
        shape=(padded_dim, padded_dim),
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
    )
    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxels)
    transfer_theory = cxs.ContrastTransferTheory(cxs.AstigmaticCTF())
    image_model = cxs.make_image_model(
        voxel_volume,
        image_config,
        pose=cxs.EulerAnglePose(),
        transfer_theory=transfer_theory,
        normalizes_signal=True,
        signal_centering="bg",
        quantity_mode="intensity",
    )
    image = compute_image(image_model)
    np.testing.assert_allclose(image[:, 0], 0.0, atol=5e-2)


@eqx.filter_jit
def compute_image(image_model):
    return image_model.simulate()
