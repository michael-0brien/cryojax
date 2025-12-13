import cryojax.simulator as cxs
import jax.random as jr
import numpy as np
import pytest
from cryojax.io import read_array_from_mrc


@pytest.fixture
def volume_and_pixel_size(sample_mrc_path):
    real_voxel_grid, voxel_size = read_array_from_mrc(
        sample_mrc_path, loads_grid_spacing=True
    )
    return (
        cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid, pad_scale=1.3),
        voxel_size,
    )


@pytest.fixture
def volume(volume_and_pixel_size):
    return volume_and_pixel_size[0]


@pytest.fixture
def basic_config(volume_and_pixel_size):
    volume, pixel_size = volume_and_pixel_size
    return cxs.BasicImageConfig(
        shape=volume.shape[0:2],
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )


@pytest.fixture
def dose_config(volume_and_pixel_size):
    volume, pixel_size = volume_and_pixel_size
    return cxs.DoseImageConfig(
        shape=volume.shape[0:2],
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
        electron_dose=100.0,
    )


@pytest.fixture
def image_model(volume, basic_config):
    image_model = cxs.make_image_model(
        volume,
        basic_config,
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF()),
    )
    return image_model


@pytest.fixture
def detector_image_model(volume, dose_config):
    detector = cxs.PoissonDetector(cxs.NullDQE())
    scattering_theory = cxs.WeakPhaseScatteringTheory(
        volume_integrator=cxs.FourierSliceExtraction(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF()),
    )
    image_model = cxs.ElectronCountsImageModel(
        volume,
        image_config=dose_config,
        pose=cxs.EulerAnglePose(),
        scattering_theory=scattering_theory,
        detector=detector,
        normalizes_signal=True,
    )
    return image_model


@pytest.fixture
def solvent_image_model(volume, basic_config):
    import cryojax.experimental as cxe

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        volume_integrator=cxs.FourierSliceExtraction(),
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF()),
        solvent=cxe.GRFSolvent2D(100.0),
    )
    image_model = cxs.ContrastImageModel(
        volume,
        image_config=basic_config,
        pose=cxs.EulerAnglePose(),
        scattering_theory=scattering_theory,
        normalizes_signal=True,
    )
    return image_model


@pytest.mark.parametrize(
    "cls, model",
    [
        (cxs.GaussianWhiteNoiseModel, "image_model"),
        (cxs.GaussianColoredNoiseModel, "image_model"),
    ],
)
def test_simulate_equals_compute_signal(cls, model, request):
    model = request.getfixturevalue(model)
    noise_model = cls(model)
    np.testing.assert_allclose(model.simulate(), noise_model.compute_signal())


@pytest.mark.parametrize("variance, dim", ((1.0, 150), (2.0, 150), (4.0, 150)))
def test_variance_correct(variance, dim):
    rng_seed = 123
    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(
        np.zeros((dim, dim, dim))
    )
    image_config = cxs.BasicImageConfig(
        (dim, dim), pixel_size=1.0, voltage_in_kilovolts=300.0
    )
    image_model = cxs.make_image_model(
        voxel_volume, image_config, pose=cxs.EulerAnglePose()
    )
    rng_key = jr.key(rng_seed)
    noise_model = cxs.GaussianWhiteNoiseModel(image_model, variance=variance)
    np.testing.assert_allclose(
        noise_model.compute_noise(rng_key).std(), np.sqrt(variance), rtol=1e-2
    )


def test_detector_incompatible(detector_image_model):
    with pytest.raises(ValueError):
        _ = cxs.GaussianWhiteNoiseModel(detector_image_model)


def test_solvent2d_incompatible(solvent_image_model):
    with pytest.raises(ValueError):
        _ = cxs.GaussianWhiteNoiseModel(solvent_image_model)
