import numpy as np
import pytest

import cryojax.distributions as dist
import cryojax.simulator as cxs


@pytest.mark.parametrize(
    "cls, structure, scattering_theory, instrument_config",
    [
        (dist.IndependentGaussianPixels, "specimen", "theory", "config"),
        (dist.IndependentGaussianFourierModes, "specimen", "theory", "config"),
    ],
)
def test_simulate_signal_from_gaussian_distributions(
    cls, structure, scattering_theory, instrument_config, request
):
    structure = request.getfixturevalue(structure)
    scattering_theory = request.getfixturevalue(scattering_theory)
    instrument_config = request.getfixturevalue(instrument_config)
    image_model = cxs.ContrastImageModel(structure, instrument_config, scattering_theory)
    distribution = cls(image_model, normalizes_signal=False)
    np.testing.assert_allclose(image_model.render(), distribution.compute_signal())
