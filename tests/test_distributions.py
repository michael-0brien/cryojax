import numpy as np
import pytest

import cryojax.distributions as dist
import cryojax.simulator as cxs


@pytest.mark.parametrize(
    "cls, scattering_theory, instrument_config",
    [
        (dist.IndependentGaussianPixels, "theory", "config"),
        (dist.IndependentGaussianFourierModes, "theory", "config"),
    ],
)
def test_simulate_signal_from_gaussian_distributions(
    cls, scattering_theory, instrument_config, request
):
    scattering_theory = request.getfixturevalue(scattering_theory)
    instrument_config = request.getfixturevalue(instrument_config)
    image_model = cxs.ContrastImageModel(instrument_config, scattering_theory)
    distribution = cls(image_model, normalizes_signal=False)
    np.testing.assert_allclose(image_model.render(), distribution.compute_signal())
