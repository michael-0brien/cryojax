import cryojax.simulator as cs
import jax
import jax.numpy as jnp
import numpy as np
from cryojax.ndimage import irfftn, rfftn


def test_constant_wavefunction_gives_constant_expected_events():
    # Pick a large integrated electron flux to test
    # Create DoseImageConfig, picking a large electron flux to test
    image_config = cs.DoseImageConfig(
        (25, 25),
        1.0,
        voltage_in_kilovolts=300.0,
        electron_dose=10000.0,
    )
    electrons_per_pixel = image_config.electron_dose * image_config.pixel_size**2
    # Create squared wavefunction of just vacuum, i.e. 1 everywhere
    vacuum_squared_wavefunction = jnp.ones(image_config.shape, dtype=float)
    fourier_vacuum_squared_wavefunction = rfftn(vacuum_squared_wavefunction)
    # Create detector models
    poisson_detector = cs.PoissonDetector()
    # Compute expected events
    expected_electron_events = poisson_detector.compute_expected_counts(
        fourier_vacuum_squared_wavefunction,
        image_config,
    )
    # Make sure it is equal to the electron per pixel
    np.testing.assert_allclose(
        expected_electron_events,
        jnp.full(image_config.padded_shape, electrons_per_pixel),
        # rtol=1e-2,
    )


def test_gaussian_limit():
    # Pick a large integrated electron flux to test
    # Create DoseImageConfig, picking a large electron flux to test
    image_config = cs.DoseImageConfig(
        (25, 25),
        1.0,
        voltage_in_kilovolts=300.0,
        electron_dose=10000.0,
    )
    n_pixels = np.prod(image_config.padded_shape)
    electrons_per_pixel = image_config.electron_dose * image_config.pixel_size**2
    # Create squared wavefunction of just vacuum, i.e. 1 everywhere
    vacuum_squared_wavefunction = jnp.ones(image_config.shape, dtype=float)
    fourier_vacuum_squared_wavefunction = rfftn(vacuum_squared_wavefunction)
    # Create detector models
    key = jax.random.key(1234)
    gaussian_detector = cs.GaussianDetector()
    poisson_detector = cs.PoissonDetector()
    # Compute detector readout
    gaussian_detector_readout_fft = gaussian_detector.sample_counts(
        key, fourier_vacuum_squared_wavefunction, image_config, outputs_real_space=False
    )
    poisson_detector_readout_fft = poisson_detector.sample_counts(
        key, fourier_vacuum_squared_wavefunction, image_config, outputs_real_space=False
    )
    # Compare to see if the autocorrelation has converged
    np.testing.assert_allclose(
        irfftn(
            jnp.abs(gaussian_detector_readout_fft) ** 2
            / (n_pixels * electrons_per_pixel**2),
            s=image_config.padded_shape,
        ),
        irfftn(
            jnp.abs(poisson_detector_readout_fft) ** 2
            / (n_pixels * electrons_per_pixel**2),
            s=image_config.padded_shape,
        ),
        rtol=1e-2,
    )
