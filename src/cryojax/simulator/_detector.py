"""
Abstraction of electron detectors in a cryo-EM image.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..internal import error_if_not_fractional
from ..ndimage import irfftn, rfftn
from ._config import DoseConfig


class AbstractDQE(eqx.Module, strict=True):
    r"""Base class for a detector DQE."""

    fraction_detected_electrons: AbstractVar[Float[Array, ""]]

    @abstractmethod
    def __call__(
        self,
        frequency_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        *,
        pixel_size: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "y_dim x_dim"]:
        """**Arguments:**

        - `frequency_grid_in_angstroms_or_pixels`: A frequency grid
                                                   given in angstroms
                                                   or pixels. If given
                                                   in angstroms, `pixel_size`
                                                   must be passed
        - `pixel_size`: The pixel size of `frequency_grid_in_angstroms_or_pixels`.
        """
        raise NotImplementedError


class CountingDQE(AbstractDQE, strict=True):
    r"""A perfect DQE for a detector at a discrete pixel size.

    See Ruskin et. al. "Quantitative characterization of electron detectors for
    transmission electron microscopy." (2013) for details.
    """

    fraction_detected_electrons: Float[Array, ""]

    def __init__(self, fraction_detected_electrons: float | Float[Array, ""] = 1.0):
        self.fraction_detected_electrons = error_if_not_fractional(
            jnp.asarray(fraction_detected_electrons)
        )

    @override
    def __call__(
        self,
        frequency_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        *,
        pixel_size: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "y_dim x_dim"]:
        if pixel_size is None:
            frequency_grid_in_nyquist_units = frequency_grid_in_angstroms_or_pixels / 0.5
        else:
            frequency_grid_in_nyquist_units = (
                frequency_grid_in_angstroms_or_pixels * pixel_size
            ) / 0.5
        return (
            self.fraction_detected_electrons**2
            * jnp.sinc(frequency_grid_in_nyquist_units[..., 0] / 2) ** 2
            * jnp.sinc(frequency_grid_in_nyquist_units[..., 1] / 2) ** 2
        )


class NullDQE(AbstractDQE, strict=True):
    r"""A DQE that is perfect across all spatial frequencies."""

    fraction_detected_electrons: Float[Array, ""]

    def __init__(self, fraction_detected_electrons: float | Float[Array, ""] = 1.0):
        self.fraction_detected_electrons = error_if_not_fractional(
            jnp.asarray(fraction_detected_electrons)
        )

    @override
    def __call__(
        self,
        frequency_grid_in_angstroms_or_pixels: Float[Array, "y_dim x_dim 2"],
        *,
        pixel_size: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "y_dim x_dim"]:
        return jnp.full(
            frequency_grid_in_angstroms_or_pixels.shape[0:2],
            self.fraction_detected_electrons,
        )


class AbstractDetector(Module, strict=True):
    """Base class for an electron detector."""

    dqe: AbstractDQE

    def __init__(self, dqe: AbstractDQE):
        self.dqe = dqe

    @abstractmethod
    def sample_readout_from_expected_events(
        self, key: PRNGKeyArray, expected_electron_events: Float[Array, "y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"]:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError

    def compute_expected_electron_events(
        self,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ],
        config: DoseConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the expected electron events from the detector."""
        fourier_expected_electron_events = (
            self._compute_expected_events_or_detector_readout(
                fourier_squared_wavefunction_at_detector_plane,
                config,
                key=None,
            )
        )

        return fourier_expected_electron_events

    def compute_detector_readout(
        self,
        key: PRNGKeyArray,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ],
        config: DoseConfig,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Measure the readout from the detector."""
        fourier_detector_readout = self._compute_expected_events_or_detector_readout(
            fourier_squared_wavefunction_at_detector_plane,
            config,
            key,
        )

        return fourier_detector_readout

    def _compute_expected_events_or_detector_readout(
        self,
        fourier_squared_wavefunction_at_detector_plane: Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ],
        config: DoseConfig,
        key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Pass the image through the detector model."""
        N_pix = np.prod(config.padded_shape)
        frequency_grid = config.padded_frequency_grid_in_pixels
        # Compute the time-integrated electron flux in pixels
        electrons_per_pixel = config.electrons_per_angstrom_squared * config.pixel_size**2
        # ... now the total number of electrons over the entire image
        electrons_per_image = N_pix * electrons_per_pixel
        # Normalize the squared wavefunction to a set of probabilities
        fourier_squared_wavefunction_at_detector_plane /= (
            fourier_squared_wavefunction_at_detector_plane[0, 0]
        )
        # Compute the noiseless signal by applying the DQE to the squared wavefunction
        fourier_signal = fourier_squared_wavefunction_at_detector_plane * jnp.sqrt(
            self.dqe(frequency_grid)
        )
        # Apply the integrated dose rate
        fourier_expected_electron_events = electrons_per_image * fourier_signal
        if key is None:
            # If there is no key given, return
            return fourier_expected_electron_events
        else:
            # ... otherwise, go to real space, sample, go back to fourier,
            # and return.
            expected_electron_events = irfftn(
                fourier_expected_electron_events, s=config.padded_shape
            )
            return rfftn(
                self.sample_readout_from_expected_events(key, expected_electron_events)
            )


class GaussianDetector(AbstractDetector, strict=True):
    """A detector with a gaussian noise model. This is the gaussian limit
    of `PoissonDetector`.
    """

    @override
    def sample_readout_from_expected_events(
        self, key: PRNGKeyArray, expected_electron_events: Float[Array, "y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"]:
        return expected_electron_events + jnp.sqrt(expected_electron_events) * jr.normal(
            key, expected_electron_events.shape
        )


class PoissonDetector(AbstractDetector, strict=True):
    """A detector with a poisson noise model."""

    @override
    def sample_readout_from_expected_events(
        self, key: PRNGKeyArray, expected_electron_events: Float[Array, "y_dim x_dim"]
    ) -> Float[Array, "y_dim x_dim"]:
        return jr.poisson(key, expected_electron_events).astype(float)
