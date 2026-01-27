"""
Abstraction of electron detectors in a cryo-EM image.
"""

from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import Module
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..ndimage import irfftn, rfftn
from ._image_config import DoseImageConfig


class AbstractDetector(Module, strict=True):
    """Base class for an electron detector."""

    def compute_expected_counts(
        self,
        fourier_intensity: Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ],
        image_config: DoseImageConfig,
        *,
        outputs_real_space: bool = True,
    ) -> (
        Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]
        | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    ):
        """Compute the expected electron counts from the detector."""
        # The total number of electrons over the entire image
        n_pixels = np.prod(image_config.padded_shape)
        electrons_per_image = n_pixels * image_config.electrons_per_pixel
        # Normalize the squared wavefunction to a set of probabilities
        fourier_intensity /= fourier_intensity[0, 0]
        # Compute the noiseless signal by applying the DQE to the squared wavefunction
        # fourier_intensity = fourier_intensity * jnp.sqrt(self.dqe(image_config))
        # Apply the integrated dose rate
        fourier_expected_counts = electrons_per_image * fourier_intensity
        if outputs_real_space:
            return irfftn(fourier_expected_counts, s=image_config.padded_shape)
        else:
            return fourier_expected_counts

    @abstractmethod
    def sample_counts(
        self,
        key: PRNGKeyArray,
        fourier_intensity: Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ],
        image_config: DoseImageConfig,
        *,
        outputs_real_space: bool = True,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        """Measure the electron counts from the detector."""
        raise NotImplementedError


class GaussianDetector(AbstractDetector, strict=True):
    """A detector with a gaussian noise model. This is the gaussian limit
    of `PoissonDetector`.
    """

    @override
    def sample_counts(
        self,
        key: PRNGKeyArray,
        fourier_intensity: Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ],
        image_config: DoseImageConfig,
        *,
        outputs_real_space: bool = True,
    ) -> Float[Array, "y_dim x_dim"]:
        expected_electron_counts = self.compute_expected_counts(
            fourier_intensity, image_config, outputs_real_space=True
        )
        electron_counts = expected_electron_counts + jnp.sqrt(
            expected_electron_counts
        ) * jr.normal(key, expected_electron_counts.shape)
        return electron_counts if outputs_real_space else rfftn(electron_counts)


class PoissonDetector(AbstractDetector, strict=True):
    """A detector with a poisson noise model."""

    @override
    def sample_counts(
        self,
        key: PRNGKeyArray,
        fourier_intensity: Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ],
        image_config: DoseImageConfig,
        *,
        outputs_real_space: bool = True,
    ) -> Float[Array, "y_dim x_dim"]:
        expected_electron_counts = self.compute_expected_counts(
            fourier_intensity, image_config, outputs_real_space=True
        )
        electron_counts = jr.poisson(key, expected_electron_counts).astype(float)
        return electron_counts if outputs_real_space else rfftn(electron_counts)
