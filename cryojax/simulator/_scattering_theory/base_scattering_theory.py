from abc import abstractmethod
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, PRNGKeyArray

from ...jax_util import FloatLike
from ...ndimage import fftn, ifftn, rfftn
from .._image_config import AbstractImageConfig
from .._transfer_theory import (
    WaveTransferTheory,
)
from .._volume import AbstractVolumeRepresentation


class AbstractScatteringTheory(eqx.Module, strict=True):
    """Base class for a scattering theory."""

    @abstractmethod
    def compute_contrast_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        raise NotImplementedError

    @abstractmethod
    def compute_intensity_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        raise NotImplementedError


class AbstractWaveScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a wave-based scattering theory."""

    transfer_theory: eqx.AbstractVar[WaveTransferTheory]

    @abstractmethod
    def compute_exit_wave(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        # ... compute the exit wave
        fourier_wavefunction = fftn(
            self.compute_exit_wave(volume_representation, image_config, rng_key)
        )
        # ... propagate to the detector plane
        fourier_wavefunction = self.transfer_theory.propagate_exit_wave(
            fourier_wavefunction,
            image_config,
            defocus_offset=defocus_offset,
        )
        wavefunction = ifftn(fourier_wavefunction)
        # ... get the squared wavefunction and return to fourier space
        intensity_spectrum = rfftn((wavefunction * jnp.conj(wavefunction)).real)

        return intensity_spectrum

    @override
    def compute_contrast_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        """Compute the contrast at the detector plane, given the squared wavefunction."""
        # ... compute the exit wave
        fourier_wavefunction = fftn(
            self.compute_exit_wave(volume_representation, image_config, rng_key)
        )
        # ... propagate to the detector plane
        fourier_wavefunction = self.transfer_theory.propagate_exit_wave(
            fourier_wavefunction,
            image_config,
            defocus_offset=defocus_offset,
        )
        wavefunction = ifftn(fourier_wavefunction)
        # ... get the squared wavefunction
        squared_wavefunction = (wavefunction * jnp.conj(wavefunction)).real
        # ... compute the contrast directly from the squared wavefunction
        # as C = -1 + psi^2 / 1 + psi^2
        contrast_spectrum = rfftn(
            (-1 + squared_wavefunction) / (1 + squared_wavefunction)
        )

        return contrast_spectrum
