from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...ndimage import fftn, ifftn, rfftn
from .._config import AbstractConfig
from .._potential_representation import AbstractPotentialRepresentation
from .._transfer_theory import (
    ContrastTransferTheory,
    WaveTransferTheory,
)


class AbstractScatteringTheory(eqx.Module, strict=True):
    """Base class for a scattering theory."""

    @abstractmethod
    def compute_contrast_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError

    @abstractmethod
    def compute_intensity_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError


class AbstractWaveScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a wave-based scattering theory."""

    transfer_theory: eqx.AbstractVar[WaveTransferTheory]
    amplitude_contrast_ratio: eqx.AbstractVar[Float[Array, ""]]

    @abstractmethod
    def compute_exit_wave(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        # ... compute the exit wave
        fourier_wavefunction = fftn(self.compute_exit_wave(potential, config, rng_key))
        # ... propagate to the detector plane
        fourier_wavefunction = self.transfer_theory.propagate_exit_wave(
            fourier_wavefunction,
            config,
            defocus_offset=defocus_offset,
        )
        wavefunction = ifftn(fourier_wavefunction)
        # ... get the squared wavefunction and return to fourier space
        intensity_spectrum = rfftn((wavefunction * jnp.conj(wavefunction)).real)

        return intensity_spectrum

    @override
    def compute_contrast_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the contrast at the detector plane, given the squared wavefunction."""
        # ... compute the exit wave
        fourier_wavefunction = fftn(self.compute_exit_wave(potential, config, rng_key))
        # ... propagate to the detector plane
        fourier_wavefunction = self.transfer_theory.propagate_exit_wave(
            fourier_wavefunction,
            config,
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


class AbstractWeakPhaseScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a scattering theory in linear image formation theory
    (the weak-phase approximation).
    """

    transfer_theory: eqx.AbstractVar[ContrastTransferTheory]

    @abstractmethod
    def compute_object_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = config.padded_shape
        # ... compute the squared wavefunction directly from the image contrast
        # as |psi|^2 = 1 + 2C.
        contrast_spectrum = self.compute_contrast_spectrum(
            potential, config, rng_key, defocus_offset=defocus_offset
        )
        intensity_spectrum = (2 * contrast_spectrum).at[0, 0].add(1.0 * N1 * N2)
        return intensity_spectrum
