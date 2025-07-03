from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...ndimage import fftn, ifftn, rfftn
from .._instrument_config import InstrumentConfig
from .._potential_representation import AbstractPotentialRepresentation
from .._transfer_theory import (
    ContrastTransferTheory,
    WaveTransferTheory,
)


class AbstractScatteringTheory(eqx.Module, strict=True):
    """Base class for a scattering theory."""

    @abstractmethod
    def compute_contrast_spectrum_at_detector_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError

    @abstractmethod
    def compute_intensity_spectrum_at_detector_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError


class AbstractWaveScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a wave-based scattering theory."""

    transfer_theory: eqx.AbstractVar[WaveTransferTheory]
    amplitude_contrast_ratio: eqx.AbstractVar[Float[Array, ""]]

    @abstractmethod
    def compute_wavefunction_at_exit_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum_at_detector_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        # ... compute the exit wave
        fourier_wavefunction_at_exit_plane = fftn(
            self.compute_wavefunction_at_exit_plane(potential, instrument_config, rng_key)
        )
        # ... propagate to the detector plane
        fourier_wavefunction_at_detector_plane = (
            self.transfer_theory.propagate_wavefunction_to_detector_plane(
                fourier_wavefunction_at_exit_plane,
                instrument_config,
                defocus_offset=defocus_offset,
            )
        )
        wavefunction_at_detector_plane = ifftn(fourier_wavefunction_at_detector_plane)
        # ... get the squared wavefunction and return to fourier space
        intensity_spectrum_at_detector_plane = rfftn(
            (
                wavefunction_at_detector_plane * jnp.conj(wavefunction_at_detector_plane)
            ).real
        )

        return intensity_spectrum_at_detector_plane

    @override
    def compute_contrast_spectrum_at_detector_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the contrast at the detector plane, given the squared wavefunction."""
        # ... compute the exit wave
        fourier_wavefunction_at_exit_plane = fftn(
            self.compute_wavefunction_at_exit_plane(potential, instrument_config, rng_key)
        )
        # ... propagate to the detector plane
        fourier_wavefunction_at_detector_plane = (
            self.transfer_theory.propagate_wavefunction_to_detector_plane(
                fourier_wavefunction_at_exit_plane,
                instrument_config,
                defocus_offset=defocus_offset,
            )
        )
        wavefunction_at_detector_plane = ifftn(fourier_wavefunction_at_detector_plane)
        # ... get the squared wavefunction
        squared_wavefunction_at_detector_plane = (
            wavefunction_at_detector_plane * jnp.conj(wavefunction_at_detector_plane)
        ).real
        # ... compute the contrast directly from the squared wavefunction
        # as C = -1 + psi^2 / 1 + psi^2
        contrast_spectrum_at_detector_plane = rfftn(
            (-1 + squared_wavefunction_at_detector_plane)
            / (1 + squared_wavefunction_at_detector_plane)
        )

        return contrast_spectrum_at_detector_plane


class AbstractWeakPhaseScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base class for a scattering theory in linear image formation theory
    (the weak-phase approximation).
    """

    transfer_theory: eqx.AbstractVar[ContrastTransferTheory]

    @abstractmethod
    def compute_object_spectrum_at_exit_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        raise NotImplementedError

    @override
    def compute_intensity_spectrum_at_detector_plane(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = instrument_config.padded_shape
        # ... compute the squared wavefunction directly from the image contrast
        # as |psi|^2 = 1 + 2C.
        contrast_spectrum_at_detector_plane = (
            self.compute_contrast_spectrum_at_detector_plane(
                potential, instrument_config, rng_key, defocus_offset=defocus_offset
            )
        )
        intensity_spectrum_at_detector_plane = (
            (2 * contrast_spectrum_at_detector_plane).at[0, 0].add(1.0 * N1 * N2)
        )
        return intensity_spectrum_at_detector_plane
