from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...internal import error_if_not_fractional
from ...ndimage import ifftn, irfftn
from .._instrument_config import InstrumentConfig
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractSolvent
from .._structure import AbstractBiologicalStructure
from .._transfer_theory import WaveTransferTheory
from .base_scattering_theory import AbstractWaveScatteringTheory
from .common_functions import apply_amplitude_contrast_ratio, apply_interaction_constant


class HighEnergyScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """Scattering theory in the high-energy approximation (eikonal approximation).

    This is the simplest model for multiple scattering events.

    **References:**

    - For the definition of the exit wave in the eikonal approximation, see Chapter 69,
      Page 2012, from *Hawkes, Peter W., and Erwin Kasper. Principles of Electron
      Optics, Volume 4: Advanced Wave Optics. Academic Press, 2022.*
    """

    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: WaveTransferTheory
    solvent: Optional[AbstractSolvent]
    amplitude_contrast_ratio: Float[Array, ""]

    def __init__(
        self,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: WaveTransferTheory,
        solvent: Optional[AbstractSolvent] = None,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
    ):
        """**Arguments:**

        - `potential_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The wave transfer theory.
        - `solvent`: The model for the solvent.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        """
        self.potential_integrator = potential_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)

    @override
    def compute_wavefunction_at_exit_plane(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Compute the integrated potential in the exit plane
        potential = structure.get_potential_in_transformed_frame(apply_translation=False)
        fourier_integrated_potential = (
            self.potential_integrator.compute_integrated_potential(
                potential, instrument_config, outputs_real_space=False
            )
        )
        # The integrated potential may not be from an rfft; this depends on
        # if it is a projection approx
        is_projection_approx = self.potential_integrator.is_projection_approximation
        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_integrated_potential = (
                    self.solvent.compute_integrated_potential_with_solvent(
                        rng_key,
                        fourier_integrated_potential,
                        instrument_config,
                        input_is_rfft=is_projection_approx,
                    )
                )
        # Back to real-space; need to be careful if the object spectrum is not an
        # rfftn
        do_ifft = lambda ft: (
            irfftn(ft, s=instrument_config.padded_shape)
            if is_projection_approx
            else ifftn(ft, s=instrument_config.padded_shape)
        )
        integrated_potential = apply_amplitude_contrast_ratio(
            do_ifft(fourier_integrated_potential), self.amplitude_contrast_ratio
        )
        object_at_exit_plane = apply_interaction_constant(
            integrated_potential, instrument_config.wavelength_in_angstroms
        )
        # Compute wavefunction, with amplitude and phase contrast
        return jnp.exp(1.0j * object_at_exit_plane)
