from typing import Optional
from typing_extensions import override

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...internal import error_if_not_fractional
from .._instrument_config import InstrumentConfig
from .._multislice_integrator import AbstractMultisliceIntegrator
from .._potential_representation import AbstractPotentialRepresentation
from .._transfer_theory import WaveTransferTheory
from .base_scattering_theory import AbstractWaveScatteringTheory


class MultisliceScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """A scattering theory using the multislice method."""

    integrator: AbstractMultisliceIntegrator
    transfer_theory: WaveTransferTheory
    amplitude_contrast_ratio: Float[Array, ""]

    def __init__(
        self,
        integrator: AbstractMultisliceIntegrator,
        transfer_theory: WaveTransferTheory,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
    ):
        """**Arguments:**

        - `integrator`: The multislice method.
        - `transfer_theory`: The wave transfer theory.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        """
        self.integrator = integrator
        self.transfer_theory = transfer_theory
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)

    @override
    def compute_wavefunction(
        self,
        potential: AbstractPotentialRepresentation,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        # Compute the wavefunction in the exit plane
        wavefunction = self.integrator.integrate(
            potential, instrument_config, self.amplitude_contrast_ratio
        )

        return wavefunction
