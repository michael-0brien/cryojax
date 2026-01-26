from typing_extensions import override

from jaxtyping import Array, Complex, PRNGKeyArray

from .._image_config import AbstractImageConfig
from .._multislice import AbstractMultisliceIntegrator
from .._scattering_theory import AbstractWaveScatteringTheory
from .._transfer_theory import WaveTransferTheory
from .._volume import AbstractVolumeRepresentation


class MultisliceScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """A scattering theory using the multislice method."""

    volume_integrator: AbstractMultisliceIntegrator
    transfer_theory: WaveTransferTheory

    def __init__(
        self,
        volume_integrator: AbstractMultisliceIntegrator,
        transfer_theory: WaveTransferTheory,
    ):
        """**Arguments:**

        - `volume_integrator`: The multislice method.
        - `transfer_theory`: The wave transfer theory.
        """
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory

    @override
    def compute_exit_wave(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        del rng_key
        # Compute the wavefunction in the exit plane
        wavefunction = self.volume_integrator.integrate(
            volume_representation, image_config
        )

        return wavefunction
