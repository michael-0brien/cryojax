from typing_extensions import override

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...jax_util import FloatLike, error_if_not_fractional
from .._image_config import AbstractImageConfig
from .._multislice import AbstractMultisliceIntegrator
from .._scattering_theory import AbstractWaveScatteringTheory
from .._transfer_theory import WaveTransferTheory
from .._volume import AbstractVolumeRepresentation


class MultisliceScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """A scattering theory using the multislice method."""

    volume_integrator: AbstractMultisliceIntegrator
    transfer_theory: WaveTransferTheory
    amplitude_contrast_ratio: Float[Array, ""]

    def __init__(
        self,
        volume_integrator: AbstractMultisliceIntegrator,
        transfer_theory: WaveTransferTheory,
        amplitude_contrast_ratio: FloatLike = 0.1,
    ):
        """**Arguments:**

        - `volume_integrator`: The multislice method.
        - `transfer_theory`: The wave transfer theory.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        """
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)

    @override
    def compute_exit_wave(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        # Compute the wavefunction in the exit plane
        wavefunction = self.volume_integrator.integrate(
            volume_representation, image_config, self.amplitude_contrast_ratio
        )

        return wavefunction
