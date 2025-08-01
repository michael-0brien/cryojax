from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...internal import error_if_not_fractional
from ...ndimage import ifftn, irfftn
from .._common_functions import apply_amplitude_contrast_ratio, apply_interaction_constant
from .._image_config import AbstractImageConfig
from .._solvent_2d import AbstractRandomSolvent2D
from .._transfer_theory import WaveTransferTheory
from .._volume_integrator import AbstractDirectIntegrator, AbstractDirectVoxelIntegrator
from .._volume_parametrisation import AbstractVolumeRepresentation
from .base_scattering_theory import AbstractWaveScatteringTheory


class HighEnergyScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """Scattering theory in the high-energy approximation (eikonal approximation).

    This is the simplest model for multiple scattering events.

    **References:**

    - For the definition of the exit wave in the eikonal approximation, see Chapter 69,
      Page 2012, from *Hawkes, Peter W., and Erwin Kasper. Principles of Electron
      Optics, Volume 4: Advanced Wave Optics. Academic Press, 2022.*
    """

    volume_integrator: AbstractDirectIntegrator
    transfer_theory: WaveTransferTheory
    solvent: Optional[AbstractRandomSolvent2D]
    amplitude_contrast_ratio: Float[Array, ""]

    def __init__(
        self,
        volume_integrator: AbstractDirectIntegrator,
        transfer_theory: WaveTransferTheory,
        solvent: Optional[AbstractRandomSolvent2D] = None,
        amplitude_contrast_ratio: float | Float[Array, ""] = 0.1,
    ):
        """**Arguments:**

        - `volume_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The wave transfer theory.
        - `solvent`: The model for the solvent.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        """
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent
        self.amplitude_contrast_ratio = error_if_not_fractional(amplitude_contrast_ratio)

    def __check_init__(self):
        if isinstance(self.volume_integrator, AbstractDirectVoxelIntegrator):
            if not self.volume_integrator.outputs_integral:
                raise AttributeError(
                    "If the `volume_integrator` is voxel-based, "
                    "it must have `volume_integrator.outputs_integral = True` "
                    "to be passed to a `HighEnergyScatteringTheory`."
                )

    @override
    def compute_exit_wave(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        # Compute the integrated potential in the exit plane
        fourier_in_plane_potential = self.volume_integrator.integrate(
            volume_representation, image_config, outputs_real_space=False
        )
        # The integrated potential may not be from an rfft; this depends on
        # if it is a projection approx
        is_projection_approx = self.volume_integrator.is_projection_approximation
        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_in_plane_potential = self.solvent.compute_in_plane_potential(
                    rng_key,
                    fourier_in_plane_potential,
                    image_config,
                    input_is_rfft=is_projection_approx,
                )
        # Back to real-space; need to be careful if the object spectrum is not an
        # rfftn
        do_ifft = lambda ft: (
            irfftn(ft, s=image_config.padded_shape)
            if is_projection_approx
            else ifftn(ft, s=image_config.padded_shape)
        )
        integrated_potential = apply_amplitude_contrast_ratio(
            do_ifft(fourier_in_plane_potential), self.amplitude_contrast_ratio
        )
        object = apply_interaction_constant(
            integrated_potential, image_config.wavelength_in_angstroms
        )
        # Compute wavefunction, with amplitude and phase contrast
        return jnp.exp(1.0j * object)
