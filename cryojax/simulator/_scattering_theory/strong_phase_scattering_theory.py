from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact, PRNGKeyArray

from ...jax_util import FloatLike, error_if_not_fractional
from ...ndimage import ifftn, irfftn
from .._image_config import AbstractImageConfig
from .._solvent_2d import AbstractRandomSolvent2D
from .._transfer_theory import WaveTransferTheory
from .._volume import AbstractVolumeRepresentation
from .._volume_integrator import AbstractVolumeIntegrator
from .base_scattering_theory import AbstractWaveScatteringTheory


class StrongPhaseScatteringTheory(AbstractWaveScatteringTheory, strict=True):
    """Scattering theory for strong phase objects. This is analogous to
    a Moliere high-energy approximation in high-energy physics.

    This is the simplest model for multiple scattering events.

    !!! info
        Unlike in the weak-phase approximation, it is not possible to absorb a model
        for amplitude contrast (here via the amplitude contrast ratio) into the CTF.
        Instead, it is necessary to compute a complex scattering potential, where the
        imaginary part captures inelastic scattering.

        In particular, given a projected electrostatic potential $\\u(x, y)$, the
        complex potential $\\phi(x, y)$ for amplitude contrast ratio $\\alpha$ is

        $$\\phi(x, y) = \\sqrt{1 - \\alpha^2} \\ u(x, y) + i \\alpha \\ u(x, y).$$

    **References:**

    - See Chapter 69, Page 2012, from *Hawkes, Peter W., and Erwin Kasper.
      Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic
      Press, 2022.*
    - See Section 3.4, Page 61, from *Spence, John CH. High-resolution electron
      microscopy. OUP Oxford, 2013.*
    """

    volume_integrator: AbstractVolumeIntegrator
    transfer_theory: WaveTransferTheory
    solvent: AbstractRandomSolvent2D | None
    amplitude_contrast_ratio: Float[Array, ""]

    def __init__(
        self,
        volume_integrator: AbstractVolumeIntegrator,
        transfer_theory: WaveTransferTheory,
        solvent: AbstractRandomSolvent2D | None = None,
        amplitude_contrast_ratio: FloatLike = 0.1,
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

    @override
    def compute_exit_wave(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
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
        integrated_potential = _compute_complex_potential(
            do_ifft(fourier_in_plane_potential), self.amplitude_contrast_ratio
        )
        object = image_config.interaction_constant * integrated_potential
        # Compute wavefunction, with amplitude and phase contrast
        return jnp.exp(1.0j * object)


def _compute_complex_potential(
    in_plane_potential: Inexact[Array, "y_dim x_dim"],
    amplitude_contrast_ratio: Float[Array, ""] | float,
) -> Complex[Array, "y_dim x_dim"]:
    ac = amplitude_contrast_ratio
    if jnp.iscomplexobj(in_plane_potential):
        raise NotImplementedError(
            "You may have tried to use a `StrongPhaseScatteringTheory` "
            "together with `EwaldSphereExtraction` for simulating images. "
            "This is not implemented!"
        )
        # return jnp.sqrt(1.0 - ac**2) * integrated_potential.real + 1.0j * (
        #     integrated_potential.imag + ac * integrated_potential.real
        # )
    else:
        return (jnp.sqrt(1.0 - ac**2) + 1.0j * ac) * in_plane_potential
