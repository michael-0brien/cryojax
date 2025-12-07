from abc import abstractmethod
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact, PRNGKeyArray

from ..jax_util import FloatLike, error_if_not_fractional
from ..ndimage import fftn, ifftn, irfftn, rfftn
from ._image_config import AbstractImageConfig
from ._solvent_2d import AbstractRandomSolvent2D
from ._transfer_theory import ContrastTransferTheory, WaveTransferTheory
from ._volume import AbstractVolumeIntegrator, AbstractVolumeRepresentation


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


class WeakPhaseScatteringTheory(AbstractScatteringTheory, strict=True):
    """Base linear image formation theory."""

    volume_integrator: AbstractVolumeIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: AbstractRandomSolvent2D | None = None

    def __init__(
        self,
        volume_integrator: AbstractVolumeIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: AbstractRandomSolvent2D | None = None,
    ):
        """**Arguments:**

        - `volume_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    def compute_object_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        # Compute the integrated potential
        fourier_in_plane_potential = self.volume_integrator.integrate(
            volume_representation, image_config, outputs_real_space=False
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_in_plane_potential = self.solvent.compute_in_plane_potential(  # noqa: E501
                    rng_key,
                    fourier_in_plane_potential,
                    image_config,
                    input_is_rfft=not self.volume_integrator.outputs_ewald_sphere,
                )

        object_spectrum = image_config.interaction_constant * fourier_in_plane_potential

        return object_spectrum

    @override
    def compute_contrast_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        object_spectrum = self.compute_object_spectrum(
            volume_representation, image_config, rng_key
        )
        contrast_spectrum = self.transfer_theory.propagate_object(  # noqa: E501
            object_spectrum,
            image_config,
            input_is_ewald_sphere=self.volume_integrator.outputs_ewald_sphere,
            defocus_offset=defocus_offset,
        )

        return contrast_spectrum

    @override
    def compute_intensity_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        """Compute the squared wavefunction at the detector plane, given the
        contrast.
        """
        N1, N2 = image_config.padded_shape
        # ... compute the squared wavefunction directly from the image contrast
        # as |psi|^2 = 1 + 2C.
        contrast_spectrum = self.compute_contrast_spectrum(
            volume_representation, image_config, rng_key, defocus_offset=defocus_offset
        )
        intensity_spectrum = (2 * contrast_spectrum).at[0, 0].add(1.0 * N1 * N2)
        return intensity_spectrum


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
        is_projection_approx = not self.volume_integrator.outputs_ewald_sphere
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
