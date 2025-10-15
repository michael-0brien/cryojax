from typing_extensions import override

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from .._image_config import AbstractImageConfig
from .._solvent_2d import AbstractRandomSolvent2D
from .._transfer_theory import ContrastTransferTheory
from .._volume import AbstractVolumeRepresentation
from .._volume_integrator import AbstractVolumeIntegrator
from .base_scattering_theory import AbstractWeakPhaseScatteringTheory


class WeakPhaseScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
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

    @override
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
                    input_is_rfft=self.volume_integrator.is_projection_approximation,
                )

        object_spectrum = image_config.interaction_constant * fourier_in_plane_potential

        return object_spectrum

    @override
    def compute_contrast_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: PRNGKeyArray | None = None,
        defocus_offset: float | Float[Array, ""] | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        object_spectrum = self.compute_object_spectrum(
            volume_representation, image_config, rng_key
        )
        contrast_spectrum = self.transfer_theory.propagate_object(  # noqa: E501
            object_spectrum,
            image_config,
            is_projection_approximation=self.volume_integrator.is_projection_approximation,
            defocus_offset=defocus_offset,
        )

        return contrast_spectrum
