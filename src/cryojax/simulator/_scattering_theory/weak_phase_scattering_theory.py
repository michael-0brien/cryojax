from typing import Optional
from typing_extensions import override

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from .._common_functions import apply_interaction_constant
from .._image_config import AbstractImageConfig
from .._solvent_2d import AbstractRandomSolvent2D
from .._transfer_theory import ContrastTransferTheory
from .._volume_integrator import AbstractDirectIntegrator, AbstractDirectVoxelIntegrator
from .._volume_parametrisation import AbstractVolumeRepresentation
from .base_scattering_theory import AbstractWeakPhaseScatteringTheory


class WeakPhaseScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
    """Base linear image formation theory."""

    volume_integrator: AbstractDirectIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractRandomSolvent2D] = None

    def __init__(
        self,
        volume_integrator: AbstractDirectIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractRandomSolvent2D] = None,
    ):
        """**Arguments:**

        - `volume_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    def __check_init__(self):
        if isinstance(self.volume_integrator, AbstractDirectVoxelIntegrator):
            if not self.volume_integrator.outputs_integral:
                raise AttributeError(
                    "If the `volume_integrator` is voxel-based, "
                    "it must have `volume_integrator.outputs_integral = True` "
                    "to be passed to a `HighEnergyScatteringTheory`."
                )

    @override
    def compute_object_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: Optional[PRNGKeyArray] = None,
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

        object_spectrum = apply_interaction_constant(
            fourier_in_plane_potential,
            image_config.voltage_in_kilovolts,
        )

        return object_spectrum

    @override
    def compute_contrast_spectrum(
        self,
        volume_representation: AbstractVolumeRepresentation,
        image_config: AbstractImageConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
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
