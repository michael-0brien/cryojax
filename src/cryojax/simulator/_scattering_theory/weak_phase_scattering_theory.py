from typing import Optional
from typing_extensions import override

from jaxtyping import Array, Complex, PRNGKeyArray

from .._instrument_config import InstrumentConfig
from .._potential_integrator import AbstractPotentialIntegrator
from .._solvent import AbstractSolvent
from .._structure import AbstractBiologicalStructure
from .._transfer_theory import ContrastTransferTheory
from .base_scattering_theory import AbstractWeakPhaseScatteringTheory
from .common_functions import apply_interaction_constant


class WeakPhaseScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
    """Base linear image formation theory."""

    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractSolvent] = None

    def __init__(
        self,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractSolvent] = None,
    ):
        """**Arguments:**

        - `potential_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.potential_integrator = potential_integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    @override
    def compute_object_spectrum_at_exit_plane(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        # Compute the integrated potential
        fourier_integrated_potential = _integrate_potential_to_exit_plane(
            structure, self.potential_integrator, instrument_config
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_integrated_potential = self.solvent.compute_integrated_potential_with_solvent(  # noqa: E501
                    rng_key,
                    fourier_integrated_potential,
                    instrument_config,
                    input_is_rfft=self.potential_integrator.is_projection_approximation,
                )

        object_spectrum_at_exit_plane = apply_interaction_constant(
            fourier_integrated_potential, instrument_config.wavelength_in_angstroms
        )

        return object_spectrum_at_exit_plane

    @override
    def compute_contrast_spectrum_at_detector_plane(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim//2+1}"
    ]:
        object_spectrum_at_exit_plane = self.compute_object_spectrum_at_exit_plane(
            structure, instrument_config, rng_key
        )
        contrast_spectrum_at_detector_plane = self.transfer_theory.propagate_object_to_detector_plane(  # noqa: E501
            object_spectrum_at_exit_plane,
            instrument_config,
            is_projection_approximation=self.potential_integrator.is_projection_approximation,
            defocus_offset=structure.pose.offset_z_in_angstroms,
        )
        # ... apply in-plane translation
        translation_operator = structure.pose.compute_translation_operator(
            instrument_config.padded_frequency_grid_in_angstroms
        )
        contrast_spectrum_at_detector_plane = structure.pose.translate_image(
            contrast_spectrum_at_detector_plane,
            translation_operator,
            instrument_config.padded_shape,
        )

        return contrast_spectrum_at_detector_plane


def _integrate_potential_to_exit_plane(
    structure: AbstractBiologicalStructure,
    potential_integrator: AbstractPotentialIntegrator,
    instrument_config: InstrumentConfig,
):
    # Get potential in the lab frame
    potential = structure.get_potential_in_transformed_frame(apply_translation=False)
    # Compute the phase shifts in the exit plane
    fourier_integrated_potential = potential_integrator.compute_integrated_potential(
        potential, instrument_config, outputs_real_space=False
    )
    return fourier_integrated_potential
