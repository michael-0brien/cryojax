from typing import Optional
from typing_extensions import override

from jaxtyping import Array, Complex, Float, PRNGKeyArray

from .._config import AbstractConfig
from .._potential_integrator import AbstractPotentialIntegrator
from .._potential_representation import AbstractPotentialRepresentation
from .._solvent import AbstractRandomSolvent
from .._transfer_theory import ContrastTransferTheory
from .base_scattering_theory import AbstractWeakPhaseScatteringTheory
from .common_functions import apply_interaction_constant


class WeakPhaseScatteringTheory(AbstractWeakPhaseScatteringTheory, strict=True):
    """Base linear image formation theory."""

    integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    solvent: Optional[AbstractRandomSolvent] = None

    def __init__(
        self,
        integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        solvent: Optional[AbstractRandomSolvent] = None,
    ):
        """**Arguments:**

        - `integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `solvent`: The model for the solvent.
        """
        self.integrator = integrator
        self.transfer_theory = transfer_theory
        self.solvent = solvent

    @override
    def compute_object_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        # Compute the integrated potential
        fourier_in_plane_potential = self.integrator.integrate(
            potential, config, outputs_real_space=False
        )

        if rng_key is not None:
            # Get the potential of the specimen plus the ice
            if self.solvent is not None:
                fourier_in_plane_potential = self.solvent.compute_in_plane_potential(  # noqa: E501
                    rng_key,
                    fourier_in_plane_potential,
                    config,
                    input_is_rfft=self.integrator.is_projection_approximation,
                )

        object_spectrum = apply_interaction_constant(
            fourier_in_plane_potential, config.wavelength_in_angstroms
        )

        return object_spectrum

    @override
    def compute_contrast_spectrum(
        self,
        potential: AbstractPotentialRepresentation,
        config: AbstractConfig,
        rng_key: Optional[PRNGKeyArray] = None,
        defocus_offset: Optional[float | Float[Array, ""]] = None,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim//2+1}"]:
        object_spectrum = self.compute_object_spectrum(potential, config, rng_key)
        contrast_spectrum = self.transfer_theory.propagate_object(  # noqa: E501
            object_spectrum,
            config,
            is_projection_approximation=self.integrator.is_projection_approximation,
            defocus_offset=defocus_offset,
        )

        return contrast_spectrum
