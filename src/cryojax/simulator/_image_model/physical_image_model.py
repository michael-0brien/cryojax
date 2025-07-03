"""
Image formation models.
"""

from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._detector import AbstractDetector
from .._instrument_config import InstrumentConfig
from .._scattering_theory import AbstractScatteringTheory
from .._structure import AbstractBiologicalStructure
from .base_image_model import AbstractImageModel, ImageArray, PaddedImageArray


class AbstractPhysicalImageModel(AbstractImageModel, strict=True):
    """An image formation model that simulates physical
    quantities. This uses the `AbstractScatteringTheory` class.
    """

    scattering_theory: eqx.AbstractVar[AbstractScatteringTheory]


class ContrastImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns the image contrast from a linear
    scattering theory.
    """

    structure: AbstractBiologicalStructure
    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `instrument_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        """
        self.structure = structure
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the potential
        potential = self.structure.get_potential_in_transformed_frame()
        # Compute the squared wavefunction
        contrast_spectrum_at_detector_plane = (
            self.scattering_theory.compute_contrast_spectrum_at_detector_plane(
                potential,
                self.instrument_config,
                rng_key,
                defocus_offset=self.structure.pose.offset_z_in_angstroms,
            )
        )
        # Apply the translation
        contrast_spectrum_at_detector_plane = self._apply_translation(
            contrast_spectrum_at_detector_plane
        )

        return contrast_spectrum_at_detector_plane


class IntensityImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns an intensity distribution---or in other
    words a squared wavefunction.

    **Attributes:**

    - `instrument_config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    structure: AbstractBiologicalStructure
    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `instrument_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        """
        self.structure = structure
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        potential = self.structure.get_potential_in_transformed_frame()
        scattering_theory = self.scattering_theory
        fourier_intensity_at_detector_plane = (
            scattering_theory.compute_intensity_spectrum_at_detector_plane(
                potential,
                self.instrument_config,
                rng_key,
                defocus_offset=self.structure.pose.offset_z_in_angstroms,
            )
        )
        fourier_intensity_at_detector_plane = self._apply_translation(
            fourier_intensity_at_detector_plane
        )

        return fourier_intensity_at_detector_plane


class ElectronCountsImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns electron counts, given a
    model for the detector.
    """

    structure: AbstractBiologicalStructure
    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `instrument_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        """
        self.structure = structure
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.detector = detector

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        potential = self.structure.get_potential_in_transformed_frame()
        if rng_key is None:
            # Compute the squared wavefunction
            scattering_theory = self.scattering_theory
            fourier_intensity_at_detector_plane = (
                scattering_theory.compute_intensity_spectrum_at_detector_plane(
                    potential,
                    self.instrument_config,
                    defocus_offset=self.structure.pose.offset_z_in_angstroms,
                )
            )
            fourier_intensity_at_detector_plane = self._apply_translation(
                fourier_intensity_at_detector_plane
            )
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.detector.compute_expected_electron_events(
                    fourier_intensity_at_detector_plane, self.instrument_config
                )
            )

            return fourier_expected_electron_events
        else:
            keys = jax.random.split(rng_key)
            # Compute the squared wavefunction
            scattering_theory = self.scattering_theory
            fourier_intensity_at_detector_plane = (
                scattering_theory.compute_intensity_spectrum_at_detector_plane(
                    potential,
                    self.instrument_config,
                    keys[0],
                    defocus_offset=self.structure.pose.offset_z_in_angstroms,
                )
            )
            fourier_intensity_at_detector_plane = self._apply_translation(
                fourier_intensity_at_detector_plane
            )
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[1],
                fourier_intensity_at_detector_plane,
                self.instrument_config,
            )

            return fourier_detector_readout
