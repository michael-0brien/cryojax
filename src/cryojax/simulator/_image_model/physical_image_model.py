"""
Image formation models.
"""

from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import Array, Bool, PRNGKeyArray

from .._config import AbstractConfig, DoseConfig
from .._detector import AbstractDetector
from .._pose import AbstractPose
from .._scattering_theory import AbstractScatteringTheory
from .._structure_modeling import AbstractStructureMapping
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

    structure_mapping: AbstractStructureMapping
    pose: AbstractPose
    config: AbstractConfig
    scattering_theory: AbstractScatteringTheory

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure_mapping: AbstractStructureMapping,
        pose: AbstractPose,
        config: AbstractConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure_mapping`:
            The map to a biological structure.
        - `pose`:
            The pose of a structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalize the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractConfig.shape`.
        """
        self.structure_mapping = structure_mapping
        self.pose = pose
        self.config = config
        self.scattering_theory = scattering_theory
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the structure. Its data should be a scattering potential
        # to simulate in physical units
        structure = self.structure_mapping.map_to_structure()
        # Rotate it to the lab frame
        structure = structure.rotate_to_pose(self.pose)
        # Compute the contrast
        contrast_spectrum = self.scattering_theory.compute_contrast_spectrum(
            structure,
            self.config,
            rng_key,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        # Apply the translation
        if self.applies_translation:
            contrast_spectrum = self._apply_translation(contrast_spectrum)

        return contrast_spectrum


class IntensityImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns an intensity distribution---or in other
    words a squared wavefunction.
    """

    structure_mapping: AbstractStructureMapping
    pose: AbstractPose
    config: AbstractConfig
    scattering_theory: AbstractScatteringTheory

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure_mapping: AbstractStructureMapping,
        pose: AbstractPose,
        config: AbstractConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure_mapping`:
            The map to a biological structure.
        - `pose`:
            The pose of a structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalize the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractConfig.shape`.
        """
        self.structure_mapping = structure_mapping
        self.pose = pose
        self.config = config
        self.scattering_theory = scattering_theory
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the structure. Its data should be a scattering potential
        # to simulate in physical units
        structure = self.structure_mapping.map_to_structure()
        # Rotate it to the lab frame
        structure = structure.rotate_to_pose(self.pose)
        # Compute the intensity spectrum
        intensity_spectrum = self.scattering_theory.compute_intensity_spectrum(
            structure,
            self.config,
            rng_key,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        if self.applies_translation:
            intensity_spectrum = self._apply_translation(intensity_spectrum)

        return intensity_spectrum


class ElectronCountsImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns electron counts, given a
    model for the detector.
    """

    structure_mapping: AbstractStructureMapping
    pose: AbstractPose
    config: DoseConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure_mapping: AbstractStructureMapping,
        pose: AbstractPose,
        config: DoseConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure_mapping`:
            The map to a biological structure.
        - `pose`:
            The pose of a structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalize the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractConfig.shape`.
        """
        self.structure_mapping = structure_mapping
        self.pose = pose
        self.config = config
        self.scattering_theory = scattering_theory
        self.detector = detector
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the structure. Its data should be a scattering potential
        # to simulate in physical units
        structure = self.structure_mapping.map_to_structure()
        # Rotate it to the lab frame
        structure = structure.rotate_to_pose(self.pose)
        if rng_key is None:
            # Compute the intensity
            fourier_intensity = self.scattering_theory.compute_intensity_spectrum(
                structure,
                self.config,
                defocus_offset=self.pose.offset_z_in_angstroms,
            )
            if self.applies_translation:
                fourier_intensity = self._apply_translation(fourier_intensity)
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.detector.compute_expected_electron_events(
                    fourier_intensity, self.config
                )
            )

            return fourier_expected_electron_events
        else:
            keys = jax.random.split(rng_key)
            # Compute the squared wavefunction
            fourier_intensity = self.scattering_theory.compute_intensity_spectrum(
                structure,
                self.config,
                keys[0],
                defocus_offset=self.pose.offset_z_in_angstroms,
            )
            if self.applies_translation:
                fourier_intensity = self._apply_translation(fourier_intensity)
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[1],
                fourier_intensity,
                self.config,
            )

            return fourier_detector_readout
