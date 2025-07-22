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
from .._scattering_theory import AbstractScatteringTheory
from .._structure import AbstractStructure
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

    structure: AbstractStructure
    config: AbstractConfig
    scattering_theory: AbstractScatteringTheory

    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure: AbstractStructure,
        config: AbstractConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `normalizes_signal`:
            If `True`, normalize the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageModel.shape`.
        """
        self.structure = structure
        self.config = config
        self.scattering_theory = scattering_theory
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the potential
        potential = self.structure.get_potential_in_transformed_frame()
        # Compute the squared wavefunction
        contrast_spectrum = self.scattering_theory.compute_contrast_spectrum(
            potential,
            self.config,
            rng_key,
            defocus_offset=self.structure.pose.offset_z_in_angstroms,
        )
        # Apply the translation
        contrast_spectrum = self._apply_translation(contrast_spectrum)

        return contrast_spectrum


class IntensityImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns an intensity distribution---or in other
    words a squared wavefunction.
    """

    structure: AbstractStructure
    config: AbstractConfig
    scattering_theory: AbstractScatteringTheory

    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure: AbstractStructure,
        config: AbstractConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `normalizes_signal`:
            If `True`, normalize the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageModel.shape`.
        """
        self.structure = structure
        self.config = config
        self.scattering_theory = scattering_theory
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        potential = self.structure.get_potential_in_transformed_frame()
        scattering_theory = self.scattering_theory
        fourier_intensity = scattering_theory.compute_intensity_spectrum(
            potential,
            self.config,
            rng_key,
            defocus_offset=self.structure.pose.offset_z_in_angstroms,
        )
        fourier_intensity = self._apply_translation(fourier_intensity)

        return fourier_intensity


class ElectronCountsImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns electron counts, given a
    model for the detector.
    """

    structure: AbstractStructure
    config: DoseConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure: AbstractStructure,
        config: DoseConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `normalizes_signal`:
            If `True`, normalize the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageModel.shape`.
        """
        self.structure = structure
        self.config = config
        self.scattering_theory = scattering_theory
        self.detector = detector
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        potential = self.structure.get_potential_in_transformed_frame()
        if rng_key is None:
            # Compute the squared wavefunction
            scattering_theory = self.scattering_theory
            fourier_intensity = scattering_theory.compute_intensity_spectrum(
                potential,
                self.config,
                defocus_offset=self.structure.pose.offset_z_in_angstroms,
            )
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
            scattering_theory = self.scattering_theory
            fourier_intensity = scattering_theory.compute_intensity_spectrum(
                potential,
                self.config,
                keys[0],
                defocus_offset=self.structure.pose.offset_z_in_angstroms,
            )
            fourier_intensity = self._apply_translation(fourier_intensity)
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[1],
                fourier_intensity,
                self.config,
            )

            return fourier_detector_readout
