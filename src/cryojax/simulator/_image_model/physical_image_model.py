"""
Image formation models.
"""

from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ...ndimage.transforms import FilterLike, MaskLike
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

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `instrument_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `filter: `A filter to apply to the image.
        - `mask`: A mask to apply to the image.
        """
        self.structure = structure
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        applies_mask: bool = True,
        applies_filter: bool = True,
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

        return self._maybe_postprocess(
            contrast_spectrum_at_detector_plane,
            removes_padding=removes_padding,
            outputs_real_space=outputs_real_space,
            applies_mask=applies_mask,
            applies_filter=applies_filter,
        )


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

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `instrument_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `filter: `A filter to apply to the image.
        - `mask`: A mask to apply to the image.
        """
        self.structure = structure
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        applies_mask: bool = True,
        applies_filter: bool = True,
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

        return self._maybe_postprocess(
            fourier_intensity_at_detector_plane,
            removes_padding=removes_padding,
            outputs_real_space=outputs_real_space,
            applies_mask=applies_mask,
            applies_filter=applies_filter,
        )


class ElectronCountsImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns electron counts, given a
    model for the detector.
    """

    structure: AbstractBiologicalStructure
    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        """**Arguments:**

        - `structure`:
            The biological structure.
        - `instrument_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `scattering_theory`:
            The scattering theory.
        - `filter: `A filter to apply to the image.
        - `mask`: A mask to apply to the image.
        """
        self.structure = structure
        self.instrument_config = instrument_config
        self.scattering_theory = scattering_theory
        self.detector = detector
        self.filter = filter
        self.mask = mask

    @override
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        applies_mask: bool = True,
        applies_filter: bool = True,
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

            return self._maybe_postprocess(
                fourier_expected_electron_events,
                removes_padding=removes_padding,
                outputs_real_space=outputs_real_space,
                applies_mask=applies_mask,
                applies_filter=applies_filter,
            )
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

            return self._maybe_postprocess(
                fourier_detector_readout,
                removes_padding=removes_padding,
                outputs_real_space=outputs_real_space,
                applies_mask=applies_mask,
                applies_filter=applies_filter,
            )
