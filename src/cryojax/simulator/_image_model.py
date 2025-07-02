"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..ndimage import irfftn, rfftn
from ..ndimage.transforms import FilterLike, MaskLike
from ._detector import AbstractDetector
from ._instrument_config import InstrumentConfig
from ._potential_integrator import AbstractPotentialIntegrator
from ._scattering_theory import AbstractScatteringTheory
from ._structure import AbstractBiologicalStructure
from ._transfer_theory import ContrastTransferTheory


ImageArray = (
    Float[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"]
    | Complex[Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}"]
)
PaddedImageArray = (
    Float[
        Array,
        "{self.instrument_config.padded_y_dim} " "{self.instrument_config.padded_x_dim}",
    ]
    | Complex[
        Array,
        "{self.instrument_config.padded_y_dim} "
        "{self.instrument_config.padded_x_dim//2+1}",
    ]
)


class AbstractImageModel(eqx.Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractImageModel`'s `render` routine.
    """

    structure: eqx.AbstractVar[AbstractBiologicalStructure]
    instrument_config: eqx.AbstractVar[InstrumentConfig]
    filter: eqx.AbstractVar[Optional[FilterLike]]
    mask: eqx.AbstractVar[Optional[MaskLike]]

    @abstractmethod
    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        applies_mask: bool = True,
        applies_filter: bool = True,
    ) -> ImageArray | PaddedImageArray:
        """Render an image without any stochasticity.

        **Arguments:**

        - `rng_key`:
            The random number generator key. If not passed, render an image
            with no stochasticity.
        - `removes_padding`:
            If `True`, return an image cropped to `InstrumentConfig.shape`.
            Otherwise, return an image at the `InstrumentConfig.padded_shape`.
            If `removes_padding = False`, the `AbstractImageModel.filter`
            and `AbstractImageModel.mask` are not applied, overriding
            the booleans `applies_mask` and `applies_filter`.
        - `outputs_real_space`:
            If `True`, return the image in real space.
        - `applies_mask`:
            If `True`, apply mask stored in `AbstractImageModel.mask`.
        - `applies_filter`:
            If `True`, apply filter stored in `AbstractImageModel.filter`.
        """
        raise NotImplementedError

    def postprocess(
        self,
        image: Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ],
        *,
        outputs_real_space: bool = True,
        applies_mask: bool = True,
        applies_filter: bool = True,
    ) -> ImageArray:
        """Return an image postprocessed with filters, cropping, and masking
        in either real or fourier space.
        """
        instrument_config = self.instrument_config
        if (
            self.mask is None
            and instrument_config.padded_shape == instrument_config.shape
        ):
            # ... if there are no masks and we don't need to crop,
            # minimize moving back and forth between real and fourier space
            if self.filter is not None:
                image = self.filter(image) if applies_filter else image
            return (
                irfftn(image, s=instrument_config.shape) if outputs_real_space else image
            )
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            is_filter_applied = True if self.filter is None else False
            if (
                self.filter is not None
                and self.filter.array.shape
                == instrument_config.padded_frequency_grid_in_pixels.shape[0:2]
            ):
                # ... apply the filter here if it is the same size as the padded
                # coordinates
                is_filter_applied = True
                image = self.filter(image) if applies_filter else image
            image = irfftn(image, s=instrument_config.padded_shape)
            image = instrument_config.crop_to_shape(image)
            if self.mask is not None:
                image = self.mask(image) if applies_mask else image
            if is_filter_applied or self.filter is None:
                return image if outputs_real_space else rfftn(image)
            else:
                # ... otherwise, apply the filter here and return. assume
                # the filter is the same size as the non-padded coordinates
                image = self.filter(rfftn(image)) if applies_filter else rfftn(image)
                return (
                    irfftn(image, s=instrument_config.shape)
                    if outputs_real_space
                    else image
                )

    def _maybe_postprocess(
        self,
        image: Complex[
            Array,
            "{self.instrument_config.padded_y_dim} "
            "{self.instrument_config.padded_x_dim//2+1}",
        ],
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        applies_mask: bool = True,
        applies_filter: bool = True,
    ) -> PaddedImageArray | ImageArray:
        instrument_config = self.instrument_config
        if removes_padding:
            return self.postprocess(
                image,
                outputs_real_space=outputs_real_space,
                applies_mask=applies_mask,
                applies_filter=applies_filter,
            )
        else:
            return (
                irfftn(image, s=instrument_config.padded_shape)
                if outputs_real_space
                else image
            )


class LinearImageModel(AbstractImageModel, strict=True):
    structure: AbstractBiologicalStructure
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        instrument_config: InstrumentConfig,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
        self.instrument_config = instrument_config
        self.potential_integrator = potential_integrator
        self.structure = structure
        self.transfer_theory = transfer_theory
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
        # Get potential in the lab frame
        potential = self.structure.get_potential_in_transformed_frame(
            apply_translation=False
        )
        # Compute the projection image
        fourier_projection = self.potential_integrator.compute_integrated_potential(
            potential, self.instrument_config, outputs_real_space=False
        )
        # Compute the image
        image = self.transfer_theory.propagate_object_to_detector_plane(  # noqa: E501
            fourier_projection,
            self.instrument_config,
            is_projection_approximation=self.potential_integrator.is_projection_approximation,
            defocus_offset=self.structure.pose.offset_z_in_angstroms,
        )
        # Now for the in-plane translation
        translation_operator = self.structure.pose.compute_translation_operator(
            self.instrument_config.padded_frequency_grid_in_angstroms
        )
        image = self.structure.pose.translate_image(
            image, translation_operator, self.instrument_config.padded_shape
        )

        return self._maybe_postprocess(
            image,
            removes_padding=removes_padding,
            outputs_real_space=outputs_real_space,
            applies_mask=applies_mask,
            applies_filter=applies_filter,
        )


class AbstractPhysicalImageModel(AbstractImageModel, strict=True):
    """An image formation model that simulates physical
    quantities. This uses the `AbstractScatteringTheory` class.
    """

    scattering_theory: eqx.AbstractVar[AbstractScatteringTheory]


class ContrastImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns the image contrast from a linear
    scattering theory.

    **Attributes:**

    - `instrument_config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory. This must be a linear scattering
                           theory.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
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
        # Compute the squared wavefunction
        fourier_contrast_at_detector_plane = (
            self.scattering_theory.compute_contrast_spectrum_at_detector_plane(
                self.structure, self.instrument_config, rng_key
            )
        )

        return self._maybe_postprocess(
            fourier_contrast_at_detector_plane,
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

    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
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
        theory = self.scattering_theory
        fourier_squared_wavefunction_at_detector_plane = (
            theory.compute_intensity_spectrum_at_detector_plane(
                self.structure, self.instrument_config, rng_key
            )
        )

        return self._maybe_postprocess(
            fourier_squared_wavefunction_at_detector_plane,
            removes_padding=removes_padding,
            outputs_real_space=outputs_real_space,
            applies_mask=applies_mask,
            applies_filter=applies_filter,
        )


class ElectronCountsImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns electron counts, given a
    model for the detector.

    **Attributes:**

    - `instrument_config`: The configuration of the instrument, such as for the pixel size
                and the wavelength.
    - `scattering_theory`: The scattering theory.
    - `detector`: The electron detector.
    - `filter: `A filter to apply to the image.
    - `mask`: A mask to apply to the image.
    """

    instrument_config: InstrumentConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    filter: Optional[FilterLike]
    mask: Optional[MaskLike]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        filter: Optional[FilterLike] = None,
        mask: Optional[MaskLike] = None,
    ):
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
        if rng_key is None:
            # Compute the squared wavefunction
            theory = self.scattering_theory
            fourier_squared_wavefunction_at_detector_plane = (
                theory.compute_intensity_spectrum_at_detector_plane(
                    self.structure, self.instrument_config
                )
            )
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.detector.compute_expected_electron_events(
                    fourier_squared_wavefunction_at_detector_plane, self.instrument_config
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
            theory = self.scattering_theory
            fourier_squared_wavefunction_at_detector_plane = (
                theory.compute_intensity_spectrum_at_detector_plane(
                    self.structure, self.instrument_config, keys[0]
                )
            )
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[1],
                fourier_squared_wavefunction_at_detector_plane,
                self.instrument_config,
            )

            return self._maybe_postprocess(
                fourier_detector_readout,
                removes_padding=removes_padding,
                outputs_real_space=outputs_real_space,
                applies_mask=applies_mask,
                applies_filter=applies_filter,
            )
