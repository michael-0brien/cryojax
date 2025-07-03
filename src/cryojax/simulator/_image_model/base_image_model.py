"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...ndimage import irfftn, rfftn
from ...ndimage.transforms import FilterLike, MaskLike
from .._instrument_config import InstrumentConfig
from .._potential_integrator import AbstractPotentialIntegrator
from .._structure import AbstractBiologicalStructure
from .._transfer_theory import ContrastTransferTheory


RealImageArray = Float[
    Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim}"
]
FourierImageArray = Complex[
    Array, "{self.instrument_config.y_dim} {self.instrument_config.x_dim//2+1}"
]
PaddedRealImageArray = Float[
    Array,
    "{self.instrument_config.padded_y_dim} " "{self.instrument_config.padded_x_dim}",
]
PaddedFourierImageArray = Complex[
    Array,
    "{self.instrument_config.padded_y_dim} " "{self.instrument_config.padded_x_dim//2+1}",
]

ImageArray = RealImageArray | FourierImageArray
PaddedImageArray = PaddedRealImageArray | PaddedFourierImageArray


class AbstractImageModel(eqx.Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractImageModel`'s `render` routine.
    """

    structure: eqx.AbstractVar[AbstractBiologicalStructure]
    instrument_config: eqx.AbstractVar[InstrumentConfig]

    @abstractmethod
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        """Render an image without postprocessing."""
        raise NotImplementedError

    def render(
        self,
        rng_key: Optional[PRNGKeyArray] = None,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> ImageArray | PaddedImageArray:
        """Render an image.

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
        - `mask`:
            Optionally apply a mask to the image.
        - `filter`:
            Optionally apply a filter to the image.
        """
        fourier_image = self.compute_fourier_image(rng_key)

        return self._maybe_postprocess(
            fourier_image,
            removes_padding=removes_padding,
            outputs_real_space=outputs_real_space,
            mask=mask,
            filter=filter,
        )

    def postprocess(
        self,
        image: PaddedFourierImageArray,
        *,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> ImageArray:
        """Return an image postprocessed with filters, cropping, and masking
        in either real or fourier space.
        """
        instrument_config = self.instrument_config
        if mask is None and instrument_config.padded_shape == instrument_config.shape:
            # ... if there are no masks and we don't need to crop,
            # minimize moving back and forth between real and fourier space
            if filter is not None:
                image = filter(image)
            return (
                irfftn(image, s=instrument_config.shape) if outputs_real_space else image
            )
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            is_filter_applied = True if filter is None else False
            if (
                filter is not None
                and filter.array.shape
                == instrument_config.padded_frequency_grid_in_pixels.shape[0:2]
            ):
                # ... apply the filter here if it is the same size as the padded
                # coordinates
                is_filter_applied = True
                image = filter(image)
            image = irfftn(image, s=instrument_config.padded_shape)
            image = instrument_config.crop_to_shape(image)
            if mask is not None:
                image = mask(image)
            if is_filter_applied or filter is None:
                return image if outputs_real_space else rfftn(image)
            else:
                # ... otherwise, apply the filter here and return. assume
                # the filter is the same size as the non-padded coordinates
                image = filter(rfftn(image))
                return (
                    irfftn(image, s=instrument_config.shape)
                    if outputs_real_space
                    else image
                )

    def _apply_translation(
        self, fourier_image: PaddedFourierImageArray
    ) -> PaddedFourierImageArray:
        pose = self.structure.pose
        phase_shifts = pose.compute_translation_operator(
            self.instrument_config.padded_frequency_grid_in_angstroms
        )
        fourier_image = pose.translate_image(
            fourier_image,
            phase_shifts,
            self.instrument_config.padded_shape,
        )

        return fourier_image

    def _maybe_postprocess(
        self,
        image: PaddedFourierImageArray,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> PaddedImageArray | ImageArray:
        instrument_config = self.instrument_config
        if removes_padding:
            return self.postprocess(
                image, outputs_real_space=outputs_real_space, mask=mask, filter=filter
            )
        else:
            return (
                irfftn(image, s=instrument_config.padded_shape)
                if outputs_real_space
                else image
            )


class LinearImageModel(AbstractImageModel, strict=True):
    """An simple image model in linear image formation theory."""

    structure: AbstractBiologicalStructure
    potential_integrator: AbstractPotentialIntegrator
    transfer_theory: ContrastTransferTheory
    instrument_config: InstrumentConfig

    def __init__(
        self,
        structure: AbstractBiologicalStructure,
        potential_integrator: AbstractPotentialIntegrator,
        transfer_theory: ContrastTransferTheory,
        instrument_config: InstrumentConfig,
    ):
        self.instrument_config = instrument_config
        self.potential_integrator = potential_integrator
        self.structure = structure
        self.transfer_theory = transfer_theory

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
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
        fourier_image = self.transfer_theory.propagate_object_to_detector_plane(  # noqa: E501
            fourier_projection,
            self.instrument_config,
            is_projection_approximation=self.potential_integrator.is_projection_approximation,
            defocus_offset=self.structure.pose.offset_z_in_angstroms,
        )
        # Now for the in-plane translation
        fourier_image = self._apply_translation(fourier_image)

        return fourier_image
