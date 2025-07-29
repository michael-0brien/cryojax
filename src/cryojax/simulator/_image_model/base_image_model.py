"""
Image formation models.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray

from ...ndimage import irfftn, rfftn
from ...ndimage.transforms import FilterLike, MaskLike
from .._config import AbstractConfig
from .._direct_integrator import AbstractDirectIntegrator
from .._pose import AbstractPose
from .._structure_mapping import AbstractStructureMapping
from .._transfer_theory import ContrastTransferTheory


RealImageArray = Float[Array, "{self.config.y_dim} {self.config.x_dim}"]
FourierImageArray = Complex[Array, "{self.config.y_dim} {self.config.x_dim//2+1}"]
PaddedRealImageArray = Float[
    Array,
    "{self.config.padded_y_dim} " "{self.config.padded_x_dim}",
]
PaddedFourierImageArray = Complex[
    Array,
    "{self.config.padded_y_dim} " "{self.config.padded_x_dim//2+1}",
]

ImageArray = RealImageArray | FourierImageArray
PaddedImageArray = PaddedRealImageArray | PaddedFourierImageArray


class AbstractImageModel(eqx.Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractImageModel`'s `render` routine.
    """

    structure: eqx.AbstractVar[AbstractStructureMapping]
    pose: eqx.AbstractVar[AbstractPose]
    config: eqx.AbstractVar[AbstractConfig]

    applies_translation: eqx.AbstractVar[bool]
    normalizes_signal: eqx.AbstractVar[bool]
    signal_region: eqx.AbstractVar[Optional[Bool[Array, "_ _"]]]

    @abstractmethod
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        """Render an image without postprocessing."""
        raise NotImplementedError

    def simulate(
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
            If `True`, return an image cropped to `BasicConfig.shape`.
            Otherwise, return an image at the `BasicConfig.padded_shape`.
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
        fourier_image: PaddedFourierImageArray,
        *,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> ImageArray:
        """Return an image postprocessed with filters, cropping, masking,
        and normalization in either real or fourier space.
        """
        config = self.config
        if (
            mask is None
            and config.padded_shape == config.shape
            and not self.normalizes_signal
        ):
            # ... if there are no masks, we don't need to crop, and we are
            # not normalizing, minimize moving back and forth between real
            # and fourier space
            if filter is not None:
                fourier_image = filter(fourier_image)
            return (
                irfftn(fourier_image, s=config.shape)
                if outputs_real_space
                else fourier_image
            )
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            padded_rfft_shape = config.padded_frequency_grid_in_pixels.shape[0:2]
            if filter is not None:
                # ... apply the filter
                if not filter.array.shape == padded_rfft_shape:
                    raise ValueError(
                        "Found that the `filter` was shape "
                        f"{filter.array.shape}, but expected it to be "
                        f"shape {padded_rfft_shape}. You may have passed a "
                        f"fitler according to the `AbstractImageModel.config.shape`, "
                        "when the `AbstractImageModel.config.padded_shape` was expected."
                    )
                fourier_image = filter(fourier_image)
            image = irfftn(fourier_image, s=config.padded_shape)
            if config.padded_shape != config.shape:
                image = config.crop_to_shape(image)
            if self.normalizes_signal:
                image = self._normalize_image(image)
            if mask is not None:
                image = mask(image)
            return image if outputs_real_space else rfftn(image)

    def _apply_translation(
        self, fourier_image: PaddedFourierImageArray
    ) -> PaddedFourierImageArray:
        phase_shifts = self.pose.compute_translation_operator(
            self.config.padded_frequency_grid_in_angstroms
        )
        fourier_image = self.pose.translate_image(
            fourier_image,
            phase_shifts,
            self.config.padded_shape,
        )

        return fourier_image

    def _normalize_image(self, image: RealImageArray) -> RealImageArray:
        mean, std = (
            jnp.mean(image, where=self.signal_region),
            jnp.std(image, where=self.signal_region),
        )
        image = (image - mean) / std

        return image

    def _maybe_postprocess(
        self,
        image: PaddedFourierImageArray,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> PaddedImageArray | ImageArray:
        config = self.config
        if removes_padding:
            return self.postprocess(
                image, outputs_real_space=outputs_real_space, mask=mask, filter=filter
            )
        else:
            return irfftn(image, s=config.padded_shape) if outputs_real_space else image


class LinearImageModel(AbstractImageModel, strict=True):
    """An simple image model in linear image formation theory."""

    structure: AbstractStructureMapping
    pose: AbstractPose
    integrator: AbstractDirectIntegrator
    transfer_theory: ContrastTransferTheory
    config: AbstractConfig

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure: AbstractStructureMapping,
        pose: AbstractPose,
        config: AbstractConfig,
        integrator: AbstractDirectIntegrator,
        transfer_theory: ContrastTransferTheory,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure`:
            The map to a biological structure.
        - `pose`:
            The pose of a structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalizes_signal the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractConfig.shape`.
        """
        # Simulator components
        self.structure = structure
        self.pose = pose
        self.config = config
        self.integrator = integrator
        self.transfer_theory = transfer_theory
        # Options
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the structure
        structure = self.structure.map_to_structure()
        # Rotate it to the lab frame
        structure = structure.rotate_to_pose(self.pose)
        # Compute the projection image
        fourier_image = self.integrator.integrate(
            structure, self.config, outputs_real_space=False
        )
        # Compute the image
        fourier_image = self.transfer_theory.propagate_object(  # noqa: E501
            fourier_image,
            self.config,
            is_projection_approximation=self.integrator.is_projection_approximation,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        # Now for the in-plane translation
        if self.applies_translation:
            fourier_image = self._apply_translation(fourier_image)

        return fourier_image


class ProjectionImageModel(AbstractImageModel, strict=True):
    """An simple image model for computing a projection."""

    structure: AbstractStructureMapping
    pose: AbstractPose
    integrator: AbstractDirectIntegrator
    config: AbstractConfig

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Optional[Bool[Array, "_ _"]]

    def __init__(
        self,
        structure: AbstractStructureMapping,
        pose: AbstractPose,
        config: AbstractConfig,
        integrator: AbstractDirectIntegrator,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Optional[Bool[Array, "_ _"]] = None,
    ):
        """**Arguments:**

        - `structure`:
            The map to a biological structure.
        - `pose`:
            The pose of a structure.
        - `config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `integrator`: The method for integrating the scattering potential.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalizes_signal the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractConfig.shape`.
        """
        # Simulator components
        self.structure = structure
        self.pose = pose
        self.config = config
        self.integrator = integrator
        # Options
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        self.signal_region = signal_region

    @override
    def compute_fourier_image(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> ImageArray | PaddedImageArray:
        # Get the structure
        structure = self.structure.map_to_structure()
        # Rotate it to the lab frame
        structure = structure.rotate_to_pose(self.pose)
        # Compute the projection image
        fourier_image = self.integrator.integrate(
            structure, self.config, outputs_real_space=False
        )
        # Now for the in-plane translation
        if self.applies_translation:
            fourier_image = self._apply_translation(fourier_image)

        return fourier_image
