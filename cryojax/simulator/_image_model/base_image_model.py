"""
Image formation models.
"""

from abc import abstractmethod
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray

from ...jax_util import NDArrayLike
from ...ndimage import irfftn, rfftn
from ...ndimage.transforms import FilterLike, MaskLike
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .._transfer_theory import ContrastTransferTheory
from .._volume import AbstractVolumeParametrization
from .._volume_integrator import AbstractVolumeIntegrator


RealImageArray = Float[Array, "{self.image_config.y_dim} {self.image_config.x_dim}"]
FourierImageArray = Complex[
    Array, "{self.image_config.y_dim} {self.image_config.x_dim//2+1}"
]
PaddedRealImageArray = Float[
    Array,
    "{self.image_config.padded_y_dim} {self.image_config.padded_x_dim}",
]
PaddedFourierImageArray = Complex[
    Array,
    "{self.image_config.padded_y_dim} {self.image_config.padded_x_dim//2+1}",
]

ImageArray = RealImageArray | FourierImageArray
PaddedImageArray = PaddedRealImageArray | PaddedFourierImageArray


class AbstractImageModel(eqx.Module, strict=True):
    """Base class for an image formation model.

    Call an `AbstractImageModel`'s `simulate` routine.
    """

    applies_translation: eqx.AbstractVar[bool]
    normalizes_signal: eqx.AbstractVar[bool]

    @abstractmethod
    def get_pose(self) -> AbstractPose:
        raise NotImplementedError

    @abstractmethod
    def get_image_config(self) -> AbstractImageConfig:
        raise NotImplementedError

    @abstractmethod
    def get_signal_region(self) -> Bool[Array, "_ _"] | None:
        raise NotImplementedError

    @abstractmethod
    def compute_fourier_image(self, rng_key: PRNGKeyArray | None = None) -> Array:
        """Render an image without postprocessing."""
        raise NotImplementedError

    def simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        mask: MaskLike | None = None,
        filter: FilterLike | None = None,
    ) -> Array:
        """Render an image.

        **Arguments:**

        - `rng_key`:
            The random number generator key. If not passed, render an image
            with no stochasticity.
        - `removes_padding`:
            If `True`, return an image cropped to `BasicImageConfig.shape`.
            Otherwise, return an image at the `BasicImageConfig.padded_shape`.
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
        fourier_image: Array,
        *,
        outputs_real_space: bool = True,
        mask: MaskLike | None = None,
        filter: FilterLike | None = None,
    ) -> Array:
        """Return an image postprocessed with filters, cropping, masking,
        and normalization in either real or fourier space.
        """
        image_config = self.get_image_config()
        if (
            mask is None
            and image_config.padded_shape == image_config.shape
            and not self.normalizes_signal
        ):
            # ... if there are no masks, we don't need to crop, and we are
            # not normalizing, minimize moving back and forth between real
            # and fourier space
            if filter is not None:
                fourier_image = filter(fourier_image)
            return (
                irfftn(fourier_image, s=image_config.shape)
                if outputs_real_space
                else fourier_image
            )
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            padded_rfft_shape = image_config.padded_frequency_grid_in_pixels.shape[0:2]
            if filter is not None:
                # ... apply the filter
                if not filter.array.shape == padded_rfft_shape:
                    raise ValueError(
                        "Found that the `filter` was shape "
                        f"{filter.array.shape}, but expected it to be "
                        f"shape {padded_rfft_shape}. You may have passed a "
                        f"fitler according to the "
                        "`AbstractImageModel.image_config.shape`, "
                        "when the `AbstractImageModel.image_config.padded_shape` "
                        "was expected."
                    )
                fourier_image = filter(fourier_image)
            image = irfftn(fourier_image, s=image_config.padded_shape)
            if image_config.padded_shape != image_config.shape:
                image = image_config.crop_to_shape(image)
            if self.normalizes_signal:
                image = self._normalize_image(image)
            if mask is not None:
                image = mask(image)
            return image if outputs_real_space else rfftn(image)

    def _apply_translation(self, fourier_image: Array) -> Array:
        pose, image_config = self.get_pose(), self.get_image_config()
        phase_shifts = pose.compute_translation_operator(
            image_config.padded_frequency_grid_in_angstroms
        )
        fourier_image = pose.translate_image(
            fourier_image,
            phase_shifts,
            image_config.padded_shape,
        )

        return fourier_image

    def _normalize_image(self, image: Array) -> Array:
        signal_region = self.get_signal_region()
        mean, std = (
            jnp.mean(image, where=signal_region),
            jnp.std(image, where=signal_region),
        )
        image = (image - mean) / std

        return image

    def _maybe_postprocess(
        self,
        image: Array,
        *,
        removes_padding: bool = True,
        outputs_real_space: bool = True,
        mask: MaskLike | None = None,
        filter: FilterLike | None = None,
    ) -> Array:
        image_config = self.get_image_config()
        if removes_padding:
            return self.postprocess(
                image, outputs_real_space=outputs_real_space, mask=mask, filter=filter
            )
        else:
            return (
                irfftn(image, s=image_config.padded_shape)
                if outputs_real_space
                else image
            )


class LinearImageModel(AbstractImageModel, strict=True):
    """An simple image model in linear image formation theory."""

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    volume_integrator: AbstractVolumeIntegrator
    transfer_theory: ContrastTransferTheory
    image_config: AbstractImageConfig

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Bool[Array, "_ _"] | None

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        volume_integrator: AbstractVolumeIntegrator,
        transfer_theory: ContrastTransferTheory,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of an imaging volume.
        - `pose`:
            The pose of the volume.
        - `image_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `volume_integrator`: The method for integrating the scattering potential.
        - `transfer_theory`: The contrast transfer theory.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalizes_signal the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageConfig.shape`.
        """
        # Simulator components
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory
        # Options
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def get_pose(self) -> AbstractPose:
        return self.pose

    @override
    def get_image_config(self) -> AbstractImageConfig:
        return self.image_config

    @override
    def get_signal_region(self) -> Bool[Array, "_ _"] | None:
        return self.signal_region

    @override
    def compute_fourier_image(
        self, rng_key: PRNGKeyArray | None = None
    ) -> PaddedFourierImageArray:
        # Get the representation of the volume
        if rng_key is None:
            volume_representation = self.volume_parametrization.compute_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.compute_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Compute the projection image
        fourier_image = self.volume_integrator.integrate(
            volume_representation, self.image_config, outputs_real_space=False
        )
        # Compute the image
        fourier_image = self.transfer_theory.propagate_object(  # noqa: E501
            fourier_image,
            self.image_config,
            is_projection_approximation=self.volume_integrator.is_projection_approximation,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        # Now for the in-plane translation
        if self.applies_translation:
            fourier_image = self._apply_translation(fourier_image)

        return fourier_image


class ProjectionImageModel(AbstractImageModel, strict=True):
    """An simple image model for computing a projection."""

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    volume_integrator: AbstractVolumeIntegrator
    image_config: AbstractImageConfig

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Bool[Array, "_ _"] | None

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        volume_integrator: AbstractVolumeIntegrator,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of the imaging volume
        - `pose`:
            The pose of the volume.
        - `image_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `volume_integrator`: The method for integrating the scattering potential.
        - `applies_translation`:
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - `normalizes_signal`:
            If `True`, normalizes_signal the image before returning.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageConfig.shape`.
        """
        # Simulator components
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.volume_integrator = volume_integrator
        # Options
        self.applies_translation = applies_translation
        self.normalizes_signal = normalizes_signal
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def get_pose(self) -> AbstractPose:
        return self.pose

    @override
    def get_image_config(self) -> AbstractImageConfig:
        return self.image_config

    @override
    def get_signal_region(self) -> Bool[Array, "_ _"] | None:
        return self.signal_region

    @override
    def compute_fourier_image(
        self, rng_key: PRNGKeyArray | None = None
    ) -> ImageArray | PaddedImageArray:
        # Get the representation of the volume
        if rng_key is None:
            volume_representation = self.volume_parametrization.compute_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.compute_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Compute the projection image
        fourier_image = self.volume_integrator.integrate(
            volume_representation, self.image_config, outputs_real_space=False
        )
        # Now for the in-plane translation
        if self.applies_translation:
            fourier_image = self._apply_translation(fourier_image)

        return fourier_image
