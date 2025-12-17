"""
Image formation models.
"""

from abc import abstractmethod
from typing import Literal
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray

from ..jax_util import NDArrayLike
from ..ndimage import (
    AbstractFilter,
    AbstractImageTransform,
    AbstractMask,
    crop_to_shape,
    irfftn,
    rfftn,
)
from ._detector import AbstractDetector
from ._image_config import AbstractImageConfig, DoseImageConfig
from ._pose import AbstractPose
from ._scattering_theory import AbstractScatteringTheory
from ._transfer_theory import ContrastTransferTheory
from ._volume import (
    AbstractAtomVolume,
    AbstractVolumeIntegrator,
    AbstractVolumeParametrization,
    AbstractVolumeRepresentation,
    AutoVolumeProjection,
)


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

    image_config: eqx.AbstractVar[AbstractImageConfig]
    pose: eqx.AbstractVar[AbstractPose]

    transform: eqx.AbstractVar[AbstractImageTransform | None]

    signal_region: eqx.AbstractVar[Bool[Array, "_ _"] | None]
    normalize_mode: eqx.AbstractVar[Literal["bg_subtract", "standardize", "none"]]
    translate_mode: eqx.AbstractVar[Literal["fft", "atom", "none"]]

    def __check_init__(self):
        if self.normalize_mode not in ["bg_subtract", "standardize", "none"]:
            raise ValueError(
                "Found invalid value for "
                f"`{self.__class__.__name__}(..., normalize_mode=...)`. "
                "Values 'standardize', 'bg_subtract', and 'none' are "
                f"supported, but got {self.normalize_mode}."
            )
        if self.translate_mode not in ["fft", "atom", "none"]:
            raise ValueError(
                "Found invalid value for "
                f"`{self.__class__.__name__}(..., translate_mode=...)`. "
                "Values 'fft', 'atom', and 'none' are "
                f"supported, but got {self.translate_mode}."
            )

    @abstractmethod
    def raw_simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
    ) -> Array:
        """Render an image without postprocessing."""
        raise NotImplementedError

    def simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> Array:
        """Render an image.

        **Arguments:**

        - `rng_key`:
            The random number generator key. If not passed, render an image
            with no stochasticity.
        - `outputs_real_space`:
            If `True`, return the image in real space.
        - `mask`:
            Optionally apply a mask to the image.
        - `filter`:
            Optionally apply a filter to the image.
        """
        fourier_image = self.raw_simulate(rng_key, outputs_real_space=False)

        return self.postprocess(
            fourier_image,
            outputs_real_space=outputs_real_space,
            mask=mask,
            filter=filter,
            apply_transform=True,
        )

    def postprocess(
        self,
        fourier_image: Array,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
        apply_transform: bool = True,
    ) -> Array:
        """Return an image postprocessed with filters, cropping, masking,
        and normalization in either real or fourier space.
        """
        image_config = self.image_config
        if apply_transform:
            mask_c, filter_c = self._compose_transform(mask, filter)
        else:
            mask_c, filter_c = mask, filter
        if (
            mask_c is None
            and image_config.padded_shape == image_config.shape
            and self.normalize_mode == "none"
        ):
            # ... if there are no masks, we don't need to crop, and we are
            # not normalizing, minimize moving back and forth between real
            # and fourier space
            if filter_c is not None:
                fourier_image = filter_c(fourier_image)
            return (
                irfftn(fourier_image, s=image_config.shape)
                if outputs_real_space
                else fourier_image
            )
        else:
            # ... otherwise, apply filter, crop, and mask, again trying to
            # minimize moving back and forth between real and fourier space
            padded_rfft_shape = image_config.padded_frequency_grid_in_pixels.shape[0:2]
            if filter_c is not None:
                # ... apply the filter
                if filter is not None:
                    if not filter.get().shape == padded_rfft_shape:
                        raise ValueError(
                            "Found that the `filter` was shape "
                            f"{filter.get().shape}, but expected it to be "
                            f"shape {padded_rfft_shape}. You may have passed a "
                            f"filter according to the "
                            f"`{image_config.__class__.__name__}.shape`, "
                            f"when the `{image_config.__class__.__name__}.padded_shape` "
                            "was expected."
                        )
                fourier_image = filter_c(fourier_image)
            padded_image = irfftn(fourier_image, s=image_config.padded_shape)
            if image_config.padded_shape != image_config.shape:
                image = crop_to_shape(padded_image, image_config.shape)
            else:
                image = padded_image
            if self.normalize_mode == "standardize":
                image = self._standardize_signal(image)
            elif self.normalize_mode == "bg_subtract":
                image = self._bg_subtract_signal(image, padded_image)
            if mask_c is not None:
                image = mask_c(image)
            return image if outputs_real_space else rfftn(image)

    def _phase_shift_translate(self, fourier_image: Array) -> Array:
        phase_shifts = self.pose.compute_translation_operator(
            self.image_config.padded_frequency_grid_in_angstroms
        )
        fourier_image = self.pose.translate_image(
            fourier_image,
            phase_shifts,
            self.image_config.padded_shape,
        )

        return fourier_image

    def _atom_translate(self, volrep: AbstractVolumeRepresentation) -> AbstractAtomVolume:
        if isinstance(volrep, AbstractAtomVolume):
            return volrep.translate_to_pose(self.pose)
        else:
            raise ValueError(
                "Tried to apply translation in `translate_mode = 'atom'`, but "
                "found a volume representation that was not an `AbstractAtomVolume`."
                f"Got a `{volrep.__class__.__name__}` class."
            )

    def _compose_transform(
        self, mask: AbstractMask | None, filter: AbstractFilter | None
    ) -> tuple[AbstractImageTransform | None, AbstractImageTransform | None]:
        if self.transform is None:
            return mask, filter
        else:
            if self.transform.is_real_space:
                if mask is None:
                    return self.transform, filter
                else:
                    return mask * self.transform, filter
            else:
                if filter is None:
                    return mask, self.transform
                else:
                    return mask, filter * self.transform

    def _standardize_signal(self, image: Array) -> Array:
        mean, std = (
            jnp.mean(image, where=self.signal_region),
            jnp.std(image, where=self.signal_region),
        )
        image = (image - mean) / std

        return image

    def _bg_subtract_signal(self, image: Array, padded_image: Array) -> Array:
        edge_values = jnp.concatenate(
            (
                padded_image[0, :],
                padded_image[-1, :],
                padded_image[1:-1, 0],
                padded_image[1:-1, -1],
            ),
            axis=0,
        )
        bg_value, std = jnp.median(edge_values), jnp.std(image, where=self.signal_region)
        image = (image - bg_value) / std

        return image


class LinearImageModel(AbstractImageModel, strict=True):
    """An simple image model in linear image formation theory."""

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    volume_integrator: AbstractVolumeIntegrator
    transfer_theory: ContrastTransferTheory
    image_config: AbstractImageConfig

    transform: AbstractImageTransform | None
    signal_region: Bool[Array, "_ _"] | None
    normalize_mode: Literal["standardize", "bg_subtract", "none"]
    translate_mode: Literal["fft", "atom", "none"]

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        transfer_theory: ContrastTransferTheory,
        volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
        *,
        transform: AbstractImageTransform | None = None,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
        normalize_mode: Literal["standardize", "bg_subtract", "none"] = "none",
        translate_mode: Literal["fft", "atom", "none"] = "fft",
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of an imaging volume.
        - `pose`:
            The pose of the volume.
        - `image_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `volume_integrator`: The method for integrating the volume onto the plane.
        - `transfer_theory`: The contrast transfer theory.
        - `transform`:
            A [`cryojax.ndimage.AbstractImageTransform`][] applied to the
            image after simulation.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageConfig.shape`.
        - `normalize_mode`:
            How to normalize the image. Options are
            - 'standardize':
                Normalize the image to be mean 0 and
                standard deviation 1 within `signal_region`.
            - 'bg_subtract':
                Subtract mean value at image edges and divide by
                the image standard deviation within `signal_region`.
                This makes the image fades to a background with values
                equal to zero. Requires that `image_config.padded_shape`
                is large enough so that the signal sufficiently decays.
            - 'none':
                Do not normalize the image.
        - `translate_mode`:
            How to apply in-plane translation to the volume. Options are
            - 'fft':
                Apply phase shifts in the Fourier domain.
            - 'atom':
                Apply translation to atom positions before
                projection. For this method, the
                [`cryojax.simulator.AbstractVolumeParametrization`][]
                must be or return an [`cryojax.simulator.AbstractAtomVolume`][].
            - 'none':
                Do not apply the translation.
        """
        # Simulator components
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.volume_integrator = volume_integrator
        self.transfer_theory = transfer_theory
        # Options
        self.transform = transform
        self.translate_mode = translate_mode
        self.normalize_mode = normalize_mode
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def raw_simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
    ) -> PaddedFourierImageArray:
        # Get the representation of the volume
        if rng_key is None:
            volume_representation = self.volume_parametrization.to_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.to_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Translate if using atom translations
        if self.translate_mode == "atom":
            volume_representation = self._atom_translate(volume_representation)
        # Compute the projection image
        fourier_image = self.volume_integrator.integrate(
            volume_representation, self.image_config, outputs_real_space=False
        )
        # Compute the image
        fourier_image = self.transfer_theory.propagate_object(  # noqa: E501
            fourier_image,
            self.image_config,
            input_is_ewald_sphere=self.volume_integrator.outputs_ewald_sphere,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        # Now for the in-plane translation if using phase shifts
        if self.translate_mode == "fft":
            fourier_image = self._phase_shift_translate(fourier_image)

        return (
            irfftn(fourier_image, s=self.image_config.padded_shape)
            if outputs_real_space
            else fourier_image
        )


class ProjectionImageModel(AbstractImageModel, strict=True):
    """An simple image model for computing a projection."""

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    volume_integrator: AbstractVolumeIntegrator
    image_config: AbstractImageConfig

    transform: AbstractImageTransform | None
    signal_region: Bool[Array, "_ _"] | None
    normalize_mode: Literal["standardize", "bg_subtract", "none"]
    translate_mode: Literal["fft", "atom", "none"]

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
        *,
        transform: AbstractImageTransform | None = None,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
        normalize_mode: Literal["standardize", "bg_subtract", "none"] = "none",
        translate_mode: Literal["fft", "atom", "none"] = "fft",
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of the imaging volume
        - `pose`:
            The pose of the volume.
        - `image_config`:
            The configuration of the instrument, such as for the pixel size
            and the wavelength.
        - `volume_integrator`: The method for integrating the volume onto the plane.
        - `transform`:
            A [`cryojax.ndimage.AbstractImageTransform`][] applied to the
            image after simulation.
        - `signal_region`:
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractImageConfig.shape`.
        - `normalize_mode`:
            How to normalize the image. Options are
            - 'standardize':
                Normalize the image to be mean 0 and
                standard deviation 1 within `signal_region`.
            - 'bg_subtract':
                Subtract mean value at image edges and divide by
                the image standard deviation within `signal_region`.
                This makes the image fades to a background with values
                equal to zero. Requires that `image_config.padded_shape`
                is large enough so that the signal sufficiently decays.
            - 'none':
                Do not normalize the image.
        - `translate_mode`:
            How to apply in-plane translation to the volume. Options are
            - 'fft':
                Apply phase shifts in the Fourier domain.
            - 'atom':
                Apply translation to atom positions before
                projection. For this method, the
                [`cryojax.simulator.AbstractVolumeParametrization`][]
                must be or return an [`cryojax.simulator.AbstractAtomVolume`][].
            - 'none':
                Do not apply the translation.
        """
        # Simulator components
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.volume_integrator = volume_integrator
        # Options
        self.transform = transform
        self.translate_mode = translate_mode
        self.normalize_mode = normalize_mode
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def raw_simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
    ) -> ImageArray | PaddedImageArray:
        # Get the representation of the volume
        if rng_key is None:
            volume_representation = self.volume_parametrization.to_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.to_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Translate if using atom translations
        if self.translate_mode == "atom":
            volume_representation = self._atom_translate(volume_representation)
        # Compute the projection image
        fourier_image = self.volume_integrator.integrate(
            volume_representation, self.image_config, outputs_real_space=False
        )
        # Now for the in-plane translation
        if self.translate_mode == "fft":
            fourier_image = self._phase_shift_translate(fourier_image)

        return (
            irfftn(fourier_image, s=self.image_config.padded_shape)
            if outputs_real_space
            else fourier_image
        )


class AbstractPhysicalImageModel(AbstractImageModel, strict=True):
    """An image formation model that simulates physical
    quantities. This uses the `AbstractScatteringTheory` class.
    """

    scattering_theory: eqx.AbstractVar[AbstractScatteringTheory]


class ContrastImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns the image contrast from a linear
    scattering theory.
    """

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    image_config: AbstractImageConfig
    scattering_theory: AbstractScatteringTheory

    transform: AbstractImageTransform | None
    signal_region: Bool[Array, "_ _"] | None
    normalize_mode: Literal["standardize", "bg_subtract", "none"]
    translate_mode: Literal["fft", "atom", "none"]

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        transform: AbstractImageTransform | None = None,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
        normalize_mode: Literal["standardize", "bg_subtract", "none"] = "none",
        translate_mode: Literal["fft", "atom", "none"] = "fft",
    ):
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.scattering_theory = scattering_theory
        self.transform = transform
        self.translate_mode = translate_mode
        self.normalize_mode = normalize_mode
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def raw_simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
    ) -> PaddedFourierImageArray:
        # Get the volume representation. Its data should be a scattering potential
        # to simulate in physical units
        if rng_key is None:
            volume_representation = self.volume_parametrization.to_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.to_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Translate if using atom translations
        if self.translate_mode == "atom":
            volume_representation = self._atom_translate(volume_representation)
        # Compute the contrast
        contrast_spectrum = self.scattering_theory.compute_contrast_spectrum(
            volume_representation,
            self.image_config,
            rng_key,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        # Apply the translation
        if self.translate_mode == "fft":
            contrast_spectrum = self._phase_shift_translate(contrast_spectrum)

        return (
            irfftn(contrast_spectrum, s=self.image_config.padded_shape)
            if outputs_real_space
            else contrast_spectrum
        )


class IntensityImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns an intensity distribution---or in other
    words a squared wavefunction.
    """

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    image_config: AbstractImageConfig
    scattering_theory: AbstractScatteringTheory

    transform: AbstractImageTransform | None
    signal_region: Bool[Array, "_ _"] | None
    normalize_mode: Literal["standardize", "bg_subtract", "none"]
    translate_mode: Literal["fft", "atom", "none"]

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        transform: AbstractImageTransform | None = None,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
        normalize_mode: Literal["standardize", "bg_subtract", "none"] = "none",
        translate_mode: Literal["fft", "atom", "none"] = "fft",
    ):
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.scattering_theory = scattering_theory
        self.transform = transform
        self.translate_mode = translate_mode
        self.normalize_mode = normalize_mode
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def raw_simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
    ) -> PaddedFourierImageArray:
        # Get the volume representation. Its data should be a scattering potential
        # to simulate in physical units
        if rng_key is None:
            volume_representation = self.volume_parametrization.to_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.to_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Translate if using atom translations
        if self.translate_mode == "atom":
            volume_representation = self._atom_translate(volume_representation)
        # Compute the intensity spectrum
        intensity_spectrum = self.scattering_theory.compute_intensity_spectrum(
            volume_representation,
            self.image_config,
            rng_key,
            defocus_offset=self.pose.offset_z_in_angstroms,
        )
        if self.translate_mode == "fft":
            intensity_spectrum = self._phase_shift_translate(intensity_spectrum)

        return (
            irfftn(intensity_spectrum, s=self.image_config.padded_shape)
            if outputs_real_space
            else intensity_spectrum
        )


class ElectronCountsImageModel(AbstractPhysicalImageModel, strict=True):
    """An image formation model that returns electron counts, given a
    model for the detector.
    """

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    image_config: DoseImageConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    transform: AbstractImageTransform | None
    signal_region: Bool[Array, "_ _"] | None
    normalize_mode: Literal["standardize", "bg_subtract", "none"]
    translate_mode: Literal["fft", "atom", "none"]

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: DoseImageConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        transform: AbstractImageTransform | None = None,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
        normalize_mode: Literal["standardize", "bg_subtract", "none"] = "none",
        translate_mode: Literal["fft", "atom", "none"] = "fft",
    ):
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.scattering_theory = scattering_theory
        self.detector = detector
        self.transform = transform
        self.translate_mode = translate_mode
        self.normalize_mode = normalize_mode
        if signal_region is None:
            self.signal_region = None
        else:
            self.signal_region = jnp.asarray(signal_region, dtype=bool)

    @override
    def raw_simulate(
        self,
        rng_key: PRNGKeyArray | None = None,
        *,
        outputs_real_space: bool = True,
    ) -> PaddedFourierImageArray:
        if rng_key is None:
            # Get the volume representation. Its data should be a scattering potential
            # to simulate in physical units
            volume_representation = self.volume_parametrization.to_representation()
            # Rotate it to the lab frame
            volume_representation = volume_representation.rotate_to_pose(self.pose)
            # Translate if using atom translations
            if self.translate_mode == "atom":
                volume_representation = self._atom_translate(volume_representation)
            # Compute the intensity
            fourier_intensity = self.scattering_theory.compute_intensity_spectrum(
                volume_representation,
                self.image_config,
                defocus_offset=self.pose.offset_z_in_angstroms,
            )
            if self.translate_mode == "fft":
                fourier_intensity = self._phase_shift_translate(fourier_intensity)
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.detector.compute_expected_electron_events(
                    fourier_intensity, self.image_config
                )
            )

            return (
                irfftn(fourier_expected_electron_events, s=self.image_config.padded_shape)
                if outputs_real_space
                else fourier_expected_electron_events
            )
        else:
            keys = jr.split(rng_key, 3)
            # Get the volume representation. Its data should be a scattering potential
            # to simulate in physical units
            volume_representation = self.volume_parametrization.to_representation(keys[0])
            # Rotate it to the lab frame
            volume_representation = volume_representation.rotate_to_pose(self.pose)
            # Translate if using atom translations
            if self.translate_mode == "atom":
                volume_representation = self._atom_translate(volume_representation)
            # Compute the squared wavefunction
            fourier_intensity = self.scattering_theory.compute_intensity_spectrum(
                volume_representation,
                self.image_config,
                keys[1],
                defocus_offset=self.pose.offset_z_in_angstroms,
            )
            if self.translate_mode == "fft":
                fourier_intensity = self._phase_shift_translate(fourier_intensity)
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[2],
                fourier_intensity,
                self.image_config,
            )

            return (
                irfftn(fourier_detector_readout, s=self.image_config.padded_shape)
                if outputs_real_space
                else fourier_detector_readout
            )


_init_doc = """**Arguments:**

- `volume_parametrization`:
    The parametrization of the imaging volume.
- `pose`:
    The pose of the volume.
- `image_config`:
    The configuration of the instrument, such as for the pixel size
    and the wavelength.
- `scattering_theory`:
    The scattering theory.
- `transform`:
    A [`cryojax.ndimage.AbstractImageTransform`][] applied to the
    image after simulation.
- `signal_region`:
    A boolean array that is 1 where there is signal,
    and 0 otherwise used to normalize the image.
    Must have shape equal to `AbstractImageConfig.shape`.
- `normalize_mode`:
    How to normalize the image. Options are
    - 'standardize':
        Normalize the image to be mean 0 and
        standard deviation 1 within `signal_region`.
    - 'bg_subtract':
        Subtract mean value at image edges and divide by
        the image standard deviation within `signal_region`.
        This makes the image fades to a background with values
        equal to zero. Requires that `image_config.padded_shape`
        is large enough so that the signal sufficiently decays.
    - 'none':
        Do not normalize the image.
- `translate_mode`:
    How to apply in-plane translation to the volume. Options are
    - 'fft':
        Apply phase shifts in the Fourier domain.
    - 'atom':
        Apply translation to atom positions before
        projection. For this method, the
        [`cryojax.simulator.AbstractVolumeParametrization`][]
        must be or return an [`cryojax.simulator.AbstractAtomVolume`][].
    - 'none':
        Do not apply the translation.
"""

ContrastImageModel.__init__.__doc__ = _init_doc
IntensityImageModel.__init__.__doc__ = _init_doc
ElectronCountsImageModel.__init__.__doc__ = _init_doc
