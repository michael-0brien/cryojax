"""
Image formation models.
"""

from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, PRNGKeyArray

from ...jax_util import NDArrayLike
from .._detector import AbstractDetector
from .._image_config import AbstractImageConfig, DoseImageConfig
from .._pose import AbstractPose
from .._scattering_theory import AbstractScatteringTheory
from .._volume import AbstractVolumeParametrization
from .base_image_model import AbstractImageModel, PaddedFourierImageArray


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

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Bool[Array, "_ _"] | None

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of the imaging volume.
        - `pose`:
            The pose of the volume.
        - `image_config`:
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
            Must have shape equal to `AbstractImageConfig.shape`.
        """
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.scattering_theory = scattering_theory
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
        # Get the volume representation. Its data should be a scattering potential
        # to simulate in physical units
        if rng_key is None:
            volume_representation = self.volume_parametrization.compute_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.compute_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Compute the contrast
        contrast_spectrum = self.scattering_theory.compute_contrast_spectrum(
            volume_representation,
            self.image_config,
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

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    image_config: AbstractImageConfig
    scattering_theory: AbstractScatteringTheory

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Bool[Array, "_ _"] | None

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: AbstractImageConfig,
        scattering_theory: AbstractScatteringTheory,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of the imaging volume.
        - `pose`:
            The pose of the volume.
        - `image_config`:
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
            Must have shape equal to `AbstractImageConfig.shape`.
        """
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.scattering_theory = scattering_theory
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
        # Get the volume representation. Its data should be a scattering potential
        # to simulate in physical units
        if rng_key is None:
            volume_representation = self.volume_parametrization.compute_representation()
        else:
            this_key, rng_key = jr.split(rng_key)
            volume_representation = self.volume_parametrization.compute_representation(
                rng_key=this_key
            )
        # Rotate it to the lab frame
        volume_representation = volume_representation.rotate_to_pose(self.pose)
        # Compute the intensity spectrum
        intensity_spectrum = self.scattering_theory.compute_intensity_spectrum(
            volume_representation,
            self.image_config,
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

    volume_parametrization: AbstractVolumeParametrization
    pose: AbstractPose
    image_config: DoseImageConfig
    scattering_theory: AbstractScatteringTheory
    detector: AbstractDetector

    applies_translation: bool
    normalizes_signal: bool
    signal_region: Bool[Array, "_ _"] | None

    def __init__(
        self,
        volume_parametrization: AbstractVolumeParametrization,
        pose: AbstractPose,
        image_config: DoseImageConfig,
        scattering_theory: AbstractScatteringTheory,
        detector: AbstractDetector,
        *,
        applies_translation: bool = True,
        normalizes_signal: bool = False,
        signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    ):
        """**Arguments:**

        - `volume_parametrization`:
            The parametrization of the imaging volume.
        - `pose`:
            The pose of the volume.
        - `image_config`:
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
            Must have shape equal to `AbstractImageConfig.shape`.
        """
        self.volume_parametrization = volume_parametrization
        self.pose = pose
        self.image_config = image_config
        self.scattering_theory = scattering_theory
        self.detector = detector
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
    def get_image_config(self) -> DoseImageConfig:
        return self.image_config

    @override
    def get_signal_region(self) -> Bool[Array, "_ _"] | None:
        return self.signal_region

    @override
    def compute_fourier_image(
        self, rng_key: PRNGKeyArray | None = None
    ) -> PaddedFourierImageArray:
        if rng_key is None:
            # Get the volume representation. Its data should be a scattering potential
            # to simulate in physical units
            volume_representation = self.volume_parametrization.compute_representation()
            # Rotate it to the lab frame
            volume_representation = volume_representation.rotate_to_pose(self.pose)
            # Compute the intensity
            fourier_intensity = self.scattering_theory.compute_intensity_spectrum(
                volume_representation,
                self.image_config,
                defocus_offset=self.pose.offset_z_in_angstroms,
            )
            if self.applies_translation:
                fourier_intensity = self._apply_translation(fourier_intensity)
            # ... now measure the expected electron events at the detector
            fourier_expected_electron_events = (
                self.detector.compute_expected_electron_events(
                    fourier_intensity, self.image_config
                )
            )

            return fourier_expected_electron_events
        else:
            keys = jr.split(rng_key, 3)
            # Get the volume representation. Its data should be a scattering potential
            # to simulate in physical units
            volume_representation = self.volume_parametrization.compute_representation(
                keys[0]
            )
            # Rotate it to the lab frame
            volume_representation = volume_representation.rotate_to_pose(self.pose)
            # Compute the squared wavefunction
            fourier_intensity = self.scattering_theory.compute_intensity_spectrum(
                volume_representation,
                self.image_config,
                keys[1],
                defocus_offset=self.pose.offset_z_in_angstroms,
            )
            if self.applies_translation:
                fourier_intensity = self._apply_translation(fourier_intensity)
            # ... now measure the detector readout
            fourier_detector_readout = self.detector.compute_detector_readout(
                keys[2],
                fourier_intensity,
                self.image_config,
            )

            return fourier_detector_readout
