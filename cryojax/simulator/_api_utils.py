from typing import Literal, overload

from jaxtyping import Bool

from ..jax_util import NDArrayLike
from ._detector import AbstractDetector
from ._image_config import AbstractImageConfig, DoseImageConfig
from ._image_model import (
    AbstractImageModel,
    ContrastImageModel,
    ElectronCountsImageModel,
    IntensityImageModel,
    LinearImageModel,
    ProjectionImageModel as ProjectionImageModel,
)
from ._pose import AbstractPose
from ._scattering_theory import WeakPhaseScatteringTheory
from ._transfer_theory import ContrastTransferTheory
from ._volume import (
    AbstractVolumeIntegrator,
    AbstractVolumeParametrization,
    AutoVolumeProjection,
)


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: None = None,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> ProjectionImageModel: ...


@overload
def make_image_model(  # pyright: ignore[reportOverlappingOverload]
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> LinearImageModel: ...


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["contrast"] = "contrast",
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> ContrastImageModel: ...


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["intensity"] = "intensity",
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> IntensityImageModel: ...


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["counts"] = "counts",
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> ElectronCountsImageModel: ...


def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory | None = None,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["contrast", "intensity", "counts"] | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> AbstractImageModel:
    """Construct an `AbstractImageModel` for most common use-cases.

    **Arguments:**

    - `volume_parametrization`:
        The representation of the protein volume.
        Common choices are the `FourierVoxelGridVolume`
        for fourier-space voxel grids or the `GaussianMixtureVolume`.
    - `image_config`:
        The configuration for the image and imagining instrument. Unless using
        a model that uses the electron dose as a parameter, choose the
        `BasicImageConfig`. Otherwise, choose the `DoseImageConfig`.
    - `pose`:
        The pose in a particular parameterization convention. Common options
        are the `EulerAnglePose`, `QuaternionPose`, or `AxisAnglePose`.
    - `transfer_theory`:
        The contrast transfer function and its theory for how it is applied
        to the image.
    - `volume_integrator`:
        Optionally pass the method for integrating the electrostatic potential onto
        the plane (e.g. projection via fourier slice extraction). If not provided,
        a default option is chosen.
    - `detector`:
        If `quantity_mode = 'counts'` is chosen, then an `AbstractDetector` class must be
        chosen to simulate electron counts.
    - `normalizes_signal`:
        If `True`, normalizes_signal the image before returning.
    - `signal_region`:
        A boolean array that is 1 where there is signal,
        and 0 otherwise used to normalize the image.
        Must have shape equal to `AbstractImageConfig.shape`.
    - `quantity_mode`:
        The physical observable to simulate. If `None`, simulate without scaling
        to physical units using the `LinearImageModel`.
        Options are
        - 'contrast':
            Uses the `ContrastImageModel` to simulate contrast. This is
            default.
        - 'intensity':
            Uses the `IntensityImageModel` to simulate intensity.
        - 'counts':
            Uses the `ElectronCountsImageModel` to simulate electron counts.
            If this is passed, a `detector` must also be passed.
    - `translate_mode`:
        If `'fft'`, apply in-plane translation via phase
        shifts in the Fourier domain. If `'atoms'` apply
        translation on atom positions before projection.
        If `'none'`, does not apply a translation.

    **Returns:**

    An `AbstractImageModel`. Simulate an image with syntax

    ```python
    image_model = make_image_model(...)
    image = image_model.simulate()
    ```
    """
    if transfer_theory is None:
        # Image model for projections
        image_model = ProjectionImageModel(
            volume_parametrization,
            pose,
            image_config,
            volume_integrator,
            normalizes_signal=normalizes_signal,
            signal_region=signal_region,
            translate_mode=translate_mode,
        )
    else:
        # Simulate physical observables
        if quantity_mode is not None:
            scattering_theory = WeakPhaseScatteringTheory(
                volume_integrator, transfer_theory
            )
            if quantity_mode == "counts":
                if not isinstance(image_config, DoseImageConfig):
                    raise ValueError(
                        "If using `quantity_mode = 'counts'` to simulate electron "
                        "counts, pass `image_config = DoseImageConfig(...)`. Got config "
                        f"{type(image_config).__name__}."
                    )
                if detector is None:
                    raise ValueError(
                        "If using `quantity_mode = 'counts'` to simulate electron "
                        "counts, an `AbstractDetector` must be passed."
                    )
                image_model = ElectronCountsImageModel(
                    volume_parametrization,
                    pose,
                    image_config,
                    scattering_theory,
                    detector,
                    normalizes_signal=normalizes_signal,
                    signal_region=signal_region,
                    translate_mode=translate_mode,
                )
            elif quantity_mode == "contrast":
                image_model = ContrastImageModel(
                    volume_parametrization,
                    pose,
                    image_config,
                    scattering_theory,
                    normalizes_signal=normalizes_signal,
                    signal_region=signal_region,
                    translate_mode=translate_mode,
                )
            elif quantity_mode == "intensity":
                image_model = IntensityImageModel(
                    volume_parametrization,
                    pose,
                    image_config,
                    scattering_theory,
                    normalizes_signal=normalizes_signal,
                    signal_region=signal_region,
                    translate_mode=translate_mode,
                )
            else:
                raise ValueError(
                    f"`quantity_mode = {quantity_mode}` not supported. Supported "
                    "modes for simulating "
                    "physical quantities are 'contrast', 'intensity', and 'counts'."
                )
        else:
            # Linear image model
            image_model = LinearImageModel(
                volume_parametrization,
                pose,
                image_config,
                volume_integrator,
                transfer_theory,
                normalizes_signal=normalizes_signal,
                signal_region=signal_region,
                translate_mode=translate_mode,
            )

    return image_model
