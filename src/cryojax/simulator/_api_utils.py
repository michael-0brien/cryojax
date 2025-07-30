from typing import Literal, Optional

from jaxtyping import Bool

from ..internal import NDArrayLike
from ._config import AbstractConfig, DoseConfig
from ._detector import AbstractDetector
from ._direct_integrator import (
    AbstractDirectIntegrator,
    FourierSliceExtraction,
    GaussianMixtureProjection,
    NufftProjection,
)
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
from ._structure_parametrisation import (
    AbstractStructureParameterisation,
    FourierVoxelGridVolume,
    FourierVoxelSplineVolume,
    GaussianMixtureVolume,
    PengIndependentAtomVolume,
    RealVoxelGridVolume,
)
from ._transfer_theory import ContrastTransferTheory


def make_image_model(
    structure: AbstractStructureParameterisation,
    config: AbstractConfig,
    pose: AbstractPose,
    transfer_theory: Optional[ContrastTransferTheory] = None,
    integrator: Optional[AbstractDirectIntegrator] = None,
    detector: Optional[AbstractDetector] = None,
    *,
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Optional[Bool[NDArrayLike, "_ _"]] = None,
    simulates_quantity: bool = False,
    quantity_mode: Literal["contrast", "intensity", "counts"] = "contrast",
) -> AbstractImageModel:
    """Construct an `AbstractImageModel` for most common use-cases.

    **Arguments:**

    - `structure`:
        The representation of the protein structure.
        Common choices are the `FourierVoxelGridVolume`
        for fourier-space voxel grids or the `PengIndependentAtomVolume`
        for gaussian mixtures of atoms parameterized by electron scattering factors.
    - `config`:
        The configuration for the image and imagining instrument. Unless using
        a model that uses the electron dose as a parameter, choose the
        `InstrumentConfig`. Otherwise, choose the `DoseConfig`.
    - `pose`:
        The pose in a particular parameterization convention. Common options
        are the `EulerAnglePose`, `QuaternionPose`, or `AxisAnglePose`.
    - `transfer_theory`:
        The contrast transfer function and its theory for how it is applied
        to the image.
    - `integrator`:
        Optionally pass the method for integrating the electrostatic potential onto
        the plane (e.g. projection via fourier slice extraction). If not provided,
        a default option is chosen.
    - `detector`:
        If `quantity_mode = 'counts'` is chosen, then an `AbstractDetector` class must be
        chosen to simulate electron counts.
    - `applies_translation`:
        If `True`, apply the in-plane translation in the `AbstractPose`
        via phase shifts in fourier space.
    - `normalizes_signal`:
        If `True`, normalizes_signal the image before returning.
    - `signal_region`:
        A boolean array that is 1 where there is signal,
        and 0 otherwise used to normalize the image.
        Must have shape equal to `AbstractConfig.shape`.
    - `simulates_quantity`:
        If `True`, the image simulated is a physical quantity, which is
        chosen with the `quantity_mode` argument. Otherwise, simulate an image without
        scaling to absolute units.
    - `quantity_mode`:
        The physical observable to simulate. Not used if `simulates_quantity = False`.
        Options are
        - 'contrast':
            Uses the `ContrastImageModel` to simulate contrast. This is
            default.
        - 'intensity':
            Uses the `IntensityImageModel` to simulate intensity.
        - 'counts':
            Uses the `ElectronCountsImageModel` to simulate electron counts.
            If this is passed, a `detector` must also be passed.

    **Returns:**

    An `AbstractImageModel`. Simulate an image with syntax

    ```python
    image_model = make_image_model(...)
    image = image_model.simulate()
    ```
    """
    # Select default integrator
    integrator = _select_default_integrator(structure, simulates_quantity)
    if transfer_theory is None:
        # Image model for projections
        image_model = ProjectionImageModel(
            structure,
            pose,
            config,
            integrator,
            applies_translation=applies_translation,
            normalizes_signal=normalizes_signal,
            signal_region=signal_region,
        )
    else:
        # Simulate physical observables
        if simulates_quantity:
            scattering_theory = WeakPhaseScatteringTheory(integrator, transfer_theory)
            if quantity_mode == "counts":
                if not isinstance(config, DoseConfig):
                    raise ValueError(
                        "If using `quantity_mode = 'counts'` to simulate electron "
                        "counts, pass `config = DoseConfig(...)`. Got config "
                        f"{type(config).__name__}."
                    )
                if detector is None:
                    raise ValueError(
                        "If using `quantity_mode = 'counts'` to simulate electron "
                        "counts, an `AbstractDetector` must be passed."
                    )
                image_model = ElectronCountsImageModel(
                    structure,
                    pose,
                    config,
                    scattering_theory,
                    detector,
                    applies_translation=applies_translation,
                    normalizes_signal=normalizes_signal,
                    signal_region=signal_region,
                )
            elif quantity_mode == "contrast":
                image_model = ContrastImageModel(
                    structure,
                    pose,
                    config,
                    scattering_theory,
                    applies_translation=applies_translation,
                    normalizes_signal=normalizes_signal,
                    signal_region=signal_region,
                )
            elif quantity_mode == "intensity":
                image_model = IntensityImageModel(
                    structure,
                    pose,
                    config,
                    scattering_theory,
                    applies_translation=applies_translation,
                    normalizes_signal=normalizes_signal,
                    signal_region=signal_region,
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
                structure,
                pose,
                config,
                integrator,
                transfer_theory,
                applies_translation=applies_translation,
                normalizes_signal=normalizes_signal,
                signal_region=signal_region,
            )

    return image_model


def _select_default_integrator(
    structure: AbstractStructureParameterisation, simulates_quantity: bool
) -> AbstractDirectIntegrator:
    if isinstance(structure, (FourierVoxelGridVolume, FourierVoxelSplineVolume)):
        integrator = FourierSliceExtraction(outputs_integral=simulates_quantity)
    elif isinstance(
        structure,
        (PengIndependentAtomVolume, GaussianMixtureVolume),
    ):
        integrator = GaussianMixtureProjection(use_error_functions=True)
    elif isinstance(structure, RealVoxelGridVolume):
        integrator = NufftProjection(outputs_integral=simulates_quantity)
    else:
        raise ValueError(
            "Could not select default integrator for structure of "
            f"type {type(structure).__name__}. If using a custom potential "
            "please directly pass an integrator."
        )
    return integrator
