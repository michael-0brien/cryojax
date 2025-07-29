from typing import Any, Literal, Optional

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
from ._structure import (
    AbstractStructureMapping,
    FourierVoxelGridStructure,
    FourierVoxelSplineStructure,
    GaussianIndependentAtomPotential,
    GaussianMixtureStructure,
    PengIndependentAtomPotential,
    RealVoxelGridStructure,
)
from ._transfer_theory import ContrastTransferTheory


def make_image_model(
    structure: AbstractStructureMapping,
    config: AbstractConfig,
    pose: AbstractPose,
    transfer_theory: Optional[ContrastTransferTheory] = None,
    integrator: Optional[AbstractDirectIntegrator] = None,
    detector: Optional[AbstractDetector] = None,
    *,
    options: dict[str, Any] = {},
    physical_units: bool = False,
    mode: Literal["contrast", "intensity", "counts"] = "contrast",
) -> AbstractImageModel:
    """Construct an `AbstractImageModel` for most common use-cases.

    **Arguments:**

    - `structure`:
        The representation of the protein structure.
        Common choices are the `FourierVoxelGridStructure`
        for fourier-space voxel grids or the `PengIndependentAtomPotential`
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
        If `mode = 'counts'` is chosen, then an `AbstractDetector` class must be
        chosen to simulate electron counts.
    - `options`:
        A dictionary of options to be passed to the image model. This has keys
        - "applies_translation":
            If `True`, apply the in-plane translation in the `AbstractPose`
            via phase shifts in fourier space.
        - "normalizes_signal":
            If `True`, normalizes_signal the image before returning.
        - "signal_region":
            A boolean array that is 1 where there is signal,
            and 0 otherwise used to normalize the image.
            Must have shape equal to `AbstractConfig.shape`.
    - `physical_units`:
        If `True`, the image simulated is a physical quantity, which is
        chosen with the `mode` argument. Otherwise, simulate an image without
        scaling to absolute units.
    - `mode`:
        The physical observable to simulate. Not used if `physical_units = False`.
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
    integrator = _select_default_integrator(structure, physical_units)
    if transfer_theory is None:
        # Image model for projections
        image_model = ProjectionImageModel(structure, pose, config, integrator, **options)
    else:
        # Simulate physical observables
        if physical_units:
            scattering_theory = WeakPhaseScatteringTheory(integrator, transfer_theory)
            if mode == "counts":
                if not isinstance(config, DoseConfig):
                    raise ValueError(
                        "If using `mode = 'counts'` to simulate electron counts, "
                        "pass `config = DoseConfig(...)`. Got config "
                        f"{type(config).__name__}."
                    )
                if detector is None:
                    raise ValueError(
                        "If using `mode = 'counts'` to simulate electron counts, "
                        "an `AbstractDetector` must be passed."
                    )
                image_model = ElectronCountsImageModel(
                    structure, pose, config, scattering_theory, detector, **options
                )
            elif mode == "contrast":
                image_model = ContrastImageModel(
                    structure, pose, config, scattering_theory, **options
                )
            elif mode == "intensity":
                image_model = IntensityImageModel(
                    structure, pose, config, scattering_theory, **options
                )
            else:
                raise ValueError(
                    f"`mode = {mode}` not supported. Supported modes for simulating "
                    "physical quantities are 'contrast', 'intensity', and 'counts'."
                )
        else:
            # Linear image model
            image_model = LinearImageModel(
                structure, pose, config, integrator, transfer_theory, **options
            )

    return image_model


def _select_default_integrator(
    structure: AbstractStructureMapping, physical_units: bool
) -> AbstractDirectIntegrator:
    if isinstance(structure, (FourierVoxelGridStructure, FourierVoxelSplineStructure)):
        integrator = FourierSliceExtraction(outputs_integral=physical_units)
    elif isinstance(
        structure,
        (
            PengIndependentAtomPotential,
            GaussianIndependentAtomPotential,
            GaussianMixtureStructure,
        ),
    ):
        integrator = GaussianMixtureProjection(use_error_functions=True)
    elif isinstance(structure, RealVoxelGridStructure):
        integrator = NufftProjection(outputs_integral=physical_units)
    else:
        raise ValueError(
            "Could not select default integrator for structure of "
            f"type {type(structure).__name__}. If using a custom potential "
            "please directly pass an integrator."
        )
    return integrator
