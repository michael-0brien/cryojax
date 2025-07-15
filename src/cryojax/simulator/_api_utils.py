from typing import Callable, Optional, Type

import equinox as eqx
from jaxtyping import Array

from ..ndimage.transforms import FilterLike, MaskLike
from ._config import AbstractConfig, DoseConfig
from ._detector import AbstractDetector
from ._image_model import (
    AbstractImageModel,
    AbstractPhysicalImageModel,
    ContrastImageModel,
    ElectronCountsImageModel,
    IntensityImageModel,
    LinearImageModel,
)
from ._pose import AbstractPose
from ._potential_integrator import (
    AbstractPotentialIntegrator,
    FourierSliceExtraction,
    GaussianMixtureProjection,
    NufftProjection,
)
from ._potential_representation import (
    AbstractPotentialRepresentation,
    FourierVoxelGridPotential,
    FourierVoxelSplinePotential,
    GaussianMixtureAtomicPotential,
    PengAtomicPotential,
    RealVoxelCloudPotential,
    RealVoxelGridPotential,
)
from ._scattering_theory import WeakPhaseScatteringTheory
from ._structure import Structure
from ._transfer_theory import ContrastTransferTheory


def make_image_model(
    potential: AbstractPotentialRepresentation,
    config: AbstractConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    integrator: Optional[AbstractPotentialIntegrator] = None,
    filter: Optional[FilterLike] = None,
    mask: Optional[MaskLike] = None,
    outputs_real_space: bool = True,
    *,
    model_class: Type[AbstractImageModel] = ContrastImageModel,
    detector: Optional[AbstractDetector] = None,
) -> tuple[AbstractImageModel, Callable[[AbstractImageModel], Array]]:
    """Construct an `AbstractImageModel` for most common use-cases.

    **Arguments:**

    - `potential`:
        The representation of the protein electrostatic potential.
        Common choices are the `FourierVoxelGridPotential`
        for fourier-space voxel grids or the `PengAtomicPotential`
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
    - `filter`:
        A filter to apply to the image.
    - `mask`:
        A mask to apply to the image.
    - `outputs_real_space`:
        Return the image in real or fourier space.

    **Advanced arguments:**

    - `model_class`:
        The kind of the imaging model chosen. Options are the `ContrastImageModel`
        (default), `IntensityImageModel`, or `ElectronCountsImageModel` for
        simulating image contrast, intensity, and electron counts. Addtionally,
        choose the `LinearImageModel` if a more streamlined model is desired that
        does not try to simulate in physical units.
    - `detector`:
        If a `model_class = ElectronCountsImageModel` is chosen, then pass an
        `AbstractDetector` class.

    **Returns:**

    A tuple of the `AbstractImageModel` and a function of the form

    ```python
    image_model, simulate_fn = make_image_model(...)
    image = simulate_fn(image_model)
    ```
    """
    # Build the image model
    integrator = _select_default_integrator(potential)
    structure = Structure(potential, pose)
    if issubclass(model_class, AbstractPhysicalImageModel):
        scattering_theory = WeakPhaseScatteringTheory(integrator, transfer_theory)
        if model_class is ElectronCountsImageModel:
            if not isinstance(config, DoseConfig):
                raise ValueError(
                    f"If using image model {model_class.__name__}, "
                    "pass `config = DoseConfig(...)`. Got config "
                    f"{type(config).__name__}."
                )
            if detector is None:
                raise ValueError(
                    f"If using image model {model_class.__name__}, "
                    "an `AbstractDetector` must be passed."
                )
            image_model = model_class(structure, config, scattering_theory, detector)
        elif model_class in [ContrastImageModel, IntensityImageModel]:
            image_model = model_class(structure, config, scattering_theory)
        else:
            raise ValueError(
                f"Image model of class {model_class.__name__} not supported. "
                "If creating a custom image model, please "
                "use its constructor directly."
            )
    elif model_class is LinearImageModel:
        image_model = LinearImageModel(structure, integrator, transfer_theory, config)
    else:
        raise ValueError(
            f"Unsupported `model_class` {model_class.__name__}. "
            "If creating a custom image model, please "
            "use its constructor directly."
        )

    # Grab the simulation function
    @eqx.filter_jit
    def simulate_fn(model: AbstractImageModel) -> Array:
        return model.render(
            outputs_real_space=outputs_real_space, filter=filter, mask=mask
        )

    return image_model, simulate_fn


def _select_default_integrator(
    potential: AbstractPotentialRepresentation,
) -> AbstractPotentialIntegrator:
    if isinstance(potential, (FourierVoxelGridPotential, FourierVoxelSplinePotential)):
        integrator = FourierSliceExtraction()
    elif isinstance(potential, (PengAtomicPotential, GaussianMixtureAtomicPotential)):
        integrator = GaussianMixtureProjection(use_error_functions=True)
    elif isinstance(potential, (RealVoxelCloudPotential, RealVoxelGridPotential)):
        integrator = NufftProjection()
    else:
        raise ValueError(
            "Could not select default integrator for potential of "
            f"type {type(potential).__name__}. If using a custom potential "
            "please directly pass an integrator."
        )
    return integrator
