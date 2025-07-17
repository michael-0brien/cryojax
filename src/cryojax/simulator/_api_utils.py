from typing import Callable, Literal, Optional

import equinox as eqx
from jaxtyping import Array, Bool

from ..ndimage.transforms import FilterLike, MaskLike
from ._config import AbstractConfig, DoseConfig
from ._detector import AbstractDetector
from ._image_model import (
    AbstractImageModel,
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
from ._structure import BasicStructure
from ._transfer_theory import ContrastTransferTheory


def make_image_model(
    potential: AbstractPotentialRepresentation,
    config: AbstractConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    integrator: Optional[AbstractPotentialIntegrator] = None,
    detector: Optional[AbstractDetector] = None,
    *,
    filter: Optional[FilterLike] = None,
    mask: Optional[MaskLike] = None,
    normalizes_signal: bool = False,
    signal_region: Optional[Bool[Array, "{config.y_dim} {config.x_dim}"]] = None,
    outputs_real_space: bool = True,
    physical_units: bool = True,
    mode: Literal["contrast", "intensity", "counts"] = "contrast",
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
    - `detector`:
        If `mode = 'counts'` is chosen, then an `AbstractDetector` class must be
        chosen to simulate electron counts.
    - `filter`:
        A filter to apply to the image.
    - `mask`:
        A mask to apply to the image.
    - `outputs_real_space`:
        Return the image in real or fourier space.
    - `normalizes_signal`:
        Whether or not to normalize the image.
    - `signal_region`:
        If `normalizes_signal = True`, this is a boolean array that is 1 where
        there is signal and 0 otherwise.
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

    A tuple of the `AbstractImageModel` and a function of the form

    ```python
    image_model, simulate_fn = make_image_model(...)
    image = simulate_fn(image_model)
    ```
    """
    # Build the image model
    integrator = _select_default_integrator(potential)
    structure = BasicStructure(potential, pose)
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
                structure,
                config,
                scattering_theory,
                detector,
                normalizes_signal=normalizes_signal,
                signal_region=signal_region,
            )
        elif mode == "contrast":
            image_model = ContrastImageModel(
                structure,
                config,
                scattering_theory,
                normalizes_signal=normalizes_signal,
                signal_region=signal_region,
            )
        elif mode == "intensity":
            image_model = IntensityImageModel(
                structure,
                config,
                scattering_theory,
                normalizes_signal=normalizes_signal,
                signal_region=signal_region,
            )
        else:
            raise ValueError(
                f"`mode = {mode}` not supported. Supported modes for simulating "
                "physical quantities are 'contrast', 'intensity', and 'counts'."
            )
    else:
        image_model = LinearImageModel(
            structure,
            config,
            integrator,
            transfer_theory,
            normalizes_signal=normalizes_signal,
            signal_region=signal_region,
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
