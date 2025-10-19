# Deprecation warnings
import warnings as _warnings
from typing import Any as _Any

from ._api_utils import make_image_model as make_image_model
from ._detector import (
    AbstractDetector as AbstractDetector,
    AbstractDQE as AbstractDQE,
    CountingDQE as CountingDQE,
    GaussianDetector as GaussianDetector,
    NullDQE as NullDQE,
    PoissonDetector as PoissonDetector,
)
from ._image_config import (
    AbstractImageConfig as AbstractImageConfig,
    BasicImageConfig as BasicImageConfig,
    DoseImageConfig as DoseImageConfig,
    GridHelper as GridHelper,
)
from ._image_model import (
    AbstractImageModel as AbstractImageModel,
    AbstractPhysicalImageModel as AbstractPhysicalImageModel,
    ContrastImageModel as ContrastImageModel,
    ElectronCountsImageModel as ElectronCountsImageModel,
    IntensityImageModel as IntensityImageModel,
    LinearImageModel as LinearImageModel,
    ProjectionImageModel as ProjectionImageModel,
)
from ._noise_model import (
    AbstractEmpiricalNoiseModel as AbstractEmpiricalNoiseModel,
    AbstractGaussianNoiseModel as AbstractGaussianNoiseModel,
    AbstractNoiseModel as AbstractNoiseModel,
    AbstractProbabilisticNoiseModel as AbstractProbabilisticNoiseModel,
    GaussianColoredNoiseModel as GaussianColoredNoiseModel,
    GaussianWhiteNoiseModel as GaussianWhiteNoiseModel,
)
from ._pose import (
    AbstractPose as AbstractPose,
    AxisAnglePose as AxisAnglePose,
    EulerAnglePose as EulerAnglePose,
    QuaternionPose as QuaternionPose,
)
from ._scattering_theory import (
    AbstractScatteringTheory as AbstractScatteringTheory,
    AbstractWaveScatteringTheory as AbstractWaveScatteringTheory,
    AbstractWeakPhaseScatteringTheory as AbstractWeakPhaseScatteringTheory,
    StrongPhaseScatteringTheory as StrongPhaseScatteringTheory,
    WeakPhaseScatteringTheory as WeakPhaseScatteringTheory,
)
from ._solvent_2d import AbstractRandomSolvent2D as AbstractRandomSolvent2D
from ._transfer_theory import (
    AbstractCTF as AbstractCTF,
    AbstractTransferTheory as AbstractTransferTheory,
    AstigmaticCTF as AstigmaticCTF,
    ContrastTransferTheory as ContrastTransferTheory,
    WaveTransferTheory as WaveTransferTheory,
)
from ._volume import (
    AbstractAtomVolume as AbstractAtomVolume,
    AbstractVolumeParametrization as AbstractVolumeParametrization,
    AbstractVolumeRepresentation as AbstractVolumeRepresentation,
    AbstractVoxelVolume as AbstractVoxelVolume,
    FourierVoxelGridVolume as FourierVoxelGridVolume,
    FourierVoxelSplineVolume as FourierVoxelSplineVolume,
    GaussianMixtureVolume as GaussianMixtureVolume,
    RealVoxelGridVolume as RealVoxelGridVolume,
)
from ._volume_integrator import (
    AbstractVolumeIntegrator as AbstractVolumeIntegrator,
    FourierSliceExtraction as FourierSliceExtraction,
    GaussianMixtureProjection as GaussianMixtureProjection,
    NufftProjection as NufftProjection,
)
from ._volume_rendering import (
    AbstractVolumeRenderFn as AbstractVolumeRenderFn,
    GaussianMixtureRenderFn as GaussianMixtureRenderFn,
)


def __getattr__(name: str) -> _Any:
    # Future deprecations
    if name == "AberratedAstigmaticCTF":
        _warnings.warn(
            "'AberratedAstigmaticCTF' is deprecated and will be removed in "
            "cryoJAX 0.6.0. Use 'AstigmaticCTF' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return AstigmaticCTF
    if name == "CTF":
        _warnings.warn(
            "Alias 'CTF' is deprecated and will be removed in "
            "cryoJAX 0.6.0. Use 'AstigmaticCTF' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return AstigmaticCTF
    if name == "PengScatteringFactorParameters":
        _warnings.warn(
            "'PengScatteringFactorParameters' has been moved to `cryojax.constants` "
            "will be removed from `cryojax.simulator` in "
            "cryoJAX 0.6.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        from ..constants import PengScatteringFactorParameters

        return PengScatteringFactorParameters
    if name == "PengAtomicVolume":
        _warnings.warn(
            "'PengAtomicVolume' is deprecated and will be removed in "
            "cryoJAX 0.6.0. To achieve identical functionality, use "
            "`GaussianMixtureVolume.from_tabulated_parameters`. "
            "This is a breaking change if you are "
            "directly using `PengAtomicVolume.__init__`.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return GaussianMixtureVolume
    # Deprecated in previous versions
    if name == "DiscreteStructuralEnsemble":
        raise ValueError(
            "'DiscreteStructuralEnsemble' was deprecated in cryoJAX 0.5.0. "
            "To achieve similar functionality, see the examples section "
            "of the documentation: "
            "https://michael-0brien.github.io/cryojax/examples/simulate-relion-dataset/.",
        )
    if name == "UncorrelatedGaussianNoiseModel":
        raise ValueError(
            "'UncorrelatedGaussianNoiseModel' was deprecated in cryoJAX 0.5.1 and "
            "renamed to `GaussianWhiteNoiseModel` instead.",
        )
    if name == "CorrelatedGaussianNoiseModel":
        raise ValueError(
            "'CorrelatedGaussianNoiseModel' was deprecated in cryoJAX 0.5.1 and "
            "renamed to `GaussianColoredNoiseModel` instead.",
        )

    raise AttributeError(f"cannot import name '{name}' from 'cryojax.simulator'")
