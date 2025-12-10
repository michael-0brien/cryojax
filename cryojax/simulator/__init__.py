# Deprecation warnings
import warnings as _warnings
from typing import Any as _Any

from ._api_utils import (
    load_tabulated_volume as load_tabulated_volume,
    make_image_model as make_image_model,
    render_voxel_volume as render_voxel_volume,
)
from ._detector import (
    AbstractDetector as AbstractDetector,
    AbstractDQE as AbstractDQE,
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
    AbstractLikelihoodNoiseModel as AbstractLikelihoodNoiseModel,
    AbstractNoiseModel as AbstractNoiseModel,
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
    AbstractVolumeIntegrator as AbstractVolumeIntegrator,
    AbstractVolumeParametrization as AbstractVolumeParametrization,
    AbstractVolumeRenderFn as AbstractVolumeRenderFn,
    AbstractVolumeRepresentation as AbstractVolumeRepresentation,
    AbstractVoxelVolume as AbstractVoxelVolume,
    AutoVolumeProjection as AutoVolumeProjection,
    AutoVolumeRenderFn as AutoVolumeRenderFn,
    FFTAtomProjection as FFTAtomProjection,
    FFTAtomRenderFn as FFTAtomRenderFn,
    FourierSliceExtraction as FourierSliceExtraction,
    FourierVoxelGridVolume as FourierVoxelGridVolume,
    FourierVoxelSplineVolume as FourierVoxelSplineVolume,
    GaussianMixtureProjection as GaussianMixtureProjection,
    GaussianMixtureRenderFn as GaussianMixtureRenderFn,
    GaussianMixtureVolume as GaussianMixtureVolume,
    IndependentAtomVolume as IndependentAtomVolume,
    LobatoScatteringFactor as LobatoScatteringFactor,
    PengScatteringFactor as PengScatteringFactor,
    RealVoxelGridVolume as RealVoxelGridVolume,
    RealVoxelProjection as RealVoxelProjection,
)


def __getattr__(name: str) -> _Any:
    # Future deprecations
    if name == "AberratedAstigmaticCTF":
        _warnings.warn(
            "'AberratedAstigmaticCTF' is deprecated and will be removed in "
            "cryoJAX 0.6.0. Use 'AstigmaticCTF' instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return AstigmaticCTF
    if name == "CTF":
        _warnings.warn(
            "Alias 'CTF' is deprecated and will be removed in "
            "cryoJAX 0.6.0. Use 'AstigmaticCTF' instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return AstigmaticCTF
    if name == "NufftProjection":
        _warnings.warn(
            "'NufftProjection' is deprecated and will be removed in "
            "cryoJAX 0.6.0. Use 'RealVoxelProjection' instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return RealVoxelProjection
    if name == "PengScatteringFactorParameters":
        _warnings.warn(
            "'PengScatteringFactorParameters' has been moved to `cryojax.constants` "
            "will be removed from `cryojax.simulator` in "
            "cryoJAX 0.6.0.",
            category=FutureWarning,
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
            category=FutureWarning,
            stacklevel=2,
        )
        return GaussianMixtureVolume
    if name == "UncorrelatedGaussianNoiseModel":
        _warnings.warn(
            "'UncorrelatedGaussianNoiseModel' is deprecated and "
            "will be removed in cryoJAX 0.6.0. Instead, use "
            "'GaussianWhiteNoiseModel'.",
            category=FutureWarning,
            stacklevel=2,
        )
        return GaussianWhiteNoiseModel
    if name == "CorrelatedGaussianNoiseModel":
        _warnings.warn(
            "'CorrelatedGaussianNoiseModel' is deprecated and "
            "will be removed in cryoJAX 0.6.0. Instead, use "
            "'GaussianColoredNoiseModel'.",
            category=FutureWarning,
            stacklevel=2,
        )
        return GaussianColoredNoiseModel
    # Deprecated in previous versions
    if name == "DiscreteStructuralEnsemble":
        raise ImportError(
            "'DiscreteStructuralEnsemble' was deprecated in cryoJAX 0.5.0. "
            "To achieve similar functionality, see the examples section "
            "of the documentation: "
            "https://michael-0brien.github.io/cryojax/examples/simulate-relion-dataset/.",
        )

    raise AttributeError(f"cannot import name '{name}' from 'cryojax.simulator'")
