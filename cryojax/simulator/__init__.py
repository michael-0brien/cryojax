# Deprecation warnings
import warnings as _warnings

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
    AbstractGaussianNoiseModel as AbstractGaussianNoiseModel,
    AbstractNoiseModel as AbstractNoiseModel,
    CorrelatedGaussianNoiseModel as CorrelatedGaussianNoiseModel,
    UncorrelatedGaussianNoiseModel as UncorrelatedGaussianNoiseModel,
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
    AbstractAtomicVolume as AbstractAtomicVolume,
    AbstractPointCloudVolume as AbstractPointCloudVolume,
    AbstractTabulatedAtomicVolume as AbstractTabulatedAtomicVolume,
    AbstractVolumeParametrization as AbstractVolumeParametrization,
    AbstractVolumeRepresentation as AbstractVolumeRepresentation,
    FourierVoxelGridVolume as FourierVoxelGridVolume,
    FourierVoxelSplineVolume as FourierVoxelSplineVolume,
    GaussianMixtureVolume as GaussianMixtureVolume,
    PengAtomicVolume as PengAtomicVolume,
    RealVoxelGridVolume as RealVoxelGridVolume,
)
from ._volume_integrator import (
    AbstractVolumeIntegrator as AbstractVolumeIntegrator,
    FourierSliceExtraction as FourierSliceExtraction,
    GaussianMixtureProjection as GaussianMixtureProjection,
    NufftProjection as NufftProjection,
)


def __getattr__(name: str):
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
    # Deprecated in previous versions
    if name == "DiscreteStructuralEnsemble":
        raise ValueError(
            "'DiscreteStructuralEnsemble' was deprecated in cryoJAX 0.5.0. "
            "To achieve similar functionality, see the examples section "
            "of the documentation: "
            "https://michael-0brien.github.io/cryojax/examples/simulate-relion-dataset/.",
        )

    raise AttributeError(f"cannot import name '{name}' from 'cryojax.simulator'")
