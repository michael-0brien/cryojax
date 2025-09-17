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
    AbstractWeakPhaseScatteringTheory as AbstractWeakPhaseScatteringTheory,
    WeakPhaseScatteringTheory as WeakPhaseScatteringTheory,
)
from ._solvent_2d import AbstractRandomSolvent2D as AbstractRandomSolvent2D
from ._transfer_theory import (
    AbstractCTF as AbstractCTF,
    AbstractTransferTheory as AbstractTransferTheory,
    AstigmaticCTF as AstigmaticCTF,
    ContrastTransferTheory as ContrastTransferTheory,
)
from ._volume_integrator import (
    AbstractDirectIntegrator as AbstractDirectIntegrator,
    AbstractDirectVoxelIntegrator as AbstractDirectVoxelIntegrator,
    FourierSliceExtraction as FourierSliceExtraction,
    GaussianMixtureProjection as GaussianMixtureProjection,
    NufftProjection as NufftProjection,
)
from ._volume_parametrisation import (
    AbstractEnsembleParametrisation as AbstractEnsembleParametrisation,
    AbstractPengPotential as AbstractPengPotential,
    AbstractPointCloudVolume as AbstractPointCloudVolume,
    AbstractPotentialParametrisation as AbstractPotentialParametrisation,
    AbstractTabulatedPotential as AbstractTabulatedPotential,
    AbstractVolumeParametrisation as AbstractVolumeParametrisation,
    AbstractVolumeRepresentation as AbstractVolumeRepresentation,
    DiscreteStructuralEnsemble as DiscreteStructuralEnsemble,
    FourierVoxelGridVolume as FourierVoxelGridVolume,
    FourierVoxelSplineVolume as FourierVoxelSplineVolume,
    GaussianMixtureVolume as GaussianMixtureVolume,
    PengAtomPotential as PengAtomPotential,
    PengScatteringFactorParameters as PengScatteringFactorParameters,
    RealVoxelGridVolume as RealVoxelGridVolume,
)


def __getattr__(name: str):
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

    raise ImportError(f"cannot import name '{name}' from 'cryojax.simulator'")
