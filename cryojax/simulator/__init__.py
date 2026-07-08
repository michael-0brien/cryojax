from typing import Any as _Any

from ._api_utils import (
    load_tabulated_volume as load_tabulated_volume,
    make_image_model as make_image_model,
    render_voxel_volume as render_voxel_volume,
    suggest_n_spread as suggest_n_spread,
)
from ._detector import (
    AbstractDetector as AbstractDetector,
    GaussianDetector as GaussianDetector,
    PoissonDetector as PoissonDetector,
)
from ._image_config import (
    AbstractImageConfig as AbstractImageConfig,
    BasicImageConfig as BasicImageConfig,
    DoseImageConfig as DoseImageConfig,
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
    RytovScatteringTheory as RytovScatteringTheory,
    WeakPhaseScatteringTheory as WeakPhaseScatteringTheory,
)
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
    EwaldSphereExtraction as EwaldSphereExtraction,
    FourierSliceExtraction as FourierSliceExtraction,
    FourierVoxelGridVolume as FourierVoxelGridVolume,
    FourierVoxelSplineVolume as FourierVoxelSplineVolume,
    GaussianFourierProjection as GaussianFourierProjection,
    GaussianFourierRenderFn as GaussianFourierRenderFn,
    GaussianFourierVolume as GaussianFourierVolume,
    GaussianMixtureProjection as GaussianMixtureProjection,
    GaussianMixtureRenderFn as GaussianMixtureRenderFn,
    GaussianMixtureVolume as GaussianMixtureVolume,
    RealVoxelGridVolume as RealVoxelGridVolume,
)


_REMOVED = {
    "RealVoxelCloudVolume": None,
    "RealVoxelProjection": None,
    "IndependentAtomVolume": "GaussianFourierVolume",
    "IndependentAtomProjection": "GaussianFourierProjection",
    "IndependentAtomRenderFn": "GaussianFourierRenderFn",
    "AberratedAstigmaticCTF": "AstigmaticCTF",
    "CTF": "AstigmaticCTF",
    "NufftProjection": None,
    "PengAtomicVolume": "GaussianMixtureVolume",
    "UncorrelatedGaussianNoiseModel": "GaussianWhiteNoiseModel",
    "CorrelatedGaussianNoiseModel": "GaussianColoredNoiseModel",
    "DiscreteStructuralEnsemble": None,
}
_MOVED = {
    "PengScatteringFactorParameters": "cryojax.constants",
}


def __getattr__(name: str) -> _Any:
    if name in _REMOVED:
        replacement = _REMOVED[name]
        if replacement is not None:
            raise AttributeError(
                f"'{name}' was removed in cryoJAX 0.6.0. Use '{replacement}' instead."
            )
        raise AttributeError(f"'{name}' was removed in cryoJAX 0.6.0.")
    if name in _MOVED:
        raise AttributeError(
            f"'{name}' was removed from `cryojax.simulator` in cryoJAX 0.6.0. "
            f"Use `{_MOVED[name]}.{name}` instead."
        )
    raise AttributeError(f"cannot import name '{name}' from 'cryojax.simulator'")
