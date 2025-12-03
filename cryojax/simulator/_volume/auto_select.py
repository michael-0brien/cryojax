from typing import Any, ClassVar
from typing_extensions import override

import equinox as eqx

from .._image_config import AbstractImageConfig
from .base_volume import (
    AbstractVolumeIntegrator,
    AbstractVolumeRepresentation,
    ProjectionArray,
    VolRep,
)
from .fourier_voxels import (
    FourierSliceExtraction,
    FourierVoxelGridVolume,
    FourierVoxelSplineVolume,
)
from .gaussian_volume import GaussianMixtureProjection, GaussianMixtureVolume
from .independent_atom_volume import FFTAtomProjection, IndependentAtomVolume
from .real_voxels import RealVoxelGridVolume, RealVoxelProjection


class AutoVolumeProjection(AbstractVolumeIntegrator[VolRep]):
    options: dict[str, Any] = eqx.field(default_factory=dict)

    outputs_ewald_sphere: ClassVar[bool] = False

    def _select_projection_method(
        self, volume: AbstractVolumeRepresentation
    ) -> AbstractVolumeIntegrator:
        if isinstance(volume, (FourierVoxelGridVolume, FourierVoxelSplineVolume)):
            integrator = FourierSliceExtraction(**self.options)
        elif isinstance(volume, GaussianMixtureVolume):
            integrator = GaussianMixtureProjection(**self.options)
        elif isinstance(volume, RealVoxelGridVolume):
            integrator = RealVoxelProjection(**self.options)
        elif isinstance(volume, IndependentAtomVolume):
            integrator = FFTAtomProjection(**self.options)
        else:
            raise ValueError(
                "Could not use `AutoVolumeProjection` for volume of "
                f"type {type(volume).__name__}. If using a custom volume, "
                "please directly pass an integrator."
            )
        return integrator

    @override
    def integrate(
        self,
        volume_representation: VolRep,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionArray:
        """Automatically select volume projection method given a
        volume representation.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The image configuration.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in Fourier.

        **Returns:**

        The volume projection in real or Fourier space at the
        `AbstractImageConfig.padded_shape` and the `image_config.pixel_size`.
        """
        volume_integrator = self._select_projection_method(volume_representation)
        return volume_integrator.integrate(
            volume_representation, image_config, outputs_real_space=outputs_real_space
        )


AutoVolumeProjection.__init__.__doc__ = """**Arguments:**

- `options`: Keyword arguments passed to `AbstractVolumeIntegrator.__init__`.
"""
