from typing import Any, ClassVar
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import FloatLike
from .._image_config import AbstractImageConfig
from .base_volume import (
    AbstractVolumeIntegrator,
    AbstractVolumeRenderFn,
    AbstractVolumeRepresentation,
    ProjectionArray,
    VoxelArray,
)
from .fourier_voxels import (
    FourierSliceExtraction,
    FourierVoxelGridVolume,
    FourierVoxelSplineVolume,
)
from .gaussian_volume import (
    GaussianMixtureProjection,
    GaussianMixtureRenderFn,
    GaussianMixtureVolume,
)
from .independent_atom_volume import (
    FFTAtomProjection,
    FFTAtomRenderFn,
    IndependentAtomVolume,
)
from .real_voxels import RealVoxelGridVolume, RealVoxelProjection


class AutoVolumeProjection(
    AbstractVolumeIntegrator[AbstractVolumeRepresentation], strict=True
):
    """Volume projection auto selection from cryoJAX
    `AbstractVolumeIntegrator` implementations.

    !!! info
        Based on the [`cryojax.simulator.AbstractVolumeRepresentation`][] passed
        at runtime, this class chooses a default projection method.
        In particular,

        | Volume representation | Projection method | Atom or voxel? |
        | :-------------------- | :------------------ | :------------------ |
        | [`cryojax.simulator.GaussianMixtureVolume`][] | [`cryojax.simulator.GaussianMixtureProjection`][] | atom |
        | [`cryojax.simulator.IndependentAtomVolume`][] | [`cryojax.simulator.FFTAtomProjection`][] | atom |
        | [`cryojax.simulator.FourierVoxelGridVolume`][] or [`cryojax.simulator.FourierVoxelSplineVolume`][] | [`cryojax.simulator.FourierSliceExtraction`][] | voxel |
        | [`cryojax.simulator.RealVoxelGridVolume`][] | [`cryojax.simulator.RealVoxelProjection`][] | voxel |

        To use advanced options for a given projection method,
        see each respective class.

    !!! warning
        If using [`cryojax.simulator.FFTAtomRenderFn`][] or [`cryojax.simulator.RealVoxelProjection`][], [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
        must be installed. See the cryoJAX [installation instructions](https://github.com/michael-0brien/cryojax?tab=readme-ov-file#installation)
        for installing `jax-finufft`.
    """  # noqa: E501

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
        volume_representation: AbstractVolumeRepresentation,
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

- `options`:
    Keyword arguments passed to the resolved projection method`,
    e.g. `GaussianMixtureProjection(**options)`.
"""


class AutoVolumeRenderFn(
    AbstractVolumeRenderFn[AbstractVolumeRepresentation], strict=True
):
    """Volume rendering auto selection from cryoJAX
    `AbstractVolumeRenderFn` implementations.

    !!! info
        Based on the [`cryojax.simulator.AbstractVolumeRepresentation`][] passed
        at runtime, this class chooses a default rendering function.
        In particular,

        | Volume representation | Rendering function  |
        | :-------------------- | :-----------------  |
        | [`cryojax.simulator.GaussianMixtureVolume`][] | [`cryojax.simulator.GaussianMixtureRenderFn`][] |
        | [`cryojax.simulator.IndependentAtomVolume`][] | [`cryojax.simulator.FFTAtomRenderFn`][] |

        To use advanced options for a given rendering function,
        see each respective class.

    !!! warning
        If using [`cryojax.simulator.FFTAtomRenderFn`][], [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
        must be installed. See the cryoJAX [installation instructions](https://github.com/michael-0brien/cryojax?tab=readme-ov-file#installation)
        for installing `jax-finufft`.
    """  # noqa: E501

    shape: tuple[int, int, int]
    voxel_size: Float[Array, ""]

    options: dict[str, Any]

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            The shape of the voxel grid for rendering.
        - `voxel_size`:
            The voxel size for rendering.
        - `options`:
            Keyword arguments passed to the resolved rendering function,
            e.g. `GaussianMixtureRenderFn(shape, voxel_size, **options)`.
        """
        self.shape = shape
        self.voxel_size = jnp.asarray(voxel_size, dtype=float)
        self.options = options

    def _select_render_method(
        self, volume: AbstractVolumeRepresentation
    ) -> AbstractVolumeRenderFn:
        if isinstance(volume, IndependentAtomVolume):
            return FFTAtomRenderFn(self.shape, self.voxel_size, **self.options)
        elif isinstance(volume, GaussianMixtureVolume):
            return GaussianMixtureRenderFn(self.shape, self.voxel_size, **self.options)
        else:
            raise ValueError(
                "Could not use `AutoVolumeRenderFn` for volume of "
                f"type {type(volume).__name__}. If using a custom volume, "
                "please directly pass its rendering function."
            )

    @override
    def __call__(
        self,
        volume_representation: AbstractVolumeRepresentation,
        *,
        outputs_real_space: bool = True,
        outputs_rfft: bool = False,
        fftshifted: bool = False,
    ) -> VoxelArray:
        render_fn = self._select_render_method(volume_representation)
        return render_fn(
            volume_representation,
            outputs_real_space=outputs_real_space,
            outputs_rfft=outputs_rfft,
            fftshifted=fftshifted,
        )
