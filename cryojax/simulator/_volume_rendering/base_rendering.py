import abc
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array

from .._volume import AbstractVolumeRepresentation


VolRepT = TypeVar("VolRepT", bound="AbstractVolumeRepresentation")


class AbstractVolumeRenderFn(eqx.Module, Generic[VolRepT], strict=True):
    """Base class for rendering a volume onto voxels."""

    @abc.abstractmethod
    def __call__(
        self,
        volume_representation: VolRepT,
        *,
        outputs_real_space: bool = True,
        outputs_rfft: bool = False,
        fftshifted: bool = False,
    ) -> Array:
        raise NotImplementedError
