import abc
from typing import Any, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array

from .._image_config import AbstractImageConfig
from .._volume_parametrisation import AbstractVolumeRepresentation


VolRepT = TypeVar("VolRepT", bound="AbstractVolumeRepresentation")


class AbstractVolumeIntegrator(eqx.Module, Generic[VolRepT], strict=True):
    """Base class for a method of integrating a volume into an image."""

    @abc.abstractmethod
    def integrate(
        self,
        volume_representation: VolRepT,
        image_config: AbstractImageConfig,
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError
