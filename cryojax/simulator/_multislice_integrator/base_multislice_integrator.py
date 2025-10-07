from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Complex, Float

from .._image_config import AbstractImageConfig
from .._volume import AbstractVolumeRepresentation


VolRepT = TypeVar("VolRepT", bound="AbstractVolumeRepresentation")


class AbstractMultisliceIntegrator(eqx.Module, Generic[VolRepT], strict=True):
    """Base class for a multislice integration scheme."""

    @abstractmethod
    def integrate(
        self,
        volume_representation: VolRepT,
        image_config: AbstractImageConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        raise NotImplementedError
