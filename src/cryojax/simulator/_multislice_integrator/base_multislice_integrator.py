from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Complex, Float

from .._image_config import AbstractImageConfig


VolumeT = TypeVar("VolumeT")


class AbstractMultisliceIntegrator(eqx.Module, Generic[VolumeT], strict=True):
    """Base class for a multi-slice integration scheme."""

    @abstractmethod
    def integrate(
        self,
        volume: VolumeT,
        config: AbstractImageConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        raise NotImplementedError
