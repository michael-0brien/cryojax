from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Complex, Float

from .._config import AbstractConfig


StructureT = TypeVar("StructureT")


class AbstractMultisliceIntegrator(eqx.Module, Generic[StructureT], strict=True):
    """Base class for a multi-slice integration scheme."""

    @abstractmethod
    def integrate(
        self,
        potential: StructureT,
        config: AbstractConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        raise NotImplementedError
