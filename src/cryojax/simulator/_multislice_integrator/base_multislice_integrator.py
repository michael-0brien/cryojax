from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import Module
from jaxtyping import Array, Complex, Float

from .._config import AbstractConfig


PotentialT = TypeVar("PotentialT")


class AbstractMultisliceIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a multi-slice integration scheme."""

    @abstractmethod
    def integrate(
        self,
        potential: PotentialT,
        config: AbstractConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[Array, "{config.padded_y_dim} {config.padded_x_dim}"]:
        raise NotImplementedError
