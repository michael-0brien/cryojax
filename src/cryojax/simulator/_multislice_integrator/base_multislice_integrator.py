from abc import abstractmethod
from typing import Generic, TypeVar

from equinox import Module
from jaxtyping import Array, Complex, Float

from .._instrument_config import InstrumentConfig


PotentialT = TypeVar("PotentialT")


class AbstractMultisliceIntegrator(Module, Generic[PotentialT], strict=True):
    """Base class for a multi-slice integration scheme."""

    @abstractmethod
    def integrate(
        self,
        potential: PotentialT,
        instrument_config: InstrumentConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[
        Array, "{instrument_config.padded_y_dim} {instrument_config.padded_x_dim}"
    ]:
        raise NotImplementedError
