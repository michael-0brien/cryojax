from abc import abstractmethod
from typing import Generic
from typing_extensions import override

from jaxtyping import Array, Complex, Float

from ..._image_config import AbstractImageConfig
from ..base_integrator import AbstractVolumeIntegrator, VolRepT


class AbstractMultisliceIntegrator(
    AbstractVolumeIntegrator[VolRepT], Generic[VolRepT], strict=True
):
    """Base class for a multislice integration scheme."""

    @abstractmethod
    @override
    def integrate(
        self,
        volume_representation: VolRepT,
        image_config: AbstractImageConfig,
        amplitude_contrast_ratio: Float[Array, ""] | float,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        raise NotImplementedError
