"""Shared helpers for volume rendering backends."""

from jaxtyping import Array, Float

from ...ndimage import make_1d_frequency_grid
from .._image_config import AbstractImageConfig
from .base_volume import AbstractVolumeIntegrator


def make_frequencies_1d(
    shape_u: tuple[int, ...],
    pixel_size_u: Float[Array, ""],
    modeord: int = 0,
):
    return tuple(
        make_1d_frequency_grid(
            s,
            pixel_size_u,
            outputs_rfftfreqs=False,
            fftshifted=(True if modeord == 0 else False),
        )
        for s in shape_u[::-1]
    )


def check_anisotropy_supported(
    volume_integrator: AbstractVolumeIntegrator, image_config: AbstractImageConfig
) -> None:
    if image_config.is_anisotropic and not volume_integrator.supports_anisotropy:
        raise NotImplementedError(
            f"`{type(volume_integrator).__name__}` does not support "
            "`AnisotropicImageConfig`. Use a volume integrator that supports it, "
            "e.g. `GaussianMixtureProjection` or `FourierSliceExtraction`, or a "
            "`BasicImageConfig`."
        )
