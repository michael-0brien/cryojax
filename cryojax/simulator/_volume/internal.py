"""Shared helpers for volume rendering backends."""

from jaxtyping import Array, Float

from ...ndimage import FourierGaussian, FourierSinc, make_1d_frequency_grid
from .._image_config import AbstractImageConfig
from .base_volume import AbstractVolumeIntegrator


def make_frequencies_1d(
    shape_u: tuple[int, ...],
    pixel_size_u: Float[Array, ""],
    modeord: int = 0,
    outputs_rfft: bool = False,
):
    """One-dimensional frequencies in `(x, y[, z])` order. If `outputs_rfft`, the x
    entry holds only the non-negative frequencies (requires `modeord=1`)."""
    if outputs_rfft and modeord != 1:
        raise ValueError(
            "`make_frequencies_1d` with `outputs_rfft` requires `modeord=1`. "
            f"Got `modeord = {modeord}`."
        )
    return tuple(
        make_1d_frequency_grid(
            s,
            pixel_size_u,
            outputs_rfftfreqs=outputs_rfft and i == 0,
            fftshifted=(True if modeord == 0 else False),
        )
        for i, s in enumerate(shape_u[::-1])
    )


def eval_separable_impl(
    kernel_fn: FourierGaussian | FourierSinc, frequencies_1d: tuple[Array, ...]
):
    ndim = len(frequencies_1d)
    assert 1 in kernel_fn.spatial_dims and ndim in [2, 3]
    if len(frequencies_1d) == 2:
        q_x, q_y = frequencies_1d
        return kernel_fn(q_x)[None, :] * kernel_fn(q_y)[:, None]
    else:
        q_x, q_y, q_z = frequencies_1d
        return (
            kernel_fn(q_x)[None, None, :]
            * kernel_fn(q_y)[None, :, None]
            * kernel_fn(q_z)[:, None, None]
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
