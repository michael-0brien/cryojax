"""Shared helpers for volume rendering backends."""

import math

import jax.numpy as jnp
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


def center_shift(
    shape_u: tuple[int, ...], shape_out: tuple[int, ...], pixel_size_u: Float[Array, ""]
) -> Float[Array, " d"]:
    """The shift, in `(x, y[, z])` order, that places the physical origin at the output
    grid's center `N//2` on the upsampled grid, whose own center is `N_u//2`."""
    return jnp.asarray(
        [
            pixel_size_u * (n_u * (n // 2) / n - n_u // 2)
            for n_u, n in zip(shape_u[::-1], shape_out[::-1])
        ]
    )


def build_extraction_mesh(
    shape: tuple[int, ...],
    shape_ds: tuple[int, ...],
    *,
    modeord: int,
    outputs_rfft: bool = False,
):
    """Build indices to extract modes from FFT output. If `outputs_rfft`, the input is
    an `rfftn` and the last axis keeps `0 ... shape_ds[-1] // 2` (needs `modeord=1`)."""
    if outputs_rfft and modeord != 1:
        raise ValueError(
            "`build_extraction_mesh` with `outputs_rfft` requires `modeord=1`. "
            f"Got `modeord = {modeord}`."
        )
    assert len(shape) == len(shape_ds)
    assert len(shape) in [2, 3]
    mean_factor = math.prod(shape_ds) / math.prod(shape)
    indices_1d = tuple(
        build_extraction_indices_1d(dim, dim_ds, modeord=modeord)
        for dim, dim_ds in zip(shape, shape_ds)
    )
    if outputs_rfft:
        indices_1d = indices_1d[:-1] + (jnp.arange(shape_ds[-1] // 2 + 1),)
    return jnp.meshgrid(*indices_1d, indexing="ij"), mean_factor


def build_extraction_indices_1d(dim: int, dim_ds: int, *, modeord: int):
    """Build indices to extract modes from FFT output. Adapted from `nufftax`"""
    q_min = -(dim_ds // 2)
    q_max = (dim_ds - 1) // 2

    idx_pos = jnp.arange(q_max + 1)
    idx_neg = jnp.arange(dim + q_min, dim)

    if modeord == 0:
        indices = jnp.concatenate([idx_neg, idx_pos])
    else:
        indices = jnp.concatenate([idx_pos, idx_neg])

    return indices


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
