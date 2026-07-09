"""Shared helpers for volume rendering backends."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...ndimage import make_1d_frequency_grid


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


def spread_and_sum_gaussian_components(
    spread_one: Callable[[Array, Array, int], Array],
    amplitudes: Float[Array, "... n_gaussians"],
    variances: Float[Array, "... n_gaussians"],
    n_spread: int | tuple[int, ...],
) -> Array:
    """Spread each gaussian component -- indexed by the trailing axis of
    `amplitudes`/`variances` -- via `spread_one(amplitude, variance,
    n_spread)`, and sum the results. Used by `GaussianMixtureVolume` (where
    the leading axes are `(n_positions,)`).

    If `n_spread` is a single `int`, `spread_one` is `vmap`'d over the
    `n_gaussians` axis, sharing one compiled/traced call across every
    component -- fast, but every component, however narrow or wide, is
    spread with the same width.

    If `n_spread` is a `tuple` of length `n_gaussians` (one value per
    component, e.g. from [`cryojax.simulator.suggest_n_spread`][]), components
    are grouped by their shared `n_spread` value, and `spread_one` is
    `vmap`'d once per distinct value across that group -- not once per
    component. This is necessary to have per-component spread widths at all
    -- not just an optimization choice -- because `n_spread` fixes array
    shapes inside `spread_one` (see `cryojax.ndimage.spread_gaussians_2d`/
    `spread_gaussians_3d`), so components with different `n_spread` cannot
    share one `vmap`'d/traced call. But components that happen to share the
    same `n_spread` (common -- e.g. X-ray/electron scattering factors written
    as a sum of 5 Gaussians often have several components of similar width)
    still get to share a single `vmap`'d call, rather than each getting its
    own fully unrolled `spread_one` call.
    """
    n_gaussians = amplitudes.shape[-1]
    if isinstance(n_spread, int):
        contributions = jax.vmap(
            lambda a, v: spread_one(a, v, n_spread), in_axes=(-1, -1)
        )(amplitudes, variances)
        return jnp.sum(contributions, axis=0)
    if len(n_spread) != n_gaussians:
        raise ValueError(
            f"`n_spread` was a tuple of length {len(n_spread)}, but the "
            f"number of gaussian components was {n_gaussians}. These must "
            "be equal (one `n_spread` value per gaussian component)."
        )
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(n_spread):
        groups.setdefault(n, []).append(i)

    def _spread_group(n: int, indices: list[int]) -> Array:
        idx = jnp.asarray(indices)
        contributions = jax.vmap(lambda a, v: spread_one(a, v, n), in_axes=(-1, -1))(
            amplitudes[..., idx], variances[..., idx]
        )
        return jnp.sum(contributions, axis=0)

    group_totals = [_spread_group(n, indices) for n, indices in groups.items()]
    total = group_totals[0]
    for group_total in group_totals[1:]:
        total = total + group_total
    return total
