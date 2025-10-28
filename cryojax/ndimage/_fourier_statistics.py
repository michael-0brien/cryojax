"""
Helper routines to compute power spectra.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ._radial_average import compute_binned_radial_average


def compute_binned_powerspectrum(
    fourier_image_or_volume: (
        Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]
    ),
    radial_frequency_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    pixel_or_voxel_size: float | Float[Array, ""] = 1.0,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]:
    """Compute the power spectrum of an image averaged on a set
    of radial bins.

    !!! warning
        If `radial_frequency_grid` is passed in inverse angstroms, then
        it is necessary to also pass the `pixel_size` argument.

        ```python
        from cryojax.coordinates import make_radial_frequency_grid
        from cryojax.image import rfftn

        image, pixel_size = ...
        fourier_image = rfftn(image)
        radial_frequency_grid_in_angstroms = make_radial_frequency_grid(
            image.shape, pixel_size, outputs_rfftfreqs=True
        )
        powerspectrum, bins = compute_binned_powerspectrum(
            rfftn(image), radial_frequency_grid_in_angstroms, pixel_size
        )
        ```

        This is also true for other fourier statistics in `cryojax.image`,
        such as the `fourier_shell_correlation` function.

    **Arguments:**

    - `fourier_image_or_volume`:
        An image or volume in Fourier space.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of `fourier_image_or_volume`.
    - `pixel_or_voxel_size`:
        The pixel or voxel size of `fourier_image_or_volume`. If
        `radial_frequency_grid` is passed in inverse angstroms,
        this argument must be included.
    - `minimum_frequency`:
        Minimum frequency bin. By default, `0.0`. This is
        not measured in inverse angstroms, even if `radial_frequency_grid`
        is in inverse angstroms.
    - `maximum_frequency`:
        Maximum frequency bin. By default, `math.sqrt(2) / 2`. This is
        not measured in inverse angstroms, even if `radial_frequency_grid`
        is in inverse angstroms.

    **Returns:**

    A tuple of the radially averaged power spectrum and the frequency bins
    over which it is computed.
    """
    # Compute squared amplitudes
    squared_fourier_amplitudes = (
        fourier_image_or_volume * jnp.conjugate(fourier_image_or_volume)
    ).real
    # Compute bins
    frequency_bins = _make_radial_frequency_bins(
        fourier_image_or_volume.shape,
        minimum_frequency,
        maximum_frequency,
        pixel_or_voxel_size,
    )
    # Compute radially averaged power spectrum as a 1D profile
    radially_binned_powerspectrum = compute_binned_radial_average(
        squared_fourier_amplitudes, radial_frequency_grid, frequency_bins
    )

    return radially_binned_powerspectrum, frequency_bins


def compute_fourier_ring_correlation(
    fourier_image_1: Inexact[Array, "y_dim x_dim"],
    fourier_image_2: Inexact[Array, "y_dim x_dim"],
    radial_frequency_grid: Float[Array, "y_dim x_dim"],
    pixel_size: float | Float[Array, ""] = 1.0,
    threshold: float | Float[Array, ""] = 0.5,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"], Float[Array, ""]]:
    """Compute the fourier ring correlation for two images.

    **Arguments:**

    - `fourier_image_1`:
        An image in fourier space. For example, this can be from the output of
        `cryojax.image.rfftn`.
    - `fourier_image_2`:
        Another image in fourier space. See documentation for `image_1`
        for conventions.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of the images.
    - `pixel_size`:
        The pixel of the images. If `radial_frequency_grid` is passed in
        inverse angstroms, this argument must be included.
    - `threshold`:
        The threshold at which to draw the distinction between input images.
    - `minimum_frequency`:
        Minimum frequency bin. By default, `0.0`. This is
        not measured in inverse angstroms, even if `radial_frequency_grid`
        is in inverse angstroms.
    - `maximum_frequency`:
        Maximum frequency bin. By default, `math.sqrt(2) / 2`. This is
        not measured in inverse angstroms, even if `radial_frequency_grid`
        is in inverse angstroms.

    **Returns:**

    - `frc_curve`:
        The fourier ring correlations as a function of `frequency_bins`.
    - `frequency_bins`:
        The array of frequencies for which we have calculated the
        correlations.
    - `frequency_threshold`:
        The frequency at which the correlation drops below the
        specified threshold.
    """
    frc_curve, frequency_bins, frequency_threshold = _compute_fourier_correlation(
        fourier_image_1,
        fourier_image_2,
        radial_frequency_grid,
        pixel_size,
        threshold=threshold,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return frc_curve, frequency_bins, frequency_threshold


def compute_fourier_shell_correlation(
    volume_1: Inexact[Array, "z_dim y_dim x_dim"],
    volume_2: Inexact[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: Float[Array, "z_dim y_dim x_dim"],
    voxel_size: float | Float[Array, ""] = 1.0,
    threshold: float | Float[Array, ""] = 0.5,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"], Float[Array, ""]]:
    """Compute the fourier shell correlation for two voxel maps.

    **Arguments:**

    - `volume_1`:
        A volume in real or fourier space. If it is in fourier space,
        it should be from the output of `cryojax.image.rfftn` (e.g.
        the zero-frequency component should be in the corner). If it
        is in real-space, it cannot be complex-valued.
    - `volume_2`:
        A volume in real or fourier space. See documentation for `volume_1`
        for conventions.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of the volumes.
    - `voxel_size`:
        The voxel size of the volumes. If `radial_frequency_grid` is passed
        in inverse angstroms, this argument must be included.
    - `threshold`:
        The threshold at which to draw the distinction between input maps.
        By default, `threshold = 0.5` for two 'known' volumes according to
        the half-bit criterion. If using half-maps derived from ab initio
        refinements, set `threshold = 0.143` by convention.

    **Returns:**

    - `fsc_curve`:
        The fourier shell correlations as a function of `frequency_bins`.
    - `frequency_bins`:
        The array of frequencies for which we have calculated the
        correlations.
    - `frequency_threshold`:
        The frequencies at which the correlation drops below the
        specified threshold.

    !!! warning

        It is common to obtain a `frequency_threshold` given in inverse angstroms.
        This function achieves this behavior if the `voxel_size` argument is passed
        and the `radial_frequency_grid` argument is given in inverse angstroms. For
        example,

        ```python
        from cryojax.coordinates import make_radial_frequency_grid
        from cryojax.image import rfftn

        volume_1, volume_2, voxel_size = ...
        fourier_volume_1, fourier_volume_2 = rfftn(volume_1), rfftn(volume_2)
        radial_frequency_grid = make_radial_frequency_grid(
            volume_1.shape, voxel_size, outputs_rfftfreqs=True
        )
        fsc_curve, frequency_bins, frequency_threshold = compute_fourier_shell_correlation(
            fourier_volume_1, fourier_volume_2, radial_frequency_grid, voxel_size
        )
        ```

        This behavior can be similarly achieved by leaving out the `voxel_size` of
        the functions `make_radial_frequency_grid` and `compute_fourier_shell_correlation`
        and computing `frequency_bins / voxel_size` and `frequency_threshold / voxel_size`.
    """  # noqa: E501
    fsc_curve, frequency_bins, frequency_threshold = _compute_fourier_correlation(
        volume_1,
        volume_2,
        radial_frequency_grid,
        voxel_size,
        threshold=threshold,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return fsc_curve, frequency_bins, frequency_threshold


def _compute_fourier_correlation(
    fourier_array_1: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    fourier_array_2: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    grid_spacing: float | Float[Array, ""],
    threshold: float | Float[Array, ""],
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"], Float[Array, ""]]:
    # Compute FSC/FRC radially averaged 1D profile
    correlation_map = (
        (fourier_array_1 * jnp.conjugate(fourier_array_2))
        / jnp.sqrt(jnp.abs(fourier_array_1) ** 2 * jnp.abs(fourier_array_2) ** 2)
    ).real
    frequency_bins = _make_radial_frequency_bins(
        fourier_array_1.shape, minimum_frequency, maximum_frequency, grid_spacing
    )
    correlation_curve = compute_binned_radial_average(
        correlation_map,
        radial_frequency_grid,
        frequency_bins,
    )
    # Find where FSC/FRC drops below the specified threshold
    # TODO: Add van heel criterion.
    where_below_threshold = jnp.where(
        correlation_curve < threshold, 0, 1
    )  # 0s when below, 1s, when above
    # ... find minimum index where we flip from 0 to 1
    where_is_crossing = jnp.diff(where_below_threshold)
    # ... make an array that has a value of its index when we have a crossing, and a dummy
    # value otherwise
    arr_size = where_is_crossing.size
    arr_indices = jnp.arange(arr_size, dtype=int)
    dummy_index = arr_size + 100
    indices_at_0_to_1_flips = jnp.where(where_is_crossing == -1, arr_indices, dummy_index)
    # ... get minimum of array
    threshold_crossing_index = jnp.amin(indices_at_0_to_1_flips) + 1
    frequency_threshold = frequency_bins[threshold_crossing_index]

    return correlation_curve, frequency_bins, frequency_threshold


def _make_radial_frequency_bins(shape, minimum_frequency, maximum_frequency, pixel_size):
    q_min, q_max = minimum_frequency, maximum_frequency
    q_step = 1.0 / max(*shape)
    n_bins = 1 + int((q_max - q_min) / q_step)
    return jnp.linspace(q_min, q_max, n_bins) / pixel_size
