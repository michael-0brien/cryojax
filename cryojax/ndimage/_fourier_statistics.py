"""
Helper routines to compute power spectra.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ._fourier_utils import make_rfftn_multiplicity
from ._radial_average import compute_binned_radial_average


def compute_binned_powerspectrum(
    fourier_grid: (Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"]),
    radial_frequency_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    grid_spacing: float | Float[Array, ""] = 1.0,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
    real_shape: tuple[int, ...] | None = None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"]]:
    """Compute the power spectrum of an image or volume averaged on a set
    of radial bins.

    !!! warning
        `radial_frequency_grid` must be built *without* a `grid_spacing`, so
        that it is in units of inverse grid spacings:

        ```python
        from cryojax.ndimage import make_radial_frequency_grid
        import jax.numpy as jnp

        image, pixel_size = ...
        radial_frequency_grid = make_radial_frequency_grid(image.shape)
        powerspectrum, frequency_bins = compute_binned_powerspectrum(
            jnp.fft.rfftn(image), radial_frequency_grid, pixel_size
        )
        ```

        The returned `frequency_bins` are in inverse angstroms, so that they
        may be plotted directly. This is also true of the other fourier
        statistics in `cryojax.ndimage`, such as
        `compute_fourier_shell_correlation`.

    **Arguments:**

    - `fourier_grid`:
        An image or volume in Fourier space.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of `fourier_grid`, built
        without a `grid_spacing`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel or voxel size) of `fourier_grid`. This
        converts the returned `frequency_bins` to inverse angstroms, and does
        not otherwise affect the binning.
    - `minimum_frequency`:
        Minimum frequency bin, in the same units as `radial_frequency_grid`.
        By default, `0.0`.
    - `maximum_frequency`:
        Maximum frequency bin, in the same units as `radial_frequency_grid`.
        By default, `math.sqrt(2) / 2`.
    - `real_shape`:
        The real-space shape of `fourier_grid`, which is an
        `rfftn` array. This is used to weight each mode by its Hermitian
        multiplicity so that the radial average matches a full-grid average.
        If `None`, the last (real-transformed) axis is assumed to be even.

    **Returns:**

    A tuple of the radially averaged power spectrum and the frequency bins,
    in inverse angstroms, over which it is computed.
    """
    # Compute squared amplitudes
    squared_fourier_amplitudes = (fourier_grid * jnp.conjugate(fourier_grid)).real
    # Compute bins in units of inverse grid spacings, so that they share units
    # with `radial_frequency_grid`
    frequency_bins = _make_radial_frequency_bins(
        fourier_grid.shape, minimum_frequency, maximum_frequency
    )
    # Compute radially averaged power spectrum as a 1D profile, weighting each
    # rfft mode by its Hermitian multiplicity
    radially_binned_powerspectrum = compute_binned_radial_average(
        squared_fourier_amplitudes,
        radial_frequency_grid,
        frequency_bins,
        weights=_make_binning_weights(fourier_grid.shape, real_shape),
    )

    # ... return the bins in inverse angstroms, so they may be plotted directly
    return radially_binned_powerspectrum, frequency_bins / grid_spacing


def compute_fourier_ring_correlation(
    fourier_image_1: Inexact[Array, "y_dim x_dim"],
    fourier_image_2: Inexact[Array, "y_dim x_dim"],
    radial_frequency_grid: Float[Array, "y_dim x_dim"],
    pixel_size: float | Float[Array, ""] = 1.0,
    threshold: float | Float[Array, ""] = 0.5,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
    real_shape: tuple[int, ...] | None = None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"], Float[Array, ""]]:
    """Compute the fourier ring correlation for two images.

    **Arguments:**

    - `fourier_image_1`:
        An image in fourier space, e.g. the output of `jax.numpy.fft.rfftn`.
    - `fourier_image_2`:
        Another image in fourier space. See documentation for `fourier_image_1`
        for conventions.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of the images, in pixel units.
    - `pixel_size`:
        The pixel size of the images. This converts the returned
        `frequency_bins` and `frequency_threshold` to inverse angstroms, and
        does not otherwise affect the binning.
    - `threshold`:
        The threshold at which to draw the distinction between input images.
    - `minimum_frequency`:
        Minimum frequency bin, in pixel units. By default, `0.0`.
    - `maximum_frequency`:
        Maximum frequency bin, in pixel units. By default, `math.sqrt(2) / 2`.
    - `real_shape`:
        The real-space shape of the `rfftn` input images. Used to weight
        each mode by its Hermitian multiplicity. If `None`, the last
        (real-transformed) axis is assumed to be even.

    **Returns:**

    - `frc_curve`:
        The fourier ring correlations as a function of `frequency_bins`.
    - `frequency_bins`:
        The frequencies, in inverse angstroms, for which we have calculated
        the correlations.
    - `frequency_threshold`:
        The frequency, in inverse angstroms, at which the correlation drops
        below the specified threshold.
    """
    frc_curve, frequency_bins, frequency_threshold = _compute_fourier_correlation(
        fourier_image_1,
        fourier_image_2,
        radial_frequency_grid,
        pixel_size,
        threshold=threshold,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        real_shape=real_shape,
    )
    return frc_curve, frequency_bins, frequency_threshold


def compute_fourier_shell_correlation(
    fourier_volume_1: Inexact[Array, "z_dim y_dim x_dim"],
    fourier_volume_2: Inexact[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: Float[Array, "z_dim y_dim x_dim"],
    voxel_size: float | Float[Array, ""] = 1.0,
    threshold: float | Float[Array, ""] = 0.5,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
    real_shape: tuple[int, ...] | None = None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"], Float[Array, ""]]:
    """Compute the fourier shell correlation for two voxel maps.

    !!! warning
        `radial_frequency_grid` must be in voxel units, i.e. it must be built
        without a grid spacing:

        ```python
        from cryojax.ndimage import make_radial_frequency_grid
        import jax.numpy as jnp

        volume_1, volume_2, voxel_size = ...
        radial_frequency_grid = make_radial_frequency_grid(volume_1.shape)
        fsc_curve, frequency_bins, frequency_threshold = (
            compute_fourier_shell_correlation(
                jnp.fft.rfftn(volume_1),
                jnp.fft.rfftn(volume_2),
                radial_frequency_grid,
                voxel_size,
            )
        )
        ```

        The returned `frequency_bins` and `frequency_threshold` are in inverse
        angstroms, so that they may be plotted directly.

    **Arguments:**

    - `fourier_volume_1`:
        A volume in fourier space, e.g. the output of `jax.numpy.fft.rfftn`
        (so the zero-frequency component is in the corner).
    - `fourier_volume_2`:
        Another volume in fourier space. See documentation for
        `fourier_volume_1` for conventions.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of the volumes, in voxel units.
    - `voxel_size`:
        The voxel size of the volumes. This converts the returned
        `frequency_bins` and `frequency_threshold` to inverse angstroms, and
        does not otherwise affect the binning.
    - `threshold`:
        The threshold at which to draw the distinction between input maps.
        By default, `threshold = 0.5` for two 'known' volumes according to
        the half-bit criterion. If using half-maps derived from ab initio
        refinements, set `threshold = 0.143` by convention.
    - `minimum_frequency`:
        Minimum frequency bin, in voxel units. By default, `0.0`.
    - `maximum_frequency`:
        Maximum frequency bin, in voxel units. By default, `math.sqrt(2) / 2`.
    - `real_shape`:
        The real-space shape of the `rfftn` input volumes. Used to weight
        each mode by its Hermitian multiplicity. If `None`, the last
        (real-transformed) axis is assumed to be even.

    **Returns:**

    - `fsc_curve`:
        The fourier shell correlations as a function of `frequency_bins`.
    - `frequency_bins`:
        The frequencies, in inverse angstroms, for which we have calculated
        the correlations.
    - `frequency_threshold`:
        The frequency, in inverse angstroms, at which the correlation drops
        below the specified threshold.
    """
    fsc_curve, frequency_bins, frequency_threshold = _compute_fourier_correlation(
        fourier_volume_1,
        fourier_volume_2,
        radial_frequency_grid,
        voxel_size,
        threshold=threshold,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        real_shape=real_shape,
    )
    return fsc_curve, frequency_bins, frequency_threshold


def _compute_fourier_correlation(
    fourier_grid_1: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    fourier_grid_2: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"],
    radial_frequency_grid: (
        Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]
    ),
    grid_spacing: float | Float[Array, ""],
    threshold: float | Float[Array, ""],
    minimum_frequency: float,
    maximum_frequency: float,
    real_shape: tuple[int, ...] | None,
) -> tuple[Float[Array, " n_bins"], Float[Array, " n_bins"], Float[Array, ""]]:
    # Compute FSC/FRC radially averaged 1D profile
    correlation_map = (
        (fourier_grid_1 * jnp.conjugate(fourier_grid_2))
        / jnp.sqrt(jnp.abs(fourier_grid_1) ** 2 * jnp.abs(fourier_grid_2) ** 2)
    ).real
    # ... bins in units of inverse grid spacings, so that they share units
    # with `radial_frequency_grid`
    frequency_bins = _make_radial_frequency_bins(
        fourier_grid_1.shape, minimum_frequency, maximum_frequency
    )
    correlation_curve = compute_binned_radial_average(
        correlation_map,
        radial_frequency_grid,
        frequency_bins,
        weights=_make_binning_weights(fourier_grid_1.shape, real_shape),
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
    # ... the threshold is read off the bins by index, so converting the bins to
    # inverse angstroms puts the threshold in inverse angstroms too
    frequency_bins = frequency_bins / grid_spacing
    frequency_threshold = frequency_bins[threshold_crossing_index]

    return correlation_curve, frequency_bins, frequency_threshold


def _make_radial_frequency_bins(shape, minimum_frequency, maximum_frequency):
    # Bins are in units of inverse grid spacings, matching
    # `radial_frequency_grid`. They sit at
    # half-integer multiples of the step, starting at `1.5`. Modes on an axis
    # have `|k| = k / N`, an exact multiple of the step, so bins on those same
    # multiples would land right on top of them. Offsetting by half a step avoids
    # that. Starting at `1.5` rather than `0.5` keeps the first bin from holding
    # the zero-frequency mode alone, which `_make_binning_weights` excludes.
    q_min, q_max = minimum_frequency, maximum_frequency
    q_step = 1.0 / max(*shape)
    n_bins = 1 + int((q_max - q_min) / q_step)
    return q_min + q_step * (jnp.arange(n_bins) + 1.5)


def _make_binning_weights(rfftn_shape, real_shape):
    # Weight each mode of an `rfftn` array by its Hermitian multiplicity, and
    # drop the zero-frequency mode: that mode is the mean of the image, not part
    # of its power spectrum, and it would otherwise dominate the lowest bin.
    if real_shape is None:
        # ... assume the last (real-transformed) axis is even, so that the final
        # rfft column is a self-conjugate Nyquist mode rather than an ordinary
        # conjugate pair
        real_shape = (*rfftn_shape[:-1], 2 * (rfftn_shape[-1] - 1))
    multiplicity = make_rfftn_multiplicity(real_shape)
    is_zero_mode = jnp.ones(rfftn_shape, dtype=bool)
    for axis, n in enumerate(rfftn_shape):
        shape_along_axis = [1] * len(rfftn_shape)
        shape_along_axis[axis] = n
        is_zero_mode &= jnp.arange(n).reshape(shape_along_axis) == 0

    return jnp.where(is_zero_mode, 0.0, multiplicity)
