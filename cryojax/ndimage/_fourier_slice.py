"""
Fourier projection-slice extraction from voxel grids.

These are the low-level building blocks used by
`cryojax.simulator.FourierSliceExtraction` and
`cryojax.simulator.EwaldSphereExtraction`, exposed here as a public API for
directly extracting slices and Ewald sphere surfaces from a fourier-space
voxel grid.
"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..jax_util import FloatLike
from ._coordinates import make_1d_coordinate_grid, make_1d_frequency_grid
from ._edges import pad_to_shape
from ._fourier_utils import make_fftshift_phase, query_efficient_grid_size
from ._map_coordinates import (
    compute_spline_coefficients,
    map_coordinates,
    map_coordinates_spline,
)


def sample_fft_slice(
    voxels_or_spline: (
        Complex[Array, "dim dim dim//2+1"] | Complex[Array, "dim+2 dim+2 dim//2+3"]
    ),
    /,
    frequency_slice: Float[Array, "1 dim dim//2+1 3"] | Float[Array, "1 dim dim 3"],
    *,
    use_spline: bool = False,
    out_of_bounds_mode: str = "fill",
    unroll_gather: bool = True,
) -> Complex[Array, "dim _"]:
    """Extract a surface from a fourier-space voxel grid using the
    Fourier projection-slice theorem.

    Given a set of 3D frequency coordinates lying on a surface (a central
    slice, or a curved Ewald sphere surface), interpolate the voxel grid onto
    those coordinates. The voxel grid is assumed to be stored in the
    half-space (rfft) convention along its last axis, with the two full axes
    `fftshift`ed to the center convention (see the example below and
    `cryojax.simulator.FourierVoxelGridVolume`).

    The output is returned in the rfft/DC-in-corner convention, so it can be
    passed directly to `cryojax.ndimage.irfftn` (for a half slice) or
    `cryojax.ndimage.ifftn` (for a full Ewald sphere surface).

    !!! example "Preparing the voxel grid and extracting a central slice"

        The voxel grid must be prepared with the convention used internally by
        `cryojax.simulator.FourierVoxelGridVolume`: `fftshift` the object in
        real space, then `fftshift` the two full axes in fourier space.
        `cryojax.ndimage.prepare_sampling_fft` does this for you.

        ```python
        import cryojax.ndimage as im

        # A cubic, even-dimension voxel grid in real space
        real_voxel_grid = ...  # shape (dim, dim, dim)
        dim = real_voxel_grid.shape[0]

        fourier_voxel_grid = im.prepare_sampling_fft(real_voxel_grid)

        # The (unrotated) central-slice coordinate system, zero-centered
        frequency_slice = im.make_frequency_slice((dim, dim), fftshifted=True)

        # Extract the slice and transform back to a real-space projection
        projection_fft = im.sample_fft_slice(fourier_voxel_grid, frequency_slice)
        projection = jnp.fft.irfftn(projection_fft, s=(dim, dim))
        ```

    !!! example "Rotating a central slice with `cryojax.rotations.SO3`"

        ```python
        import jax
        import cryojax.ndimage as im
        from cryojax.rotations import SO3

        # A half (rfft) central slice of shape (1, dim, dim//2+1, 3)
        frequency_slice = im.make_frequency_slice((dim, dim), fftshifted=True)

        # Rotate the coordinate system by a random rotation.
        rotation = SO3.sample_uniform(jax.random.key(0))
        rotated_slice = rotation.apply(frequency_slice)

        projection_fft = im.sample_fft_slice(fourier_voxel_grid, rotated_slice)
        ```

    **Arguments:**

    - `voxels_or_spline`:
        The fourier-space voxel grid, truncated to the half-space
        `(dim, dim, dim // 2 + 1)` and prepared as described above. If
        `use_spline` is `True`, this should instead be the spline
        coefficients, of shape `(dim + 2, dim + 2, dim // 2 + 3)`, computed
        by `cryojax.ndimage.compute_spline_coefficients`.
    - `frequency_slice`:
        The 3D frequency coordinates to interpolate onto, in pixel units.
        This can either be a half (rfft) central slice of shape
        `(1, dim, dim // 2 + 1, 3)`, as returned by
        `cryojax.ndimage.make_frequency_slice`, or a full Ewald sphere
        surface of shape `(1, dim, dim, 3)`, as returned by
        `cryojax.ndimage.ewald_sphere_from_slice`.
    - `use_spline`:
        If `True`, `voxels_or_spline` is interpreted as spline coefficients
        and interpolated with `cryojax.ndimage.map_coordinates_spline`.
        Otherwise, linear interpolation is used via
        `cryojax.ndimage.map_coordinates`.
    - `out_of_bounds_mode`:
        Specify how to handle out-of-bounds indexing. See
        `cryojax.ndimage.map_coordinates` for documentation.
    - `unroll_gather`:
        Passed to `cryojax.ndimage.map_coordinates` /
        `cryojax.ndimage.map_coordinates_spline`.

    **Returns:**

    The extracted surface in the rfft/DC-in-corner convention. Shape
    `(dim, dim // 2 + 1)` for a half central slice, or `(dim, dim)` for a
    full Ewald sphere surface.
    """
    # Convert to logical coordinates
    N = frequency_slice.shape[1]
    if N % 2 != 0:
        raise ValueError(
            "`sample_fft_slice` does not support odd dimensions, but "
            f"got a `frequency_slice` of dimension `{N}`. Please use a voxel "
            "grid and `frequency_slice` with even dimensions."
        )
    # Validate that `voxels_or_spline` matches `use_spline`: a raw fourier
    # voxel grid is stored at `(N, N, N // 2 + 1)`, while spline coefficients
    # carry a `+ 2` pad on every axis, i.e. `(N + 2, N + 2, N // 2 + 3)`.
    expected_shape = (N + 2, N + 2, N // 2 + 3) if use_spline else (N, N, N // 2 + 1)
    if voxels_or_spline.shape != expected_shape:
        raise ValueError(
            f"`sample_fft_slice` got input with shape "
            f"`{voxels_or_spline.shape}` and `use_spline={use_spline}`, but for "
            f"a `frequency_slice` of dimension `{N}` this array is expected to "
            f"have shape `{expected_shape}`. "
            + (
                "Did you mean to pass `use_spline=False` (a raw fourier voxel grid)?"
                if use_spline
                else "Did you mean to pass `use_spline=True` (spline "
                "coefficients from `compute_spline_coefficients`)?"
            )
        )
    # `voxels_or_spline`'s last axis only stores non-negative frequencies
    # along x (i.e. `F(-q) = conj(F(q))` is not stored, only `F(q)` for
    # `q_x >= 0`). Reflect the whole 3-vector through the origin whenever
    # `q_x < 0`, so we always look up a point with `q_x >= 0`, then conjugate
    # the interpolated result to correct for it. This is exact, not an
    # approximation: it's evaluated once per query point, on the continuous
    # coordinate, before any interpolation taps are generated, so taps never
    # straddle the truncation boundary.
    sign = jnp.where(frequency_slice[..., 0] < 0, -1.0, 1.0)
    reflected = sign[..., None] * frequency_slice
    k_x = reflected[..., 0] * N  # rfft/corner convention: no N // 2 offset
    k_y = reflected[..., 1] * N + N // 2
    k_z = reflected[..., 2] * N + N // 2
    # The centered axes' Nyquist bin (frequency -0.5) is stored only at
    # index 0, not also at index N -- +0.5 and -0.5 are the same (aliased)
    # physical frequency. Reflecting a coordinate that was exactly at -0.5
    # lands exactly on index N, one past the valid range; wrap that exact
    # case back to index 0, without touching any other (genuinely
    # out-of-bounds) coordinate.
    k_y = jnp.where(k_y == N, 0.0, k_y)
    k_z = jnp.where(k_z == N, 0.0, k_z)
    kwargs: dict[str, Any] = {
        "out_of_bounds_mode": out_of_bounds_mode,
        "unroll_gather": unroll_gather,
    }
    if use_spline:
        surface = map_coordinates_spline(voxels_or_spline, (k_z, k_y, k_x), **kwargs)[
            0, :, :
        ]
    else:
        surface = map_coordinates(voxels_or_spline, (k_z, k_y, k_x), **kwargs)[0, :, :]
    surface = jnp.where(sign[0, :, :] < 0, jnp.conj(surface), surface)
    # FFT shift and multiply by (-1)^k phase factors. `surface` is itself
    # rfft-shaped only when `frequency_slice` was (i.e. only for the
    # half in-plane slice -- an Ewald sphere surface is always a full grid),
    # in which case only the first axis is shifted, mirroring the same
    # convention used for the 3D volume storage.
    if surface.shape[0] == surface.shape[1]:
        surface = jnp.fft.ifftshift(make_fftshift_phase(surface.shape) * surface)
    else:
        surface = jnp.fft.ifftshift(
            make_fftshift_phase((N, N), outputs_rfft=True) * surface, axes=(0,)
        )

    return surface


def prepare_sampling_fft(
    real_voxel_grid: Float[Array, "dim dim dim"],
    *,
    apply_deconvolve: bool = True,
    pad_scale: float = 1.0,
    use_spline: bool = False,
) -> Complex[Array, "dim dim dim//2+1"] | Complex[Array, "dim+2 dim+2 dim//2+3"]:
    """Transform a real-space voxel grid into the fourier-space array
    consumed by `cryojax.ndimage.sample_fft_slice`.

    This performs the same preprocessing used internally by
    `cryojax.simulator.FourierVoxelGridVolume` and
    `cryojax.simulator.FourierVoxelSplineVolume`: optional Fourier-padding and
    sinc² deconvolution, followed by the `rfftn` and the `fftshift`s that put
    the grid in the center convention (see `sample_fft_slice`).

    !!! example

        ```python
        import cryojax.ndimage as im

        real_voxel_grid = ...  # shape (dim, dim, dim)

        fourier_voxel_grid = im.prepare_sampling_fft(real_voxel_grid)
        frequency_slice = im.make_frequency_slice(
            fourier_voxel_grid.shape[:2], fftshifted=True
        )
        projection = im.sample_fft_slice(
            fourier_voxel_grid, frequency_slice
        )
        ```

    **Arguments:**

    - `real_voxel_grid`:
        A cubic, even-dimension voxel grid in real space.
    - `apply_deconvolve`:
        If `True`, divide out the sinc² transfer function of the trilinear
        interpolation kernel, for more accurate linear-interpolation slice
        extraction. The correction uses the (post-padding) Fourier grid size.
        Ignored when `use_spline=True`, since the sinc² correction only
        compensates for trilinear interpolation.
    - `pad_scale`:
        Scale factor at which to Fourier-pad `real_voxel_grid` before the
        transform. Must be a value `>= 1.0`.
    - `use_spline`:
        If `True`, return the cubic-spline coefficients (of shape
        `(dim + 2, dim + 2, dim // 2 + 3)`) computed by
        `cryojax.ndimage.compute_spline_coefficients`, ready for
        `sample_fft_slice(..., use_spline=True)`. Otherwise return
        the raw fourier voxel grid, of shape `(dim, dim, dim // 2 + 1)`.

    **Returns:**

    The prepared fourier-space array. `dim` is the (possibly padded)
    dimension, i.e. `real_voxel_grid.shape[0]` when `pad_scale == 1.0`.
    """
    real_voxel_grid = jnp.asarray(real_voxel_grid, dtype=float)
    if real_voxel_grid.ndim != 3 or len(set(real_voxel_grid.shape)) != 1:
        raise ValueError(
            "`prepare_sampling_fft` only supports cubic voxel grids, but "
            f"got `real_voxel_grid.shape = {real_voxel_grid.shape}`."
        )
    if real_voxel_grid.shape[0] % 2 != 0:
        raise ValueError(
            "`prepare_sampling_fft` does not support odd voxel grid "
            f"dimensions, but got `real_voxel_grid.shape = {real_voxel_grid.shape}`. "
            "Please pass a voxel grid with even dimensions."
        )
    # Fourier-pad to a query-efficient, even size before transforming.
    if pad_scale == 1.0:
        real_voxel_grid_padded = real_voxel_grid
    elif pad_scale > 1.0:
        padded_shape = query_efficient_grid_size(
            real_voxel_grid.shape, pad_scale=pad_scale, only_even=True
        )
        real_voxel_grid_padded = pad_to_shape(real_voxel_grid, padded_shape)
    else:
        raise ValueError(
            "Invalid value for `prepare_sampling_fft(..., pad_scale=...)`. "
            f"This must be a value `>= 1.0`, but got value `{pad_scale}`."
        )
    # Deconvolve after padding so the sinc² correction uses the actual
    # Fourier grid size, not the original unpadded size. The sinc² correction
    # only compensates for trilinear interpolation, so it is ignored for
    # spline coefficients.
    if apply_deconvolve and not use_spline:
        real_voxel_grid_padded = _deconvolve_linear(real_voxel_grid_padded)
    fourier_voxel_grid = _fftshift_fourier_voxel_grid(
        jnp.fft.rfftn(real_voxel_grid_padded)
    )
    if use_spline:
        return compute_spline_coefficients(fourier_voxel_grid)
    return fourier_voxel_grid


def ewald_sphere_from_slice(
    frequency_slice: Float[Array, "1 dim dim//2+1 3"],
    voxel_size: FloatLike,
    wavelength: FloatLike,
) -> Float[Array, "1 dim dim 3"]:
    """Curve a central slice onto the Ewald sphere surface.

    Take a half (rfft) central-slice coordinate system, reconstruct the full
    in-plane grid, and displace each in-plane frequency out of the plane onto
    the curved Ewald sphere surface. The result can be passed as the
    `frequency_slice` argument of
    `cryojax.ndimage.sample_fft_slice`.

    !!! example

        ```python
        import cryojax.ndimage as im

        # A half (rfft) central slice from `make_frequency_slice`
        frequency_slice = im.make_frequency_slice((dim, dim), fftshifted=True)

        frequency_surface = im.ewald_sphere_from_slice(
            frequency_slice, voxel_size, wavelength
        )
        surface = im.sample_fft_slice(fourier_voxel_grid, frequency_surface)
        ```

    **Arguments:**

    - `frequency_slice`:
        The half (rfft) central-slice coordinate system of shape
        `(1, dim, dim // 2 + 1, 3)`, as returned by
        `cryojax.ndimage.make_frequency_slice`. This will typically be a
        rotated slice.
    - `voxel_size`:
        The voxel size, in units of length.
    - `wavelength`:
        The electron wavelength, in units of length.

    **Returns:**

    The Ewald sphere surface coordinates of shape `(1, dim, dim, 3)`.
    """
    # The Ewald sphere surface curves the in-plane slice out of its own plane,
    # so its output isn't Hermitian-symmetric as a whole and every output
    # pixel is queried independently -- reconstruct the full in-plane grid
    # from the stored half one before curving.
    full_frequency_slice = _reconstruct_full_slice_from_half_slice(frequency_slice)
    return _get_ewald_sphere_surface_from_slice(
        full_frequency_slice,
        jnp.asarray(voxel_size, dtype=float),
        jnp.asarray(wavelength, dtype=float),
    )


def _reconstruct_full_slice_from_half_slice(
    half_slice: Float[Array, "1 dim dim//2+1 3"],
) -> Float[Array, "1 dim dim 3"]:
    """Reconstruct the full in-plane frequency grid from the half (rfft) one
    stored on the volume, for `EwaldSphereExtraction`, which needs the full
    grid to compute its local `xhat`/`yhat`/`zhat` basis and to produce its
    curved, non-Hermitian-symmetric-as-a-whole output surface.

    A rotated in-plane grid is an exactly linear function of the (unrotated)
    local x-coordinate, for any fixed y: `slice(x, y) = x * xhat_rot +
    slice(0, y)`, where `xhat_rot` is a single constant vector (the rotated
    local x unit vector). `xhat_rot` is recovered from any two adjacent
    columns of the half grid, then used to extrapolate every column of the
    full grid. This is exact (no interpolation, and no reflection of array
    indices, so no boundary case at the row-Nyquist frequency -- unlike a
    literal point-reflection, this only ever reads columns that are already
    stored in `half_slice`).
    """
    N = half_slice.shape[1]
    xhat_rot = N * (half_slice[:, :, 1, :] - half_slice[:, :, 0, :])
    x_full = make_1d_frequency_grid(N, outputs_rfftfreqs=False, fftshifted=True)
    x_term = x_full[None, None, :, None] * xhat_rot[:, :, None, :]
    return half_slice[:, :, 0:1, :] + x_term


def _get_ewald_sphere_surface_from_slice(
    frequency_slice_in_pixels: Array, voxel_size: Array, wavelength: Array
) -> Float[Array, "1 dim dim 3"]:
    frequency_slice_with_zero_in_corner = jnp.fft.ifftshift(
        frequency_slice_in_pixels, axes=(0, 1, 2)
    )
    # Get zhat unit vector of the frequency slice
    xhat, yhat = (
        frequency_slice_with_zero_in_corner[0, 0, 1, :],
        frequency_slice_with_zero_in_corner[0, 1, 0, :],
    )
    xhat, yhat = xhat / jnp.linalg.norm(xhat), yhat / jnp.linalg.norm(yhat)
    zhat = jnp.cross(xhat, yhat)
    # Compute the ewald sphere surface, assuming the frequency slice is
    # in a rotated frame
    q_at_slice = frequency_slice_in_pixels
    q_squared = jnp.sum(q_at_slice**2, axis=-1)
    q_at_surface = (
        q_at_slice
        + (wavelength / voxel_size)
        * (q_squared[..., None] * zhat[None, None, None, :])
        / 2
    )
    return q_at_surface


def _fftshift_fourier_voxel_grid(
    fourier_voxel_grid: Complex[Array, "dim dim dim//2+1"],
) -> Complex[Array, "dim dim dim//2+1"]:
    """Put an `rfftn` voxel grid in the center convention expected by
    `sample_fft_slice`.
    """
    dim = fourier_voxel_grid.shape[0]
    # Truncated (last) axis stays in rfft/corner convention -- only the two
    # full axes get fftshift'd to center convention. The phase realizes the
    # equivalent real-space fftshift on the object.
    phase = make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
    return jnp.fft.fftshift(phase * fourier_voxel_grid, axes=(0, 1))


def _deconvolve_linear(
    real_voxel_grid: Float[Array, "dim dim dim"],
) -> Float[Array, "dim dim dim"]:
    """Deconvolve the effect of the trilinear interpolation kernel."""
    dim = real_voxel_grid.shape[0]
    x = make_1d_coordinate_grid(dim)
    sinc_array = jnp.sinc(x / dim)
    deconvolve_factor = (
        sinc_array[:, None, None] * sinc_array[None, :, None] * sinc_array[None, None, :]
    ) ** 2
    return real_voxel_grid / deconvolve_factor
