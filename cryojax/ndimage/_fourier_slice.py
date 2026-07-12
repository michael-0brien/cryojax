"""
Fourier projection-slice extraction from voxel grids.

These are the low-level building blocks used by
`cryojax.simulator.FourierSliceExtraction` and
`cryojax.simulator.EwaldSphereExtraction`, exposed here as a public API for
directly extracting slices and Ewald sphere surfaces from a fourier-space
voxel grid.
"""

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..jax_util import FloatLike
from ._coordinates import make_1d_frequency_grid
from ._edges import pad_to_shape
from ._fourier_utils import make_fftshift_phase, query_efficient_grid_size
from ._interpolation import (
    deconvolve_interpolation_kernel,
    map_frequencies,
    parse_interp,
)


def sample_fft_slice(
    sampling_fft: Complex[Array, "dim dim dim//2+1"],
    /,
    frequency_slice: Float[Array, "1 dim dim//2+1 3"] | Float[Array, "1 dim dim 3"],
    *,
    interp: Literal["linear", "cubic"] = "linear",
    boundary: str = "fill",
    unroll: bool = True,
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

        sampling_fft = im.prepare_sampling_fft(real_voxel_grid)

        # The (unrotated) central-slice coordinate system, zero-centered
        frequency_slice = im.make_frequency_slice((dim, dim), fftshifted=True)

        # Extract the slice and transform back to a real-space projection
        projection_fft = im.sample_fft_slice(sampling_fft, frequency_slice)
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

        projection_fft = im.sample_fft_slice(sampling_fft, rotated_slice)
        ```

    **Arguments:**

    - `sampling_fft`:
        The fourier-space voxel grid, truncated to the half-space
        `(dim, dim, dim // 2 + 1)` and prepared as described above, i.e. by
        `cryojax.ndimage.prepare_sampling_fft`.
    - `frequency_slice`:
        The 3D frequency coordinates to interpolate onto, in pixel units.
        This can either be a half (rfft) central slice of shape
        `(1, dim, dim // 2 + 1, 3)`, as returned by
        `cryojax.ndimage.make_frequency_slice`, or a full Ewald sphere
        surface of shape `(1, dim, dim, 3)`, as returned by
        `cryojax.ndimage.ewald_sphere_from_slice`.
    - `interp`:
        The interpolation method, either `"linear"` or `"cubic"`. This **must**
        match the `interp` that `sampling_fft` was prepared with --- see
        `cryojax.ndimage.prepare_sampling_fft`.
    - `boundary`:
        What to return for frequencies that fall outside the fourier box, which
        happens at the corners of a rotated slice and for Ewald sphere surfaces
        curving out of it. Either `"fill"` (the default), which returns zero, or
        `"clip"`, which clamps them onto the edge of the box.
    - `unroll`:
        See `cryojax.ndimage.map_coordinates`. For `interp="cubic"`,
        `unroll=False` is often substantially faster on GPU.

    **Returns:**

    The extracted surface in the rfft/DC-in-corner convention. Shape
    `(dim, dim // 2 + 1)` for a half central slice, or `(dim, dim)` for a
    full Ewald sphere surface.
    """
    order, _ = parse_interp(interp)
    # Convert to logical coordinates
    N = frequency_slice.shape[1]
    if N % 2 != 0:
        raise ValueError(
            "`sample_fft_slice` does not support odd dimensions, but "
            f"got a `frequency_slice` of dimension `{N}`. Please use a voxel "
            "grid and `frequency_slice` with even dimensions."
        )
    expected_shape = (N, N, N // 2 + 1)
    if sampling_fft.shape != expected_shape:
        raise ValueError(
            f"`sample_fft_slice` got a `sampling_fft` with shape "
            f"`{sampling_fft.shape}`, but for a `frequency_slice` of "
            f"dimension `{N}` it is expected to have the half-space (rfft) "
            f"shape `{expected_shape}`."
        )
    surface = map_frequencies(
        sampling_fft,
        frequency_slice,
        order=order,
        mode=boundary,
        unroll=unroll,
    )[0, :, :]
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
    interp: Literal["linear", "cubic"] = "linear",
    pad_scale: float = 1.0,
) -> Complex[Array, "dim dim dim//2+1"]:
    """Transform a real-space voxel grid into the fourier-space array
    consumed by `cryojax.ndimage.sample_fft_slice`.

    This is the preprocessing that `cryojax.simulator.FourierVoxelGridVolume`
    does internally: optional padding, deconvolution of the interpolation
    kernel, and the transform itself.

    !!! info "Why deconvolution?"

        Interpolating a fourier voxel grid does not return the volume's true
        fourier transform, but that of the volume blurred by the interpolation
        kernel. The blur is known in closed form (`sinc^2` for `"linear"`,
        `sinc^4` for `"cubic"`), so it is divided out of the voxel grid here,
        *before* the transform. Slice extraction then reconstructs the true
        transform, rather than an approximation of it, and costs nothing extra at
        sampling time.

        What is left is aliasing, which shrinks with `pad_scale`.

    !!! example

        ```python
        import cryojax.ndimage as im

        real_voxel_grid = ...  # shape (dim, dim, dim)

        sampling_fft = im.prepare_sampling_fft(real_voxel_grid)
        frequency_slice = im.make_frequency_slice(
            sampling_fft.shape[:2], fftshifted=True
        )
        projection = im.sample_fft_slice(
            sampling_fft, frequency_slice
        )
        ```

    **Arguments:**

    - `real_voxel_grid`:
        A cubic, even-dimension voxel grid in real space.
    - `interp`:
        The interpolation method the returned grid is prepared for. The same
        value must be passed to `cryojax.ndimage.sample_fft_slice`. Either
        `"linear"` (the default), or `"cubic"`, which is substantially more
        accurate at the cost of reading a `4^3` rather than a `2^3` neighborhood
        per query point.
    - `pad_scale`:
        Scale factor at which to Fourier-pad `real_voxel_grid` before the
        transform. Must be a value `>= 1.0`.

    **Returns:**

    The prepared fourier voxel grid, of shape `(dim, dim, dim // 2 + 1)`.
    `dim` is the (possibly padded) dimension, i.e. `real_voxel_grid.shape[0]`
    when `pad_scale == 1.0`.
    """
    _, sinc_power = parse_interp(interp)
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
    # Deconvolve after padding so the sinc correction uses the actual Fourier
    # grid size, not the original unpadded size.
    if sinc_power > 0:
        real_voxel_grid_padded = deconvolve_interpolation_kernel(
            real_voxel_grid_padded, sinc_power
        )
    return _fftshift_sampling_fft(jnp.fft.rfftn(real_voxel_grid_padded))


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
        surface = im.sample_fft_slice(sampling_fft, frequency_surface)
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
    full_frequency_slice = _full_slice_from_half_slice(frequency_slice)
    return _ewald_sphere_from_slice(
        full_frequency_slice,
        jnp.asarray(voxel_size, dtype=float),
        jnp.asarray(wavelength, dtype=float),
    )


def _full_slice_from_half_slice(
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


def _ewald_sphere_from_slice(
    frequency_slice: Array, voxel_size: Array, wavelength: Array
) -> Float[Array, "1 dim dim 3"]:
    frequency_slice_with_zero_in_corner = jnp.fft.ifftshift(
        frequency_slice, axes=(0, 1, 2)
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
    q_at_slice = frequency_slice
    q_squared = jnp.sum(q_at_slice**2, axis=-1)
    q_at_surface = (
        q_at_slice
        + (wavelength / voxel_size)
        * (q_squared[..., None] * zhat[None, None, None, :])
        / 2
    )
    return q_at_surface


def _fftshift_sampling_fft(
    sampling_fft: Complex[Array, "dim dim dim//2+1"],
) -> Complex[Array, "dim dim dim//2+1"]:
    """Put an `rfftn` voxel grid in the center convention expected by
    `sample_fft_slice`.
    """
    dim = sampling_fft.shape[0]
    # Truncated (last) axis stays in rfft/corner convention -- only the two
    # full axes get fftshift'd to center convention. The phase realizes the
    # equivalent real-space fftshift on the object.
    phase = make_fftshift_phase((dim, dim, dim), outputs_rfft=True)
    return jnp.fft.fftshift(phase * sampling_fft, axes=(0, 1))
