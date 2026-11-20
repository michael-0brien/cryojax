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

from ..jax_util import FloatLike, NDArrayLike
from ._edges import pad_to_shape
from ._fourier_utils import make_fftshift_phase, query_efficient_grid_size
from ._interpolation import (
    deconvolve_interpolation_kernel,
    map_frequencies,
    parse_interp,
)


def sample_fft_slice(
    sampling_fft: Complex[NDArrayLike, "dim dim dim//2+1"],
    frequency_slice: Float[NDArrayLike, "1 dim dim//2+1 3"],
    *,
    interp: Literal["linear", "cubic"] = "linear",
    boundary: str = "fill",
    unroll: bool = True,
) -> Complex[Array, "dim dim//2+1"]:
    """Extract a surface from a fourier-space voxel grid using the
    Fourier projection-slice theorem.

    Given a set of 3D frequency coordinates lying on a surface (a central
    slice, or a curved Ewald sphere surface), interpolate the voxel grid onto
    those coordinates. The voxel grid is assumed to be stored in the
    half-space (rfft) convention along its last axis, with the two full axes
    `fftshift`ed to the center convention (see the example below and
    `cryojax.simulator.FourierVoxelGridVolume`).

    The output is returned in the rfft/DC-in-corner convention, so it can be
    passed directly to `jax.numpy.fft.irfftn`.

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
        The 3D frequency coordinates to interpolate onto, in pixel units, of shape
        `(1, dim, dim // 2 + 1, 3)`: a half (rfft) central slice, as returned by
        `cryojax.ndimage.make_frequency_slice`, or that slice curved by
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

    The extracted surface in the rfft/DC-in-corner convention, of shape
    `(dim, dim // 2 + 1)`.
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
    sampling_fft, frequency_slice = (
        jnp.asarray(sampling_fft),
        jnp.asarray(frequency_slice),
    )
    surface = map_frequencies(
        sampling_fft,
        (
            frequency_slice[..., 2],
            frequency_slice[..., 1],
            frequency_slice[..., 0],
        ),
        order=order,
        mode=boundary,
        unroll=unroll,
    )[0, :, :]
    # FFT shift the first axis, the only full one, as in the 3D volume storage, and
    # multiply by (-1)^k phase factors.
    #
    # The phase is `(-1)^k` in the *frequency* index `k`, so it must be applied
    # only once `surface` is in the corner (DC-at-zero) convention, i.e. after
    # the shift. Applying it to a centered axis instead reads index `k + N // 2`
    # and contributes a spurious global `(-1)^(N // 2)` per centered axis.
    return make_fftshift_phase((N, N), outputs_rfft=True) * jnp.fft.ifftshift(
        surface, axes=(0,)
    )


def prepare_sampling_fft(
    real_voxel_grid: Float[NDArrayLike, "dim dim dim"],
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
        projection_fft = im.sample_fft_slice(sampling_fft, frequency_slice)
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
) -> Float[Array, "1 dim dim//2+1 3"]:
    """Curve a half (rfft) central slice onto the Ewald sphere surface, displacing
    each in-plane frequency `q` by `-(wavelength / voxel_size) * |q|**2 / 2` along the
    slice normal. The result can be passed as the `frequency_slice` argument of
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

    The Ewald sphere surface coordinates of shape `(1, dim, dim // 2 + 1, 3)`.
    """
    voxel_size = jnp.asarray(voxel_size, dtype=float)
    wavelength = jnp.asarray(wavelength, dtype=float)
    # The slice's unit normal, from its steps along x and y at zero frequency. They
    # need not be orthogonal (e.g. at frequencies `Dᵀk`): divide by their angle's sine
    frequency_slice_with_zero_in_corner = jnp.fft.ifftshift(frequency_slice, axes=(1,))
    xhat, yhat = (
        frequency_slice_with_zero_in_corner[0, 0, 1, :],
        frequency_slice_with_zero_in_corner[0, 1, 0, :],
    )
    xhat, yhat = xhat / jnp.linalg.norm(xhat), yhat / jnp.linalg.norm(yhat)
    zhat = jnp.cross(xhat, yhat) / jnp.sqrt(1 - jnp.sum(xhat * yhat) ** 2)
    q_squared = jnp.sum(frequency_slice**2, axis=-1, keepdims=True)
    return frequency_slice - (wavelength / voxel_size) * q_squared * zhat / 2


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
