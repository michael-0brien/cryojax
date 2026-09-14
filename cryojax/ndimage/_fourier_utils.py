import math
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def convert_fftn_to_rfftn(
    fftn_array: Complex[Array, "y_dim x_dim"] | Complex[Array, "z_dim y_dim x_dim"],
    mode: Literal["zero", "one", "real"] | None = "zero",
) -> Complex[Array, "y_dim x_dim//2+1"] | Complex[Array, "z_dim y_dim x_dim//2+1"]:
    """Converts the output of a call to `jax.numpy.fft.fftn` to
    an `jax.numpy.fft.rfftn`.

    The FFT $F$ of a real-valued function obeys hermitian
    symmetry, i.e.

    $$F^*(k) = F(-k).$$

    Therefore, to convert an `fftn` output to that which would
    be returned by an `rfftn`, take the upper-half plane of
    an `fftn_array`. Also, optionally take care to make sure that
    self-conjugate components are purely real-valued.

    **Arguments:**

    - `fftn_array`:
        The output of a call to `jax.numpy.fft.fftn`.
    - `mode`:
        See the function`enforce_rfftn_self_conjugates`
        for documentation. If this is `None`, do not call this
        function.


    **Returns:**

    The `fftn_array`, as if it were the output of a call
    to `cryojax.image.rfftn` function.
    """
    shape = fftn_array.shape
    # Take upper half plane
    if fftn_array.ndim == 2:
        rfftn_array = fftn_array[:, : shape[-1] // 2 + 1]
    elif fftn_array.ndim == 3:
        rfftn_array = fftn_array[:, :, : shape[-1] // 2 + 1]
    else:
        raise NotImplementedError(
            "Only 2D and 3D arrays are supported "
            "in function `convert_fftn_to_rfftn`. "
            f"Passed an array with `ndim = {fftn_array.ndim}`."
        )
    if mode is not None:
        rfftn_array = enforce_rfftn_self_conjugates(
            rfftn_array,
            shape,  # type: ignore
            includes_dc=False,
            mode=mode,
        )
    return rfftn_array


def enforce_rfftn_self_conjugates(
    rfftn_array: (
        Complex[Array, "{shape[0]} {shape[1]}//2+1"]
        | Complex[Array, "{shape[0]} {shape[1]} {shape[2]}//2+1"]
    ),
    shape: tuple[int, int] | tuple[int, int, int],
    includes_dc: bool = False,
    mode: Literal["zero", "one", "real"] = "zero",
) -> (
    Complex[Array, "{shape[0]} {shape[1]}//2+1"]
    | Complex[Array, "{shape[0]} {shape[1]} {shape[2]}//2+1"]
):
    """For an array that is the output of a call to an "rfftn"
    function, enforce that self-conjugate components are real-valued.

    By default, do this by setting them to zero. This is important
    before applying translational phase shifts to an image in fourier space.

    **Arguments:**

    - `rfftn_array`:
        An array that is the output of a call to an
        "rfftn" function. This must have the zero-frequency
        component in the corner.
    - `shape`:
        The shape of the `rfftn_array` in real-space.
    - `includes_dc`:
        If `True`, enforce that `rfftn_array[0, 0]` is real.
        Otherwise, leave this component unmodified.
    - `mode`:
        A string controlling how the components are made
        real-valued. Supported modes are

        - "zero": sets components to zero
        - "one": sets components to one
        - "real": takes real part of components

        By default, `mode = "zero"`.

    **Return:**

    The modified `rfftn_array`, with self-conjugate components
    made real-valued.
    """
    if mode not in ("zero", "one", "real"):
        raise NotImplementedError(
            f"`mode = {mode}` not supported for function "
            "`enforce_rfftn_self_conjugates`. "
            "The supported modes are 'zero', 'one', and 'real'."
        )
    if rfftn_array.ndim == 2:
        assert len(shape) == 2
        y_dim, x_dim = shape
        y_rows, x_cols = rfftn_array.shape
        row_idx = jnp.arange(y_rows)
        col_idx = jnp.arange(x_cols)
        row_is_sc = row_idx == 0
        if y_dim % 2 == 0:
            row_is_sc = row_is_sc | (row_idx == y_dim // 2)
        col_is_sc = col_idx == 0
        if x_dim % 2 == 0:
            col_is_sc = col_is_sc | (col_idx == x_dim // 2)
        sc_mask = row_is_sc[:, None] & col_is_sc[None, :]
        if not includes_dc:
            sc_mask = sc_mask & ~((row_idx == 0)[:, None] & (col_idx == 0)[None, :])
    elif rfftn_array.ndim == 3:
        assert len(shape) == 3
        z_dim, y_dim, x_dim = shape
        z_slices, y_rows, x_cols = rfftn_array.shape
        z_idx = jnp.arange(z_slices)
        row_idx = jnp.arange(y_rows)
        col_idx = jnp.arange(x_cols)
        z_is_sc = z_idx == 0
        if z_dim % 2 == 0:
            z_is_sc = z_is_sc | (z_idx == z_dim // 2)
        row_is_sc = row_idx == 0
        if y_dim % 2 == 0:
            row_is_sc = row_is_sc | (row_idx == y_dim // 2)
        col_is_sc = col_idx == 0
        if x_dim % 2 == 0:
            col_is_sc = col_is_sc | (col_idx == x_dim // 2)
        sc_mask = (
            z_is_sc[:, None, None] & row_is_sc[None, :, None] & col_is_sc[None, None, :]
        )
        if not includes_dc:
            sc_mask = sc_mask & ~(
                (z_idx == 0)[:, None, None]
                & (row_idx == 0)[None, :, None]
                & (col_idx == 0)[None, None, :]
            )
    else:
        raise NotImplementedError(
            "Only 2D and 3D arrays are supported "
            "in function `enforce_rfftn_self_conjugates`. "
            f"Passed an array with `ndim = {rfftn_array.ndim}`."
        )
    if mode == "zero":
        rfftn_array = jnp.where(sc_mask, 0.0, rfftn_array)
    elif mode == "one":
        rfftn_array = jnp.where(sc_mask, 1.0, rfftn_array)
    else:  # mode == "real"
        rfftn_array = jnp.where(sc_mask, rfftn_array.real, rfftn_array)
    return rfftn_array


def make_rfftn_multiplicity(
    shape: tuple[int, ...],
) -> Float[Array, " {shape[-1]}//2+1"]:
    """Multiplicity of each mode in an `rfftn` output.

    `jax.numpy.fft.rfftn` stores only the non-negative frequencies along the
    last axis, relying on Hermitian symmetry $F(-k) = F^*(k)$ for the rest.
    Each retained mode along that axis therefore represents either one or two
    modes of the full `fftn` grid:

    - the zero-frequency column represents a single mode (multiplicity 1),
    - for an even-sized last axis, the Nyquist column is self-conjugate and
      also has multiplicity 1,
    - every other column represents a conjugate pair (multiplicity 2).

    Weighting by this multiplicity recovers full-grid reductions (such as an
    L2 norm or a variance) from an `rfftn` array. The other axes are not
    reduced by `rfftn`, so the returned array varies only along the last axis
    and broadcasts against the leading axes.

    **Arguments:**

    - `shape`:
        The shape of the array in real space.

    **Returns:**

    A 1D array of length `shape[-1] // 2 + 1` giving the multiplicity of each
    mode along the last (real-transformed) axis.
    """
    width = shape[-1]
    multiplicity = jnp.full(width // 2 + 1, 2.0).at[0].set(1.0)
    if width % 2 == 0:
        multiplicity = multiplicity.at[-1].set(1.0)
    return multiplicity


def query_efficient_grid_size(
    shape: tuple[int, ...], pad_scale: float = 1.0, only_even: bool = False
) -> tuple[int, ...]:
    """Select an efficient grid size for FFT"""

    padded_shape = tuple(int(math.ceil(pad_scale * s)) for s in shape)

    def next_smooth_int(nf: int) -> int:
        def is_smooth(x):
            for p in [2, 3, 5]:
                while x % p == 0:
                    x //= p
            return x == 1

        candidate = nf
        while not (is_smooth(candidate) and (not only_even or candidate % 2 == 0)):
            candidate += 1
        return candidate

    return tuple(next_smooth_int(nf) for nf in padded_shape)


def make_fftshift_phase(
    shape: tuple[int, ...],
    axes: tuple[int, ...] | None = None,
    outputs_rfft: bool = False,
) -> Array:
    """Build the `(-1)^(k1+k2+...)` sign pattern for the full (non-real) FFT.

    Multiplying `jnp.fft.fftn(x)` by this pattern gives the same result as
    `jnp.fft.fftn(jnp.fft.ifftshift(x))`, so the output has DC at corner
    (modeord=0 convention). Then `jnp.fft.fftshift` moves DC back to center
    if needed for storage.

    Only exact for even-sized dimensions.

    **Arguments:**

    - `shape`: shape of the FFT output array (same as input when s=None).
    - `axes`: axes to include; defaults to all axes.

    **Returns:**

    Broadcastable ±1 array with the same number of dimensions as `shape`.
    """
    ndim = len(shape)
    if axes is None:
        axes = tuple(range(ndim))
    else:
        axes = tuple(ax % ndim for ax in axes)
    phase = jnp.ones(())
    for idx, ax in enumerate(axes):
        n = shape[ax]
        if outputs_rfft and idx == len(axes) - 1:
            indices = jnp.arange(n // 2 + 1)
        else:
            indices = jnp.arange(n)
        p = jnp.where(indices % 2 == 0, 1.0, -1.0)
        reshape = [1] * ndim
        reshape[ax] = indices.shape[0]
        phase = phase * p.reshape(reshape)
    return phase
