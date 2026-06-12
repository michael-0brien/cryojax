import math
from typing import Literal

from jaxtyping import Array, Complex


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
    kwargs = dict(mode="promise_in_bounds", indices_are_sorted=True, unique_indices=True)
    if fftn_array.ndim == 2:
        rfftn_array = fftn_array.at[:, : shape[-1] // 2 + 1].get(
            **kwargs  # pyright: ignore[reportArgumentType]
        )
    elif fftn_array.ndim == 3:
        rfftn_array = fftn_array.at[:, :, : shape[-1] // 2 + 1].get(
            **kwargs  # pyright: ignore[reportArgumentType]
        )
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
    if mode == "zero":
        replace_fn = lambda _: 0.0
    elif mode == "one":
        replace_fn = lambda _: 1.0
    elif mode == "real":
        replace_fn = lambda arr: arr.real
    else:
        raise NotImplementedError(
            f"`mode = {mode}` not supported for function "
            "`enforce_rfftn_self_conjugates`. "
            "The supported modes are 'zero', 'one', and 'real'."
        )
    if rfftn_array.ndim == 2:
        assert len(shape) == 2
        y_dim, x_dim = shape
        if includes_dc:
            rfftn_array = rfftn_array.at[0, 0].set(replace_fn(rfftn_array[0, 0]))
        if y_dim % 2 == 0:
            rfftn_array = rfftn_array.at[y_dim // 2, 0].set(
                replace_fn(rfftn_array[y_dim // 2, 0])
            )
        if x_dim % 2 == 0:
            rfftn_array = rfftn_array.at[0, x_dim // 2].set(
                replace_fn(rfftn_array[0, x_dim // 2])
            )
        if y_dim % 2 == 0 and x_dim % 2 == 0:
            rfftn_array = rfftn_array.at[y_dim // 2, x_dim // 2].set(
                replace_fn(rfftn_array[y_dim // 2, x_dim // 2])
            )
    elif rfftn_array.ndim == 3:
        assert len(shape) == 3
        z_dim, y_dim, x_dim = shape
        if includes_dc:
            rfftn_array = rfftn_array.at[0, 0, 0].set(replace_fn(rfftn_array[0, 0, 0]))
        if z_dim % 2 == 0:
            rfftn_array = rfftn_array.at[0, z_dim // 2, 0].set(
                replace_fn(rfftn_array[0, z_dim // 2, 0])
            )
        if y_dim % 2 == 0:
            rfftn_array = rfftn_array.at[0, y_dim // 2, 0].set(
                replace_fn(rfftn_array[0, y_dim // 2, 0])
            )
        if x_dim % 2 == 0:
            rfftn_array = rfftn_array.at[0, 0, x_dim // 2].set(
                replace_fn(rfftn_array[0, 0, x_dim // 2])
            )
        if y_dim % 2 == 0 and x_dim % 2 == 0:
            rfftn_array = rfftn_array.at[0, y_dim // 2, x_dim // 2].set(
                replace_fn(rfftn_array[0, y_dim // 2, x_dim // 2])
            )
        if z_dim % 2 == 0 and x_dim % 2 == 0:
            rfftn_array = rfftn_array.at[z_dim // 2, 0, x_dim // 2].set(
                replace_fn(rfftn_array[z_dim // 2, 0, x_dim // 2])
            )
        if z_dim % 2 == 0 and y_dim % 2 == 0:
            rfftn_array = rfftn_array.at[z_dim // 2, y_dim // 2, 0].set(
                replace_fn(rfftn_array[z_dim // 2, y_dim // 2, 0])
            )
        if z_dim % 2 == 0 and y_dim % 2 == 0 and x_dim % 2 == 0:
            rfftn_array = rfftn_array.at[z_dim // 2, y_dim // 2, x_dim // 2].set(
                replace_fn(rfftn_array[z_dim // 2, y_dim // 2, x_dim // 2])
            )
    else:
        raise NotImplementedError(
            "Only 2D and 3D arrays are supported "
            "in function `enforce_rfftn_self_conjugates`. "
            f"Passed an array with `ndim = {rfftn_array.ndim}`."
        )
    return rfftn_array


def query_efficient_grid_size(
    shape: tuple[int, ...], pad_scale: float = 1.0, match_parity: bool = True
) -> tuple[int, ...]:
    """Select an efficient grid size for FFT"""

    if match_parity:
        is_same_parity = lambda n1, n2: (n1 % 2) == (n2 % 2)
    else:
        is_same_parity = lambda *_: True

    padded_shape = tuple(int(math.ceil(pad_scale * s)) for s in shape)

    def next_smooth_int(n: int, nf: int) -> int:
        # Check if n is already smooth
        def is_smooth(x):
            for p in [2, 3, 5]:
                while x % p == 0:
                    x //= p
            return x == 1

        # Find next smooth number
        candidate = nf
        while not (is_smooth(candidate) and is_same_parity(n, candidate)):
            candidate += 1
        return candidate

    return tuple(next_smooth_int(n, nf) for n, nf in zip(shape, padded_shape))


# def _make_fftshift_phase(
#     shape: tuple[int, ...],
#     axes: tuple[int, ...] | None = None,
# ) -> Array:
#     """Build the `(-1)^(k1+k2+...)` sign pattern for the full (non-real) FFT.

#     Multiplying `jnp.fft.fftn(x)` by this pattern gives the same result as
#     `jnp.fft.fftn(jnp.fft.ifftshift(x))`, so the output has DC at corner
#     (modeord=0 convention). Then `jnp.fft.fftshift` moves DC back to center
#     if needed for storage.

#     Only exact for even-sized dimensions.

#     **Arguments:**

#     - `shape`: shape of the FFT output array (same as input when s=None).
#     - `axes`: axes to include; defaults to all axes.

#     **Returns:**

#     Broadcastable ±1 array with the same number of dimensions as `shape`.
#     """
#     ndim = len(shape)
#     if axes is None:
#         axes = tuple(range(ndim))
#     else:
#         axes = tuple(ax % ndim for ax in axes)
#     phase = jnp.ones(())
#     for ax in axes:
#         n = shape[ax]
#         p = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
#         reshape = [1] * ndim
#         reshape[ax] = n
#         phase = phase * p.reshape(reshape)
#     return phase
