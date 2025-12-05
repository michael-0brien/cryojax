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
        See the function`enforce_self_conjugate_rfftn_components`
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
        rfftn_array = enforce_self_conjugate_rfftn_components(
            rfftn_array,
            shape,  # type: ignore
            includes_zero_frequency=False,
            mode=mode,
        )
    return rfftn_array


def enforce_self_conjugate_rfftn_components(
    rfftn_array: (
        Complex[Array, "{shape[0]} {shape[1]}//2+1"]
        | Complex[Array, "{shape[0]} {shape[1]} {shape[2]}//2+1"]
    ),
    shape: tuple[int, int] | tuple[int, int, int],
    includes_zero_frequency: bool = False,
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
    - `includes_zero_frequency`:
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
            "`enforce_self_conjugate_rfftn_components`. "
            "The supported modes are 'zero', 'one', and 'real'."
        )
    if rfftn_array.ndim == 2:
        assert len(shape) == 2
        y_dim, x_dim = shape
        if includes_zero_frequency:
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
        if includes_zero_frequency:
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
            "in function `enforce_self_conjugate_rfftn_components`. "
            f"Passed an array with `ndim = {rfftn_array.ndim}`."
        )
    return rfftn_array
