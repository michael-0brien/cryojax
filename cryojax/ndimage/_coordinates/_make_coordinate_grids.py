"""
Functions for creating coordinate systems.
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def make_coordinate_grid(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
) -> Float[Array, "*shape ndim"]:
    """
    Create a real-space cartesian coordinate system on a grid.

    **Arguments:**

    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.

    **Returns:**

    A cartesian coordinate system in real space.
    """
    coordinate_grid = _make_coordinates_or_frequencies(
        shape, grid_spacing=grid_spacing, outputs_real_space=True
    )
    return coordinate_grid


def make_radial_coordinate_grid(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
) -> Float[Array, " *shape"]:
    """Create a real-space radial coordinate system on a grid.

    This wraps the function `make_coordinate_grid` to compute
    the coordinate vector magnitude.

    **Arguments:**

    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.

    **Returns:**

    A radial coordinate system in real space.
    """
    # Make a cartesian grid
    coordinate_grid = make_coordinate_grid(
        shape=shape,
        grid_spacing=grid_spacing,
    )
    # Now compute the magnitude of the coordinate vector
    radial_coordinate_grid = jnp.linalg.norm(coordinate_grid, axis=-1)

    return radial_coordinate_grid


def make_frequency_grid(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    outputs_rfftfreqs: bool = True,
    fftshifted: bool = False,
) -> Float[Array, "*shape ndim"]:
    """Create a fourier-space cartesian coordinate system on a grid.
    The zero-frequency component is in the corner.

    **Arguments:**

    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.
    - `outputs_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    A cartesian coordinate system in frequency space.
    """
    frequency_grid = _make_coordinates_or_frequencies(
        shape,
        grid_spacing=grid_spacing,
        outputs_real_space=False,
        outputs_rfftfreqs=outputs_rfftfreqs,
        fftshifted=fftshifted,
    )
    return frequency_grid


def make_radial_frequency_grid(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    outputs_rfftfreqs: bool = True,
    fftshifted: bool = False,
) -> Float[Array, " *shape"]:
    """Create a fourier-space radial coordinate system on a grid.
    The zero-frequency component is in the corner.

    This wraps the function `make_frequency_grid` to compute
    the frequency magnitude, which is a common use case for
    things like computing fourier shell correlations and power spectrums.

    **Arguments:**

    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.
    - `outputs_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    A radial coordinate system in frequency space.
    """
    # Make a cartesian grid
    frequency_grid = make_frequency_grid(
        shape=shape,
        grid_spacing=grid_spacing,
        outputs_rfftfreqs=outputs_rfftfreqs,
        fftshifted=fftshifted,
    )

    # Now compute the magnitude of the frequency vector
    radial_frequency_grid = jnp.linalg.norm(frequency_grid, axis=-1)

    return radial_frequency_grid


def make_frequency_slice(
    shape: tuple[int, int],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    outputs_rfftfreqs: bool = True,
    fftshifted: bool = True,
) -> Float[Array, "1 {shape[0]} {shape[1]} 3"]:
    """Create central slice frequency coordinates. By default,
    returns in the convention required for usage with
    [`cryojax.ndimage.sample_rfft_surface`][].

    **Arguments:**

    - `shape`:
        Shape of the frequency slice, e.g. `shape = (100, 100)`.
    - `grid_spacing`:
        The grid spacing (i.e. voxel size), in units of length.
    - `outputs_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    The central, $q_z = 0$ slice of a 3D frequency grid $(q_x, q_y, q_z)$, where
    zero-frequency component is in the *center* of the grid.
    """  # noqa: E501
    frequency_slice = make_frequency_grid(
        shape, grid_spacing, outputs_rfftfreqs=outputs_rfftfreqs, fftshifted=fftshifted
    )
    frequency_slice = jnp.expand_dims(
        jnp.pad(
            frequency_slice,
            ((0, 0), (0, 0), (0, 1)),
            mode="constant",
            constant_values=0.0,
        ),
        axis=0,
    )
    return frequency_slice


def make_1d_coordinate_grid(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    fftshifted: bool = False,
) -> Float[Array, "*shape ndim"]:
    """
    Create a 1D real-space cartesian coordinate array.

    **Arguments:**

    - `size`:
        Size of the coordinate array.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.

    **Returns:**

    A 1D cartesian coordinate array in real space.
    """
    coordinate_array = _make_coordinates_or_frequencies_1d(
        size, grid_spacing=grid_spacing, outputs_real_space=True, fftshifted=fftshifted
    )
    return coordinate_array


def make_1d_frequency_grid(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    outputs_rfftfreqs: bool = True,
    fftshifted: bool = False,
) -> Float[Array, "*shape ndim"]:
    """Create a 1D fourier-space cartesian coordinate array.
    If `outputs_rfftfreqs = False`, the zero-frequency component is in the beginning.

    Arguments
    ---------
    - `size`:
        Size of the coordinate array.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.
    - `outputs_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    A 1D cartesian coordinate array in frequency space.
    """
    frequency_array = _make_coordinates_or_frequencies_1d(
        size,
        grid_spacing=grid_spacing,
        outputs_real_space=False,
        outputs_rfftfreqs=outputs_rfftfreqs,
        fftshifted=fftshifted,
    )
    return frequency_array


def _make_coordinates_or_frequencies(
    shape: tuple[int, ...],
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""] = 1.0,
    outputs_real_space: bool = False,
    outputs_rfftfreqs: bool = True,
    fftshifted: bool = False,
) -> Float[Array, "*shape ndim"]:
    ndim = len(shape)
    coords1D = []
    for idx in range(ndim):
        if outputs_real_space:
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, outputs_real_space, fftshifted=fftshifted
            )
        else:
            if not outputs_rfftfreqs:
                rfftfreq = False
            else:
                rfftfreq = False if idx < ndim - 1 else True
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx],
                grid_spacing,
                outputs_real_space,
                rfftfreq,
                fftshifted=fftshifted,
            )
        coords1D.append(c1D)
    if ndim == 2:
        y, x = coords1D
        xv, yv = jnp.meshgrid(x, y, indexing="xy")
        coords = jnp.stack([xv, yv], axis=-1)
    elif ndim == 3:
        z, y, x = coords1D
        xv, yv, zv = jnp.meshgrid(x, y, z, indexing="xy")
        xv, yv, zv = [
            jnp.transpose(rv, axes=[2, 0, 1]) for rv in [xv, yv, zv]
        ]  # Change axis ordering to [z, y, x]
        coords = jnp.stack([xv, yv, zv], axis=-1)
    else:
        raise ValueError(
            "Only 2D and 3D coordinate grids are supported. "
            f"Tried to create a grid of shape {shape}."
        )

    return coords


def _make_coordinates_or_frequencies_1d(
    size: int,
    grid_spacing: float | Float[np.ndarray, ""] | Float[Array, ""],
    outputs_real_space: bool = False,
    outputs_rfftfreqs: bool | None = None,
    fftshifted: bool = False,
) -> Float[Array, " size"]:
    """One-dimensional coordinates in real or fourier space"""
    if outputs_real_space:
        make_1d = lambda size, dx: jnp.fft.fftshift(jnp.fft.fftfreq(size, 1 / dx) * size)
    else:
        if outputs_rfftfreqs is None:
            raise ValueError("Internal error in `cryojax.coordinates`.")
        else:
            if outputs_rfftfreqs:
                fn = jnp.fft.rfftfreq
            else:
                if fftshifted:
                    fn = lambda *x: jnp.fft.fftshift(jnp.fft.fftfreq(*x))
                else:
                    fn = jnp.fft.fftfreq
            make_1d = lambda size, dx: fn(size, dx)

    return make_1d(size, grid_spacing)
