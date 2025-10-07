"""
Routines for working with MRC files.
"""

import pathlib
from typing import Literal, cast, overload

import mrcfile
import numpy as np
from jaxtyping import Array, Float


@overload
def read_array_from_mrc(
    filename: str | pathlib.Path,
    *,
    loads_grid_spacing: Literal[False],
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."]: ...


@overload
def read_array_from_mrc(  # type: ignore
    filename: str | pathlib.Path,
    *,
    loads_grid_spacing: Literal[True],
    mmap: bool = False,
    permissive: bool = False,
) -> tuple[Float[np.ndarray, "..."], float]: ...


@overload
def read_array_from_mrc(
    filename: str | pathlib.Path,
    *,
    loads_grid_spacing: bool = False,
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."]: ...


def read_array_from_mrc(
    filename: str | pathlib.Path,
    *,
    loads_grid_spacing: bool = False,
    mmap: bool = False,
    permissive: bool = False,
) -> Float[np.ndarray, "..."] | tuple[Float[np.ndarray, "..."], float]:
    """Read an MRC data into a a numpy array.

    **Arguments:**

    - `filename` : Path to data.
    - `mmap`: Whether or not to open the data as a `numpy.memmap` array.
    - `loads_grid_spacing`: If `True`, load the pixel or voxel size of the array.

    **Returns:**

    - 'array' : The array stored in the Mrcfile.
    """
    # Validate filename as MRC path
    _ = _validate_filename_and_return_suffix(filename)
    # Read MRC
    open = mrcfile.mmap if mmap else mrcfile.open
    with open(filename, mode="r", permissive=permissive) as mrc:
        array = cast(np.ndarray, mrc.data)
        if mrc.is_single_image():
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y], dtype=float
            )
        elif mrc.is_image_stack():
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y], dtype=float
            )
        elif mrc.is_volume():
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=float,
            )
        elif mrc.is_volume_stack():
            grid_spacing_per_dimension = np.asarray(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=float,
            )
        else:
            raise ValueError(
                "Mrcfile could not be identified as an image, image stack, volume, or "
                f"volume stack. Run mrcfile.validate(...) to make sure {filename} is a "
                "valid MRC file."
            )

        if not mmap:
            array = array.astype(np.float64)

        if loads_grid_spacing:
            # Only allow the same spacing in each direction
            assert all(
                grid_spacing_per_dimension != np.zeros(grid_spacing_per_dimension.shape)
            ), "Mrcfile.voxel_size must be set if reading the grid spacing. Found that "
            "Mrcfile.voxel_size = (0.0, 0.0, 0.0)"
            assert all(grid_spacing_per_dimension == grid_spacing_per_dimension[0]), (
                "Mrcfile.voxel_size must be same in all dimensions."
            )
            grid_spacing = grid_spacing_per_dimension[0]

            return cast(np.ndarray, array), float(grid_spacing)
        else:
            # ... otherwise, just return
            return cast(np.ndarray, array)


def write_image_to_mrc(
    image: Float[Array, "y_dim x_dim"] | Float[np.ndarray, "y_dim x_dim"],
    pixel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str | pathlib.Path,
    overwrite: bool = False,
    compression: str | None = None,
):
    """Write an image stack to an MRC file.

    **Arguments:**

    - `image`: The image stack, where the leading dimension indexes the image.
    - `pixel_size`: The pixel size of the images in `image_stack`.
    - `filename`: The output filename.
    - `overwrite`: If `True`, overwrite an existing file.
    - `compression`: See `mrcfile.new` for documentation.
    """
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    if suffix != ".mrc":
        raise OSError(
            f"The suffix for an image in MRC format must be .mrc. Instead, got {suffix}."
        )
    if image.ndim != 2:
        raise ValueError("image.ndim must be equal to 2.")
    # Convert image stack and pixel size to numpy arrays.
    image_as_numpy = np.asarray(image)
    pixel_size_as_numpy = np.asarray(pixel_size)
    # Create new file and write
    with mrcfile.new(filename, compression=compression, overwrite=overwrite) as mrc:
        mrc.set_data(image_as_numpy)
        mrc.voxel_size = pixel_size_as_numpy


def write_image_stack_to_mrc(
    image_stack: Float[Array, "M y_dim x_dim"] | Float[np.ndarray, "M y_dim x_dim"],
    pixel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str | pathlib.Path,
    overwrite: bool = False,
    compression: str | None = None,
):
    """Write an image stack to an MRC file.

    **Arguments:**

    - `image_stack`: The image stack, where the leading dimension indexes the image.
    - `pixel_size`: The pixel size of the images in `image_stack`.
    - `filename`: The output filename.
    - `overwrite`: If `True`, overwrite an existing file.
    - `compression`: See `mrcfile.new` for documentation.
    """
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    if suffix != ".mrcs":
        raise OSError(
            "The suffix for an image stack in MRC format must be .mrcs. "
            f"Instead, got {suffix}."
        )
    if image_stack.ndim != 3:
        raise ValueError("image_stack.ndim must be equal to 3.")
    # Convert image stack and pixel size to numpy arrays.
    image_stack_as_numpy = np.asarray(image_stack)
    pixel_size_as_numpy = np.asarray(pixel_size)
    # Create new file and write
    with mrcfile.new(filename, compression=compression, overwrite=overwrite) as mrc:
        mrc.set_data(image_stack_as_numpy)
        mrc.set_image_stack()
        mrc.voxel_size = pixel_size_as_numpy


def write_volume_to_mrc(
    voxel_grid: (
        Float[Array, "z_dim y_dim x_dim"] | Float[np.ndarray, "z_dim y_dim x_dim"]
    ),
    voxel_size: Float[Array, ""] | Float[np.ndarray, ""] | float,
    filename: str | pathlib.Path,
    overwrite: bool = False,
    compression: str | None = None,
):
    """Write a voxel grid to an MRC file.

    **Arguments:**

    - `voxel_grid`: The voxel grid as a JAX array.
    - `voxel_size`: The voxel size of the `voxel_grid`.
    - `filename`: The output filename.
    - `overwrite`: If `True`, overwrite an existing file.
    - `compression`: See `mrcfile.new` for documentation.
    """
    # Validate filename as MRC path and get suffix
    suffix = _validate_filename_and_return_suffix(filename)
    if suffix != ".mrc":
        raise OSError(
            f"The suffix for a volume in MRC format must be .mrc. Instead, got {suffix}."
        )
    if voxel_grid.ndim != 3:
        raise ValueError("voxel_grid.ndim must be equal to 3.")
    # Convert volume and voxel size to numpy arrays.
    voxel_grid_as_numpy = np.asarray(voxel_grid)
    voxel_size_as_numpy = np.asarray(voxel_size)
    # Create new file and write
    with mrcfile.new(filename, compression=compression, overwrite=overwrite) as mrc:
        mrc.set_data(voxel_grid_as_numpy)
        mrc.set_volume()
        mrc.voxel_size = voxel_size_as_numpy


def _validate_filename_and_return_suffix(filename: str | pathlib.Path):
    # Get suffixes
    suffixes = pathlib.Path(filename).suffixes
    # Make sure that leading suffix is valid MRC suffix
    if len(suffixes) == 0 or suffixes[0] not in [".mrc", ".mrcs"]:
        raise OSError(
            f"Filename should include .mrc or .mrcs suffix. Got filename {filename}."
        )
    return suffixes[0]
