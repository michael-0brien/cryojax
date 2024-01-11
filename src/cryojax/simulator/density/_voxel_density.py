"""
Voxel-based electron density representations.
"""

__all__ = ["Voxels", "VoxelType", "VoxelCloud", "VoxelGrid"]

import pathlib
from abc import abstractmethod
from typing import Any, Tuple, Type, ClassVar, TypeVar, Optional, overload
from typing_extensions import Self
from jaxtyping import Complex, Float, Array
from equinox import AbstractVar
from functools import cached_property

import equinox as eqx
import jax
import jax.numpy as jnp

from ._electron_density import ElectronDensity
from ..pose import Pose
from ...io import (
    load_mrc,
    read_atomic_model_from_pdb,
    read_atomic_model_from_cif,
    get_scattering_info_from_gemmi_model,
)
from ...core import field
from cryojax.utils import (
    make_frequencies,
    make_coordinates,
    pad,
    fftn,
)
from cryojax.typing import (
    RealCloud,
    RealVolume,
    VolumeCoords,
    CloudCoords3D,
    Real_,
)

_RealCubicVolume = Float[Array, "N N N"]
_ComplexCubicVolume = Complex[Array, "N N N"]
_VolumeSliceCoords = Float[Array, "N N//2+1 1 3"]

VoxelType = TypeVar("VoxelType", bound="Voxels")
"""Type hint for a voxel-based electron density."""


class Voxels(ElectronDensity):
    """
    Voxel-based electron density representation.

    Attributes
    ----------
    weights :
        The electron density.
    voxel_size
        The voxel size of the electron density.
    """

    weights: AbstractVar[Array]
    voxel_size: Real_ = field(stack=False)

    @overload
    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[VoxelType],
        density_grid: RealVolume,
        voxel_size: float,
        coordinate_grid: None,
        **kwargs: Any,
    ) -> VoxelType:
        ...

    @overload
    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[VoxelType],
        density_grid: RealVolume,
        voxel_size: float,
        coordinate_grid: VolumeCoords,
        **kwargs: Any,
    ) -> VoxelType:
        ...

    @classmethod
    @abstractmethod
    def from_density_grid(
        cls: Type[VoxelType],
        density_grid: RealVolume,
        voxel_size: float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
        **kwargs: Any,
    ) -> VoxelType:
        """
        Load a Voxels object from real-valued 3D electron
        density map.
        """
        raise NotImplementedError

    @classmethod
    def from_gemmi(
        cls: Type[VoxelType],
        model,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        **kwargs: Any,
    ) -> VoxelType:
        """
        Loads a PDB file as a Voxels subclass.  Uses the Gemmi library.
        Heavily based on a code from Frederic Poitevin, located at

        https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        coords, a_vals, b_vals = get_scattering_info_from_gemmi_model(model)

        coordinate_grid_in_angstroms = make_coordinates(
            n_voxels_per_side, voxel_size
        )
        density = _build_real_space_voxels_from_atoms(
            coords, a_vals, b_vals, coordinate_grid_in_angstroms
        )

        return cls.from_density_grid(
            density,
            voxel_size,
            coordinate_grid_in_angstroms / voxel_size,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls: Type[VoxelType],
        filename: str,
        *args: Any,
        **kwargs: Any,
    ) -> VoxelType:
        """Load a voxel-based electron density."""
        path = pathlib.Path(filename)
        if path.suffix == ".mrc":
            return cls.from_mrc(filename, *args, **kwargs)
        elif path.suffix == ".pdb":
            return cls.from_pdb(filename, *args, **kwargs)
        elif path.suffix == ".cif":
            return cls.from_cif(filename, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"File format {path.suffix} not supported."
            )

    @classmethod
    def from_mrc(
        cls: Type[VoxelType],
        filename: str,
        **kwargs: Any,
    ) -> VoxelType:
        """Load Voxels from MRC file format."""
        density_grid, voxel_size = load_mrc(filename)
        return cls.from_density_grid(density_grid, voxel_size, **kwargs)

    @classmethod
    def from_pdb(
        cls: Type[VoxelType],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        **kwargs: Any,
    ) -> VoxelType:
        """Load Voxels from PDB file format."""
        model = read_atomic_model_from_pdb(filename)
        return cls.from_gemmi(model, n_voxels_per_side, voxel_size, **kwargs)

    @classmethod
    def from_cif(
        cls: Type[VoxelType],
        filename: str,
        n_voxels_per_side: Tuple[int, int, int],
        voxel_size: float = 1.0,
        **kwargs: Any,
    ) -> VoxelType:
        """Load Voxels from CIF file format."""
        model = read_atomic_model_from_cif(filename)
        return cls.from_gemmi(model, n_voxels_per_side, voxel_size, **kwargs)


class VoxelGrid(Voxels):
    """
    Abstraction of a 3D electron density voxel grid.

    The voxel grid should be given in fourier space.

    Attributes
    ----------
    weights :
        3D electron density grid in fourier space.
    frequency_slice :
        Central slice of cartesian coordinate system
        in fourier space.
    """

    weights: _ComplexCubicVolume = field()
    frequency_slice: _VolumeSliceCoords = field(stack=False)

    is_real: ClassVar[bool] = False

    @cached_property
    def frequency_slice_in_angstroms(self) -> _VolumeSliceCoords:
        return self.frequency_slice / self.voxel_size

    def rotate_to_pose(self, pose: Pose) -> Self:
        """
        Compute rotations of a central slice in fourier space
        by an imaging pose.

        This rotation is the inverse rotation as in real space.
        """
        return eqx.tree_at(
            lambda d: d.frequency_slice,
            self,
            pose.rotate(self.frequency_slice, is_real=self.is_real),
        )

    @classmethod
    def from_density_grid(
        cls: Type["VoxelGrid"],
        density_grid: RealVolume,
        voxel_size: float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
        pad_scale: float = 1.0,
        **kwargs: Any,
    ) -> "VoxelGrid":
        # Change how template sits in box to match cisTEM
        density_grid = jnp.transpose(density_grid, axes=[2, 1, 0])
        # Pad template
        padded_shape = tuple([int(s * pad_scale) for s in density_grid.shape])
        padded_density_grid = pad(density_grid, padded_shape)
        # Load density and coordinates. For now, do not store the
        # fourier density only on the half space. Fourier slice extraction
        # does not currently work if rfftn is used
        fourier_density_grid = fftn(padded_density_grid)
        # ... create in-plane frequency slice on the half space
        frequency_slice = make_frequencies(
            padded_density_grid.shape[:-1], half_space=True
        )
        # ... zero pad to make the slice 3-dimensional
        frequency_slice = jnp.expand_dims(
            jnp.pad(
                frequency_slice,
                ((0, 0), (0, 0), (0, 1)),
                mode="constant",
                constant_values=0.0,
            ),
            axis=2,
        )

        return cls(
            weights=fourier_density_grid,
            frequency_slice=frequency_slice,
            voxel_size=voxel_size,
        )


class VoxelCloud(Voxels):
    """
    Abstraction of a 3D electron density voxel point cloud.

    The point cloud is given in real space.

    Attributes
    ----------
    weights :
        Flattened 3D electron density voxel grid into a
        point cloud.
    coordinate_list :
        List of coordinates for the point cloud.
    """

    weights: RealCloud = field()
    coordinate_list: CloudCoords3D = field(stack=False)

    is_real: ClassVar[bool] = True

    @cached_property
    def coordinate_list_in_angstroms(self) -> CloudCoords3D:
        return self.voxel_size * self.coordinate_list

    def rotate_to_pose(self, pose: Pose) -> Self:
        """
        Compute rotations of a point cloud by an imaging pose.

        This transformation will return a new density cloud
        with rotated coordinates.
        """
        return eqx.tree_at(
            lambda d: d.coordinate_list,
            self,
            pose.rotate(self.coordinate_list, is_real=self.is_real),
        )

    @classmethod
    def from_density_grid(
        cls: Type["VoxelCloud"],
        density_grid: RealVolume,
        voxel_size: float = 1.0,
        coordinate_grid: Optional[VolumeCoords] = None,
        mask_zeros: bool = True,
        **kwargs: Any,
    ) -> "VoxelCloud":
        # Change how template sits in the box.
        # Ideally we would change this in the same way for all
        # I/O methods. However, the algorithms used all
        # have their own xyz conventions. The choice here is to
        # make jax-finufft output match cisTEM.
        density_grid = jnp.transpose(density_grid, axes=[1, 2, 0])
        # Make coordinates if not given
        if coordinate_grid is None:
            coordinate_grid = make_coordinates(density_grid.shape)
        # Load flattened density and coordinates
        if not mask_zeros:
            flat_density = density_grid.ravel()
            coordinate_list = coordinate_grid.ravel()
        else:
            # ... mask zeros if desired to store smaller arrays. This
            # option is not jittable.
            nonzero = jnp.where(~jnp.isclose(density_grid, 0.0, **kwargs))
            flat_density = density_grid[nonzero]
            coordinate_list = coordinate_grid[nonzero]

        return cls(
            weights=flat_density,
            coordinate_list=coordinate_list,
            voxel_size=voxel_size,
        )


def _eval_3d_real_space_gaussian(
    coordinate_system: Float[Array, "N1 N2 N3 3"],
    atom_position: Float[Array, "3"],
    a: float,
    b: float,
) -> Float[Array, "N1 N2 N3"]:
    """
    Evaluate a gaussian on a 3D grid.
    The naming convention for parameters follows ``Robust
    Parameterization of Elastic and Absorptive Electron Atomic Scattering
    Factors'' by Peng et al.

    Parameters
    ----------
    coordinate_system : `Array`, shape `(N1, N2, N3, 3)`
        The coordinate_system of the grid.
    pos : `Array`, shape `(3,)`
        The center of the gaussian.
    a : `float`
        A scale factor.
    b : `float`
        The scale of the gaussian.

    Returns
    -------
    density : `Array`, shape `(N1, N2, N3)`
        The density of the gaussian on the grid.
    """
    b_inverse = 4.0 * jnp.pi / b
    sq_distances = jnp.sum(
        b_inverse * (coordinate_system - atom_position) ** 2, axis=-1
    )
    density = jnp.exp(-jnp.pi * sq_distances) * a * b_inverse ** (3.0 / 2.0)
    return density


def _eval_3d_atom_potential(
    coordinate_system: Float[Array, "N1 N2 N3 3"],
    atom_position: Float[Array, "3"],
    atomic_as: Float[Array, "5"],
    atomic_bs: Float[Array, "5"],
) -> Float[Array, "N1 N2 N3"]:
    """
    Evaluates the electron potential of a single atom on a 3D grid.

    Parameters
    ----------
    coordinate_system : `Array`, shape `(N1, N2, N3, 3)`
        The coordinate_system of the grid.
    atom_position : `Array`, shape `(3,)`
        The location of the atom.
    atomic_as : `Array`, shape `(5,)`
        The intensity values for each gaussian in the atom.
    atomic_bs : `Array`, shape `(5,)`
        The inverse scale factors for each gaussian in the atom.

    Returns
    -------
    potential : `Array`, shape `(N1, N2, N3)`
        The potential of the atom evaluate on the grid.
    """
    eval_fxn = jax.vmap(
        _eval_3d_real_space_gaussian, in_axes=(None, None, 0, 0)
    )
    return jnp.sum(
        eval_fxn(coordinate_system, atom_position, atomic_as, atomic_bs),
        axis=0,
    )


@jax.jit
def _build_real_space_voxels_from_atoms(
    atom_positions: Float[Array, "N 3"],
    ff_a: Float[Array, "N 5"],
    ff_b: Float[Array, "N 5"],
    coordinate_system: Float[Array, "N1 N2 N3 3"],
) -> Tuple[_RealCubicVolume, _VolumeSliceCoords]:
    """
    Build a voxel representation of an atomic model.

    Parameters
    ----------
    atom_coords : `Array`, shape `(N, 3)`
        The coordinates of the atoms.
    ff_a : `Array`, shape `(N, 5)` or `(N, 5, 3)`
        Intensity values for each Gaussian in the atom
    ff_b : `Array`, shape `(N, 5)` or `(N, 5, 3)`
        The inverse scale factors for each Gaussian in the atom
    coordinate_system : `Array`, shape `(N1, N2, N3, 3)`
        The coordinates of each voxel in the grid.

    Returns
    -------
    density :  `Array`, shape `(N1, N2, N3)`
        The voxel representation of the atomic model.
    z_plane_coordinates : `Array`, shape `(N1, N2, 3)`
        The coordinates of each voxel in the z=0 plane.
    """
    density = jnp.zeros(coordinate_system.shape[:-1])

    def add_gaussian_to_density(i, density):
        density += _eval_3d_atom_potential(
            coordinate_system, atom_positions[i], ff_a[i], ff_b[i]
        )
        return density

    density = jax.lax.fori_loop(
        0, atom_positions.shape[0], add_gaussian_to_density, density
    )

    return density
