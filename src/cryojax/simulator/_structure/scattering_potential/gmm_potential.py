from typing import Any
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Float

from ....internal import NDArrayLike, error_if_negative
from ..common_functions import gaussians_to_real_voxels
from ..representations import (
    AbstractIndependentAtomStructure,
)
from ..structure_conversion import (
    AbstractDiscretizesToRealVoxels as AbstractDiscretizesToRealVoxels,
)
from .base_potential import AbstractScatteringPotential


class GaussianIndependentAtomPotential(
    AbstractScatteringPotential,
    AbstractIndependentAtomStructure,
    AbstractDiscretizesToRealVoxels,
    strict=True,
):
    r"""An atomistic representation of scattering potential as a mixture of
    gaussians.

    The naming and numerical convention of parameters `amplitudes` and
    `b_factors` follows "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996), where the
    `amplitudes` are like the $a_i$ and the `b_factors` are like the $b_i$.

    !!! info
        In order to load a `GaussianMixtureAtomicPotential` from tabulated
        scattering factors, use the `cryojax.constants` submodule.

        ```python
        from cryojax.io import read_atoms_from_pdb
        from cryojax.simulator import (
            GaussianMixtureAtomicPotential, PengScatteringFactorParameters
        )

        # Load positions of atoms and one-hot encoded atom names
        atom_positions, atomic_numbers = read_atoms_from_pdb(...)
        scattering_factor_parameters = PengScatteringFactorParameters(atomic_numbers)
        potential = GaussianMixtureAtomicPotential(
            atom_positions,
            amplitudes=scattering_factor_parameters.a,
            b_factors=scattering_factor_parameters.b,
        )
        ```
    """

    atom_positions: Float[Array, "n_atoms 3"]
    amplitudes: Float[Array, "n_atoms n_gaussians"]
    b_factors: Float[Array, "n_atoms n_gaussians"]

    def __init__(
        self,
        atom_positions: Float[NDArrayLike, "n_atoms 3"],
        amplitudes: Float[NDArrayLike, "n_atoms n_gaussians"],
        b_factors: Float[NDArrayLike, "n_atoms n_gaussians"],
    ):
        """**Arguments:**

        - `atom_positions`: The coordinates of the atoms in units of angstroms.
        - `amplitudes`:
            The strength for each atom and gaussian per atom.
            This has units of angstroms.
        - `b_factors`:
            The B-factors for each atom and gaussian per atom.
            This has units of angstroms squared.
        """
        self.atom_positions = jnp.asarray(atom_positions, dtype=float)
        self.amplitudes = jnp.asarray(amplitudes, dtype=float)
        self.b_factors = error_if_negative(jnp.asarray(b_factors, dtype=float))

    @override
    def to_real_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        options: dict[str, Any] = {},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        """Return a voxel grid of the potential in real space.

        See [`PengIndependentAtomPotential.to_real_voxel_grid`](structure.md#cryojax.simulator.PengIndependentAtomPotential.to_real_voxel_grid)
        for the numerical conventions used when computing the sum of gaussians.

        **Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `options`:
            Advanced options for rendering. This is a dictionary
            with the following keys:
            - "batch_size":
                The number of z-planes to evaluate in parallel with
                `jax.vmap`. By default, `1`.
            - "n_batches":
                The number of iterations used to evaluate the volume,
                where the iteration is taken over groups of atoms.
                This is useful if `batch_size_for_z_planes = 1`
                and GPU memory is exhausted. By default, `1`.

        **Returns:**

        The rescaled potential $U_{\\ell}$ as a voxel grid of shape `shape`
        and voxel size `voxel_size`.
        """  # noqa: E501
        return gaussians_to_real_voxels(
            shape,
            jnp.asarray(voxel_size, dtype=float),
            self.atom_positions,
            self.amplitudes,
            self.b_factors,
            **options,
        )
