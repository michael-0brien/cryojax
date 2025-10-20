import warnings
from collections.abc import Callable
from typing import Any
from typing_extensions import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, PyTree

from ....constants import (
    PengScatteringFactorParameters,
    b_factor_to_variance,
    variance_to_b_factor,
)
from ....coordinates import make_1d_coordinate_grid
from ....jax_util import NDArrayLike, error_if_not_positive
from ..._pose import AbstractPose
from .base_representations import AbstractAtomVolume


class GaussianMixtureVolume(AbstractAtomVolume, strict=True):
    r"""A representation of a volume as a mixture of
    gaussians, with multiple gaussians used per position.

    The convention of allowing multiple gaussians per position
    follows "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996). The
    $a$ and $b$ parameters in this work correspond to
    `amplitudes = 4 * np.pi * a` and `variances = b / 8\pi^2`.

    !!! info
        Use the following to load a `GaussianMixtureVolume`
        from these tabulated electron scattering factors.

        ```python
        from cryojax.constants import PengScatteringFactorParameters
        from cryojax.io import read_atoms_from_pdb
        from cryojax.simulator import GaussianMixtureVolume

        # Load positions of atoms and one-hot encoded atom names
        atom_positions, atom_types = read_atoms_from_pdb(...)
        parameters = PengScatteringFactorParameters(atom_types)
        potential = GaussianMixtureVolume.from_tabulated_parameters(
            atom_positions, parameters
        )
        ```
    """

    positions: Float[Array, "n_positions 3"]
    amplitudes: Float[Array, "n_positions n_gaussians"]
    variances: Float[Array, " n_positions n_gaussians"]

    def __init__(
        self,
        positions: Float[NDArrayLike, "n_positions 3"],
        amplitudes: (
            float
            | Float[NDArrayLike, ""]
            | Float[NDArrayLike, " n_positions"]
            | Float[NDArrayLike, "n_positions n_gaussians"]
        ),
        variances: (
            float
            | Float[NDArrayLike, ""]
            | Float[NDArrayLike, " n_positions"]
            | Float[NDArrayLike, "n_positions n_gaussians"]
        ),
    ):
        """**Arguments:**

        - `positions`:
            The coordinates of the gaussians in units of angstroms.
        - `amplitudes`:
            The amplitude for each gaussian.
            To simulate in physical units of a scattering potential,
            this should have units of angstroms.
        - `variances`:
            The variance for each gaussian. This has units of angstroms
            squared.
        """
        n_positions = positions.shape[0]
        if isinstance(amplitudes, NDArrayLike):
            if amplitudes.ndim == 2:
                n_gaussians = amplitudes.shape[-1]
            elif amplitudes.ndim == 1:
                n_gaussians = 1
                amplitudes = amplitudes[:, None]
            elif amplitudes.ndim == 0:
                n_gaussians = 1
                amplitudes = amplitudes[None, None]
            else:
                raise ValueError(
                    "Passed `amplitudes` to `GaussianMixtureVolume` "
                    f"with shape {amplitudes.shape}, but must be of "
                    "shape `()`, `(n_positions,)`, or "
                    "`(n_positions, n_gaussians)`."
                )
        else:
            n_gaussians = 1
        if isinstance(variances, NDArrayLike):
            if variances.ndim == 2:
                n_gaussians = variances.shape[-1]
            elif variances.ndim == 1:
                variances = variances[:, None]
            elif variances.ndim == 0:
                variances = variances[None, None]
            else:
                raise ValueError(
                    "Passed `variances` to `GaussianMixtureVolume` "
                    f"with shape {variances.shape}, but must be of "
                    "shape `()`, `(n_positions,)`, or "
                    "`(n_positions, n_gaussians)`."
                )

        self.positions = jnp.asarray(positions, dtype=float)
        self.amplitudes = jnp.broadcast_to(
            jnp.asarray(amplitudes, dtype=float), (n_positions, n_gaussians)
        )
        self.variances = jnp.broadcast_to(
            error_if_not_positive(jnp.asarray(variances, dtype=float)),
            (n_positions, n_gaussians),
        )

    def __check_init__(self):
        if not (
            self.positions.shape[0] == self.amplitudes.shape[0] == self.variances.shape[0]
        ):
            raise ValueError(
                "The number of positions in `GaussianMixtureVolume` was "
                f"{self.positions.shape[0]}, but `amplitudes` shape was "
                f"{self.amplitudes.shape} and `variances` shape was "
                f"{self.variances.shape}. The first dimension must be equal "
                "to the number of positions."
            )
        if not (self.amplitudes.shape == self.variances.shape):
            raise ValueError(
                "In `GaussianMixtureVolume`, `amplitudes` and "
                f"`variances` shape must be equal. Found shapes "
                f"{self.amplitudes.shape} and {self.variances.shape}, "
                "respectively."
            )

    @classmethod
    def from_tabulated_parameters(
        cls,
        atom_positions: Float[NDArrayLike, "n_atoms 3"],
        parameters: PengScatteringFactorParameters,
        extra_b_factors: (
            float | Float[NDArrayLike, ""] | Float[NDArrayLike, " n_atoms"] | None
        ) = None,
    ) -> Self:
        """Initialize a `GaussianMixtureVolume` from tabulated electron
        scattering factor parameters (Peng et al. 1996). This treats
        the scattering potential as a mixture of five gaussians
        per atom.

        **References:**

        - Peng, L-M. "Electron atomic scattering factors and scattering potentials of crystals."
            Micron 30.6 (1999): 625-648.
        - Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic
            scattering factors." Acta Crystallographica Section A: Foundations of Crystallography
            52.2 (1996): 257-276.

        **Arguments:**

        - `atom_positions`:
            The coordinates of the atoms in units of angstroms.
        - `parameters`:
            A pytree for the scattering factor parameters from
            Peng et al. (1996).
        - `extra_b_factors`:
            Additional per-atom B-factors that are added to
            the values in `scattering_parameters.b`.
        """  # noqa: E501
        amplitudes = jnp.asarray(parameters.a, dtype=float)
        b_factors = jnp.asarray(parameters.b, dtype=float)
        if extra_b_factors is not None:
            extra_b_factors = jnp.asarray(extra_b_factors, dtype=float)
            if extra_b_factors.ndim == 1:
                extra_b_factors = extra_b_factors[:, None]
            b_factors += extra_b_factors
        return cls(
            atom_positions, 4 * jnp.pi * amplitudes, b_factor_to_variance(b_factors)
        )

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new potential with rotated `positions`."""
        return eqx.tree_at(
            lambda d: d.positions,
            self,
            pose.rotate_coordinates(self.positions, inverse=inverse),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        offset_in_angstroms = pose.offset_in_angstroms
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        return eqx.tree_at(
            lambda d: d.positions, self, self.positions + offset_in_angstroms
        )

    def to_real_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        batch_options: dict[str, Any] = {},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        warnings.warn(
            "'GaussianMixtureVolume.to_real_voxel_grid' is deprecated "
            "and will be removed in cryoJAX 0.6.0. Instead, use "
            "`cryojax.simulator.GaussianMixtureRenderFn`.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return _gaussians_to_real_voxels(
            shape,
            jnp.asarray(voxel_size, dtype=float),
            self.positions,
            self.amplitudes,
            variance_to_b_factor(self.variances),
            **batch_options,
        )


@eqx.filter_jit
def _gaussians_to_real_voxels(
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians_per_position"],
    b_factors: Float[Array, "n_positions n_gaussians_per_position"],
    *,
    batch_size: int = 1,
    n_batches: int = 1,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    # Make coordinate systems for each of x, y, and z dimensions
    z_dim, y_dim, x_dim = shape
    grid_x, grid_y, grid_z = [
        make_1d_coordinate_grid(dim, voxel_size) for dim in [x_dim, y_dim, z_dim]
    ]
    # Get function to compute potential over a batch of positions
    render_fn = lambda xs: _gaussians_to_real_voxels_kernel(
        grid_x,
        grid_y,
        grid_z,
        voxel_size,
        xs[0],
        xs[1],
        xs[2],
        batch_size,
    )
    if n_batches > positions.shape[0]:
        raise ValueError(
            "The `n_batches` when building a voxel grid must "
            "be an integer less than or equal to the number of positions, "
            f"which is equal to {positions.shape[0]}. Got "
            f"`n_batches = {n_batches}`."
        )
    elif n_batches == 1:
        real_voxel_grid = render_fn((positions, amplitudes, b_factors))
    elif n_batches > 1:
        real_voxel_grid = jnp.sum(
            _batched_map_with_n_batches(
                render_fn,
                (positions, amplitudes, b_factors),
                n_batches=n_batches,
                is_batch_axis_contracted=True,
            ),
            axis=0,
        )
    else:
        raise ValueError(
            "The `n_batches` when building a voxel grid must be an "
            "integer greater than or equal to 1."
        )

    return real_voxel_grid


def _gaussians_to_real_voxels_kernel(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    grid_z: Float[Array, " dim_z"],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "n_positions_in_batch 3"],
    amplitudes: Float[Array, "n_positions_in_batch n_gaussians_per_position"],
    b_factors: Float[Array, "n_positions_in_batch n_gaussians_per_position"],
    batch_size: int,
) -> Float[Array, "dim_z dim_y dim_x"]:
    # Evaluate 1D gaussian integrals for each of x, y, and z dimensions
    (
        gaussian_integrals_times_prefactor_per_interval_per_position_x,
        gaussian_integrals_per_interval_per_position_y,
        gaussian_integrals_per_interval_per_position_z,
    ) = _evaluate_gaussian_integrals(
        grid_x, grid_y, grid_z, positions, amplitudes, b_factors, voxel_size
    )
    # Get function to compute voxel grid at a single z-plane
    render_at_z_plane = (
        lambda gaussian_integrals_per_position_z: _evaluate_multivariate_gaussian(
            gaussian_integrals_times_prefactor_per_interval_per_position_x,
            gaussian_integrals_per_interval_per_position_y,
            gaussian_integrals_per_position_z,
        )
    )
    # Map over z-planes
    if batch_size > grid_z.size:
        raise ValueError(
            "The `batch_size` when building a voxel grid must be an "
            "integer less than or equal to the z-dimension of the grid, "
            f"which is equal to {grid_z.size}."
        )
    elif batch_size == 1:
        # ... compute the volume iteratively
        real_voxel_grid = jax.lax.map(
            render_at_z_plane, gaussian_integrals_per_interval_per_position_z
        )
    elif batch_size > 1:
        # ... compute the volume by tuning how many z-planes to batch over
        render_at_z_planes = jax.vmap(render_at_z_plane, in_axes=0)
        real_voxel_grid = _batched_map_with_batch_size(
            render_at_z_planes,
            gaussian_integrals_per_interval_per_position_z,
            batch_size=batch_size,
            is_batch_axis_contracted=False,
        )
    else:
        raise ValueError(
            "The `batch_size` when building a voxel grid must be an "
            "integer greater than or equal to 1."
        )

    return real_voxel_grid


def _evaluate_gaussian_integrals(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    grid_z: Float[Array, " dim_z"],
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians_per_position"],
    b_factors: Float[Array, "n_positions n_gaussians_per_position"],
    voxel_size: Float[Array, ""],
) -> tuple[
    Float[Array, "dim_x n_positions n_gaussians_per_position"],
    Float[Array, "dim_y n_positions n_gaussians_per_position"],
    Float[Array, "dim_z n_positions n_gaussians_per_position"],
]:
    """Evaluate 1D averaged gaussians in x, y, and z dimensions
    for each position and each gaussian per position.
    """
    # Define function to compute integrals for each dimension
    scaling = 2 * jnp.pi / jnp.sqrt(b_factors)
    integration_kernel = lambda delta: (
        jsp.special.erf(scaling[None, :, :] * (delta + voxel_size)[:, :, None])
        - jsp.special.erf(scaling[None, :, :] * delta[:, :, None])
    )
    # Compute outer product of left edge of grid points minus positions
    left_edge_grid_x, left_edge_grid_y, left_edge_grid_z = (
        grid_x - voxel_size / 2,
        grid_y - voxel_size / 2,
        grid_z - voxel_size / 2,
    )
    delta_x, delta_y, delta_z = (
        left_edge_grid_x[:, None] - positions[:, 0],
        left_edge_grid_y[:, None] - positions[:, 1],
        left_edge_grid_z[:, None] - positions[:, 2],
    )
    # Compute gaussian integrals for each grid point, each position, and
    # each gaussian per position
    gauss_x, gauss_y, gauss_z = (
        integration_kernel(delta_x),
        integration_kernel(delta_y),
        integration_kernel(delta_z),
    )
    # Compute the prefactors for each position and each gaussian per position
    # for the potential
    prefactor = amplitudes / (2 * voxel_size) ** 3
    # Multiply the prefactor onto one of the gaussians for efficiency
    return prefactor * gauss_x, gauss_y, gauss_z


def _evaluate_multivariate_gaussian(
    gaussian_integrals_per_interval_per_position_x: Float[
        Array, "dim_x n_positions n_gaussians_per_position"
    ],
    gaussian_integrals_per_interval_per_position_y: Float[
        Array, "dim_y n_positions n_gaussians_per_position"
    ],
    gaussian_integrals_per_position_z: Float[
        Array, "n_positions n_gaussians_per_position"
    ],
) -> Float[Array, "dim_y dim_x"]:
    # Prepare matrices with dimensions of the number of positions and the number of grid
    # points. There are as many matrices as number of gaussians per position
    gauss_x = jnp.transpose(gaussian_integrals_per_interval_per_position_x, (2, 1, 0))
    gauss_yz = jnp.transpose(
        gaussian_integrals_per_interval_per_position_y
        * gaussian_integrals_per_position_z[None, :, :],
        (2, 0, 1),
    )
    # Compute matrix multiplication then sum over the number of gaussians per position
    return jnp.sum(jnp.matmul(gauss_yz, gauss_x), axis=0)


def _batched_map_with_n_batches(
    fun: Callable,
    xs: PyTree[Array],
    n_batches: int,
    is_batch_axis_contracted: bool = False,
):
    batch_dim = jax.tree.leaves(xs)[0].shape[0]
    batch_size = batch_dim // n_batches
    return _batched_map(
        fun, xs, batch_dim, n_batches, batch_size, is_batch_axis_contracted
    )


def _batched_map_with_batch_size(
    fun: Callable,
    xs: PyTree[Array],
    batch_size: int,
    is_batch_axis_contracted: bool = False,
):
    batch_dim = jax.tree.leaves(xs)[0].shape[0]
    n_batches = batch_dim // batch_size
    return _batched_map(
        fun, xs, batch_dim, n_batches, batch_size, is_batch_axis_contracted
    )


def _batched_map(
    fun: Callable,
    xs: PyTree[Array],
    batch_dim: int,
    n_batches: int,
    batch_size: int,
    is_batch_axis_contracted: bool = False,
):
    """Like `jax.lax.map`, but map over leading axis of `xs` in
    chunks of size `batch_size`. Assumes `fun` can be evaluated in
    parallel over this leading axis.
    """
    # ... reshape into an iterative dimension and a batching dimension
    xs_per_batch = jax.tree.map(
        lambda x: x[: batch_dim - batch_dim % batch_size, ...].reshape(
            (n_batches, batch_size, *x.shape[1:])
        ),
        xs,
    )
    # .. compute the result and reshape back into one leading dimension
    result_per_batch = jax.lax.map(fun, xs_per_batch)
    if is_batch_axis_contracted:
        result = result_per_batch
    else:
        result = result_per_batch.reshape(
            (n_batches * batch_size, *result_per_batch.shape[2:])
        )
    # ... if the batch dimension is not divisible by the batch size, need
    # to take care of the remainder
    if batch_dim % batch_size != 0:
        remainder = fun(
            jax.tree.map(lambda x: x[batch_dim - batch_dim % batch_size :, ...], xs)
        )
        if is_batch_axis_contracted:
            remainder = remainder[None, ...]
        result = jnp.concatenate([result, remainder], axis=0)
    return result
