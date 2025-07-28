from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree

from ...coordinates import make_1d_coordinate_grid


def gaussians_to_real_voxels(
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    b_factors: Float[Array, "n_atoms n_gaussians_per_atom"],
    *,
    batch_size: int = 1,
    n_batches: int = 1,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    # Make coordinate systems for each of x, y, and z dimensions
    z_dim, y_dim, x_dim = shape
    grid_x, grid_y, grid_z = [
        make_1d_coordinate_grid(dim, voxel_size) for dim in [x_dim, y_dim, z_dim]
    ]
    # Get function to compute potential over a batch of atoms
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
            "be an integer less than or equal to the number of atoms, "
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
    positions: Float[Array, "n_atoms_in_batch 3"],
    amplitudes: Float[Array, "n_atoms_in_batch n_gaussians_per_atom"],
    b_factors: Float[Array, "n_atoms_in_batch n_gaussians_per_atom"],
    batch_size: int,
) -> Float[Array, "dim_z dim_y dim_x"]:
    # Evaluate 1D gaussian integrals for each of x, y, and z dimensions
    (
        gaussian_integrals_times_prefactor_per_interval_per_atom_x,
        gaussian_integrals_per_interval_per_atom_y,
        gaussian_integrals_per_interval_per_atom_z,
    ) = _evaluate_gaussian_integrals(
        grid_x, grid_y, grid_z, positions, amplitudes, b_factors, voxel_size
    )
    # Get function to compute voxel grid at a single z-plane
    render_at_z_plane = (
        lambda gaussian_integrals_per_atom_z: _evaluate_multivariate_gaussian(
            gaussian_integrals_times_prefactor_per_interval_per_atom_x,
            gaussian_integrals_per_interval_per_atom_y,
            gaussian_integrals_per_atom_z,
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
            render_at_z_plane, gaussian_integrals_per_interval_per_atom_z
        )
    elif batch_size > 1:
        # ... compute the volume by tuning how many z-planes to batch over
        render_at_z_planes = jax.vmap(render_at_z_plane, in_axes=0)
        real_voxel_grid = _batched_map_with_batch_size(
            render_at_z_planes,
            gaussian_integrals_per_interval_per_atom_z,
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
    positions: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    b_factors: Float[Array, "n_atoms n_gaussians_per_atom"],
    voxel_size: Float[Array, ""],
) -> tuple[
    Float[Array, "dim_x n_atoms n_gaussians_per_atom"],
    Float[Array, "dim_y n_atoms n_gaussians_per_atom"],
    Float[Array, "dim_z n_atoms n_gaussians_per_atom"],
]:
    """Evaluate 1D averaged gaussians in x, y, and z dimensions
    for each atom and each gaussian per atom.
    """
    # Define function to compute integrals for each dimension
    scaling = 2 * jnp.pi / jnp.sqrt(b_factors)
    integration_kernel = lambda delta: (
        jsp.special.erf(scaling[None, :, :] * (delta + voxel_size)[:, :, None])
        - jsp.special.erf(scaling[None, :, :] * delta[:, :, None])
    )
    # Compute outer product of left edge of grid points minus atomic positions
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
    # Compute gaussian integrals for each grid point, each atom, and
    # each gaussian per atom
    gauss_x, gauss_y, gauss_z = (
        integration_kernel(delta_x),
        integration_kernel(delta_y),
        integration_kernel(delta_z),
    )
    # Compute the prefactors for each atom and each gaussian per atom
    # for the potential
    prefactor = (4 * jnp.pi * amplitudes) / (2 * voxel_size) ** 3
    # Multiply the prefactor onto one of the gaussians for efficiency
    return prefactor * gauss_x, gauss_y, gauss_z


def _evaluate_multivariate_gaussian(
    gaussian_integrals_per_interval_per_atom_x: Float[
        Array, "dim_x n_atoms n_gaussians_per_atom"
    ],
    gaussian_integrals_per_interval_per_atom_y: Float[
        Array, "dim_y n_atoms n_gaussians_per_atom"
    ],
    gaussian_integrals_per_atom_z: Float[Array, "n_atoms n_gaussians_per_atom"],
) -> Float[Array, "dim_y dim_x"]:
    # Prepare matrices with dimensions of the number of atoms and the number of grid
    # points. There are as many matrices as number of gaussians per atom
    gauss_x = jnp.transpose(gaussian_integrals_per_interval_per_atom_x, (2, 1, 0))
    gauss_yz = jnp.transpose(
        gaussian_integrals_per_interval_per_atom_y
        * gaussian_integrals_per_atom_z[None, :, :],
        (2, 0, 1),
    )
    # Compute matrix multiplication then sum over the number of gaussians per atom
    return jnp.sum(jnp.matmul(gauss_yz, gauss_x), axis=0)


def _batched_map_with_n_batches(
    fun: Callable,
    xs: PyTree[Array],
    n_batches: int,
    is_batch_axis_contracted: bool = False,
):
    batch_dim = jtu.tree_leaves(xs)[0].shape[0]
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
    batch_dim = jtu.tree_leaves(xs)[0].shape[0]
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
    xs_per_batch = jtu.tree_map(
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
            jtu.tree_map(lambda x: x[batch_dim - batch_dim % batch_size :, ...], xs)
        )
        if is_batch_axis_contracted:
            remainder = remainder[None, ...]
        result = jnp.concatenate([result, remainder], axis=0)
    return result
