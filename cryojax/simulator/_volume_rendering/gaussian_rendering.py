from collections.abc import Callable
from typing import Any, TypedDict
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, PyTree

from ...constants import variance_to_b_factor
from ...coordinates import make_1d_coordinate_grid
from ...jax_util import NDArrayLike, error_if_not_positive
from ...ndimage import fftn, rfftn
from .._volume import GaussianMixtureVolume
from .base_rendering import AbstractVolumeRenderFn


class BatchOptions(TypedDict):
    batch_size: int
    n_batches: int


class GaussianMixtureRenderFn(AbstractVolumeRenderFn[GaussianMixtureVolume], strict=True):
    """Render a voxel grid from the `GaussianMixtureVolume`.

    If `GaussianMixtureVolume` is instantiated from electron scattering
    factors via `from_peng_parameters`, this renders an electrostatic
    potential as tabulated in Peng et al. 1996. The elastic electron
    scattering factors defined in this work are

    $$f^{(e)}(\\mathbf{q}) = \\sum\\limits_{i = 1}^5 a_i \\exp(- b_i |\\mathbf{q}|^2),$$

    where $4 \\pi a_i$ is stored as `GaussianMixtureVolume.amplitudes`,
    $b_i / 8 \\pi^2$ are the `GaussianMixtureVolume.variances`, and
    $\\mathbf{q}$ is the scattering vector.

    Under usual scattering approximations (i.e. the first-born approximation),
    the rescaled electrostatic potential energy $U(\\mathbf{r})$ for a given atom type is
    $4 \\pi \\mathcal{F}^{-1}[f^{(e)}(\\boldsymbol{\\xi} / 2)](\\mathbf{r})$, which is computed
    analytically as

    $$U(\\mathbf{r}) = 4 \\pi \\sum\\limits_{i = 1}^5 \\frac{a_i}{(2\\pi (b_i / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{r} - \\mathbf{r}'|^2}{2 (b_i / 8 \\pi^2)}),$$

    where $\\mathbf{r}'$ is the position of the atom. Including an additional B-factor (denoted by
    $B$) gives the expression for the potential
    $U(\\mathbf{r})$ of a single atom type and its fourier transform pair $\\tilde{U}(\\boldsymbol{\\xi}) \\equiv \\mathcal{F}[U](\\boldsymbol{\\xi})$,

    $$U(\\mathbf{r}) = 4 \\pi \\sum\\limits_{i = 1}^5 \\frac{a_i}{(2\\pi ((b_i + B) / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{r} - \\mathbf{r}'|^2}{2 ((b_i + B) / 8 \\pi^2)}),$$

    $$\\tilde{U}(\\boldsymbol{\\xi}) = 4 \\pi \\sum\\limits_{i = 1}^5 a_i \\exp(- (b_i + B) |\\boldsymbol{\\xi}|^2 / 4) \\exp(2 \\pi i \\boldsymbol{\\xi}\\cdot\\mathbf{r}'),$$

    where $\\mathbf{q} = \\boldsymbol{\\xi} / 2$ gives the relationship between the wave vector and the
    scattering vector.

    In practice, for a discretization on a grid with voxel size $\\Delta r$ and grid point $\\mathbf{r}_{\\ell}$,
    the potential is evaluated as the average value inside the voxel

    $$U_{\\ell} = 4 \\pi \\frac{1}{\\Delta r^3} \\sum\\limits_{i = 1}^5 a_i \\prod\\limits_{j = 1}^3 \\int_{r^{\\ell}_j-\\Delta r/2}^{r^{\\ell}_j+\\Delta r/2} dr_j \\ \\frac{1}{{\\sqrt{2\\pi ((b_i + B) / 8 \\pi^2)}}} \\exp(- \\frac{(r_j - r'_j)^2}{2 ((b_i + B) / 8 \\pi^2)}),$$

    where $j$ indexes the components of the spatial coordinate vector $\\mathbf{r}$. The above expression is evaluated using the error function as

    $$U_{\\ell} = 4 \\pi \\frac{1}{(2 \\Delta r)^3} \\sum\\limits_{i = 1}^5 a_i \\prod\\limits_{j = 1}^3 \\textrm{erf}(\\frac{r_j^{\\ell} - r'_j + \\Delta r / 2}{\\sqrt{2 ((b_i + B) / 8\\pi^2)}}) - \\textrm{erf}(\\frac{r_j^{\\ell} - r'_j - \\Delta r / 2}{\\sqrt{2 ((b_i + B) / 8\\pi^2)}}).$$
    """  # noqa: E501

    shape: tuple[int, int, int]
    voxel_size: Float[Array, ""]

    batch_options: BatchOptions

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        batch_options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `batch_options`:
            Advanced options for controlling batching. This is a dictionary
            with the following keys:
            - "batch_size":
                The number of z-planes to evaluate in parallel with
                `jax.vmap`. By default, `1`.
            - "n_batches":
                The number of iterations used to evaluate the volume,
                where the iteration is taken over groups of atoms.
                This is useful if `batch_size = 1`
                and GPU memory is exhausted. By default, `1`.
        """
        self.shape = shape
        self.voxel_size = error_if_not_positive(jnp.asarray(voxel_size, dtype=float))
        self.batch_options = _dict_to_batch_options(batch_options)

    @override
    def __call__(
        self,
        volume_representation: GaussianMixtureVolume,
        *,
        outputs_real_space: bool = True,
        outputs_rfft: bool = False,
        fftshifted: bool = False,
    ) -> Float[Array, "{self.shape[0]} {self.shape[1]} {self.shape[2]}"]:
        """**Arguments:**

        - `volume_representation`:
            The `GaussianMixtureVolume`.
        - `outputs_real_space`:
            If `True`, return a voxel grid in real-space.
        - `outputs_rfft`:
            If `True`, return a fourier-space voxel grid transformed with
            `cryojax.ndimage.rfftn`. Otherwise, use `fftn`. Does nothing
            if `outputs_real_space = True`.
        - `fftshifted`:
            If `True`, return a fourier-space voxel grid with the zero
            frequency component in the center of the grid via
            `jax.numpy.fft.fftshift`. Otherwise, the zero frequency
            component is in the corner. Does nothing if
            `outputs_real_space = True`.
        """
        real_voxel_grid = _gaussians_to_real_voxels(
            self.shape,
            self.voxel_size,
            volume_representation.positions,
            volume_representation.amplitudes,
            variance_to_b_factor(volume_representation.variances),
            **self.batch_options,
        )
        if outputs_real_space:
            return real_voxel_grid
        else:
            if outputs_rfft:
                return (
                    jnp.fft.fftshift(rfftn(real_voxel_grid), axes=(0, 1))
                    if fftshifted
                    else rfftn(real_voxel_grid)
                )
            else:
                return (
                    jnp.fft.fftshift(fftn(real_voxel_grid))
                    if fftshifted
                    else fftn(real_voxel_grid)
                )


def _dict_to_batch_options(d):
    batch_size = d["batch_size"] if "batch_size" in d else 1
    n_batches = d["n_batches"] if "batch_size" in d else 1
    return BatchOptions(batch_size=batch_size, n_batches=n_batches)


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
