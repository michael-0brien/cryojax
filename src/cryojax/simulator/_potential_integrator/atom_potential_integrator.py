import math
from typing import ClassVar, Optional
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Complex, Float

from ...constants import convert_variance_to_b_factor
from ...coordinates import make_1d_coordinate_grid
from ...ndimage import (
    downsample_to_shape_with_fourier_cropping,
    resize_with_crop_or_pad,
    rfftn,
)
from .._config import AbstractConfig
from .._potential_representation import (
    GaussianMixtureAtomicPotential,
    PengAtomicPotential,
)
from .base_potential_integrator import AbstractPotentialIntegrator


class GaussianMixtureProjection(
    AbstractPotentialIntegrator[GaussianMixtureAtomicPotential | PengAtomicPotential],
    strict=True,
):
    upsampling_factor: Optional[int]
    shape: Optional[tuple[int, int]]
    use_error_functions: bool
    n_batches: int

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        upsampling_factor: Optional[int] = None,
        shape: Optional[tuple[int, int]] = None,
        use_error_functions: bool = False,
        n_batches: int = 1,
    ):
        """**Arguments:**

        - `upsampling_factor`:
            The factor by which to upsample the computation of the images.
            If `upsampling_factor` is greater than 1, the images will be computed
            at a higher resolution and then downsampled to the original resolution.
            This can be useful for reducing aliasing artifacts in the images.
        - `shape`:
            The shape of the plane on which projections are computed before padding or
            cropping to the `AbstractConfig.padded_shape`. This argument is particularly
            useful if the `AbstractConfig.padded_shape` is much larger than the protein.
        - `use_error_functions`:
            If `True`, use error functions to evaluate the projected potential at
            a pixel to be the average value within the pixel using gaussian
            integrals. If `False`, the potential at a pixel will simply be evaluated
            as a gaussian.
        - `n_batches`:
            The number of batches over groups of atoms
            used to evaluate the projection.
            This is useful if GPU memory is exhausted. By default,
            `1`, which computes a projection for all atoms at once.
        """  # noqa: E501
        self.upsampling_factor = upsampling_factor
        self.shape = shape
        self.use_error_functions = use_error_functions
        self.n_batches = n_batches

    def __check_init__(self):
        if self.upsampling_factor is not None and self.upsampling_factor < 1:
            raise AttributeError(
                "`GaussianMixtureProjection.upsampling_factor` must "
                f"be greater than `1`. Got a value of {self.upsampling_factor}."
            )

    @override
    def integrate(
        self,
        potential: GaussianMixtureAtomicPotential | PengAtomicPotential,
        config: AbstractConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{config.padded_y_dim} {config.padded_x_dim//2+1}",
        ]
        | Float[Array, "{config.padded_y_dim} {config.padded_x_dim}"]
    ):
        """Compute a projection from the atomic potential and transform it to Fourier
        space.

        **Arguments:**

        - `potential`: The atomic potential to project.
        - `config`: The configuration of the imaging instrument.

        **Returns:**

        The integrated potential in real or fourier space at the `AbstractConfig.padded_shape`.
        """  # noqa: E501
        # Grab the image configuration
        shape = config.padded_shape if self.shape is None else self.shape
        pixel_size = config.pixel_size
        if self.upsampling_factor is not None:
            u = self.upsampling_factor
            upsampled_pixel_size, upsampled_shape = (
                pixel_size / u,
                (
                    shape[0] * u,
                    shape[1] * u,
                ),
            )
        else:
            upsampled_pixel_size, upsampled_shape = pixel_size, shape
        # Grab the gaussian amplitudes and widths
        if isinstance(potential, PengAtomicPotential):
            gaussian_amplitudes = potential.scattering_factor_a
            gaussian_widths = potential.scattering_factor_b
            if potential.b_factors is not None:
                gaussian_widths += potential.b_factors[:, None]
        elif isinstance(potential, GaussianMixtureAtomicPotential):
            gaussian_amplitudes = potential.gaussian_amplitudes
            gaussian_widths = jnp.asarray(
                convert_variance_to_b_factor(potential.gaussian_variances)
            )
        else:
            raise ValueError(
                "Supported types for `potential` are `PengAtomicPotential` and "
                "`GaussianMixtureAtomicPotential`."
            )
        # Compute the projection
        in_plane_potential = _compute_projected_potential_from_atoms(
            upsampled_shape,
            upsampled_pixel_size,
            potential.atom_positions,
            gaussian_amplitudes,
            gaussian_widths,
            self.use_error_functions,
            self.n_batches,
        )
        if self.upsampling_factor is not None:
            # Downsample back to the original pixel size, rescaling so that the
            # downsampling produces an average in a given region, not a sum
            n_pixels, upsampled_n_pixels = math.prod(shape), math.prod(upsampled_shape)
            if self.shape is None:
                return downsample_to_shape_with_fourier_cropping(
                    in_plane_potential * (n_pixels / upsampled_n_pixels),
                    downsampled_shape=shape,
                    outputs_real_space=outputs_real_space,
                )
            else:
                in_plane_potential = downsample_to_shape_with_fourier_cropping(
                    in_plane_potential * (n_pixels / upsampled_n_pixels),
                    downsampled_shape=shape,
                    outputs_real_space=True,
                )
                in_plane_potential = resize_with_crop_or_pad(
                    in_plane_potential, config.padded_shape
                )
                return (
                    in_plane_potential
                    if outputs_real_space
                    else rfftn(in_plane_potential)
                )
        else:
            if self.shape is None:
                return (
                    in_plane_potential
                    if outputs_real_space
                    else rfftn(in_plane_potential)
                )
            else:
                in_plane_potential = resize_with_crop_or_pad(
                    in_plane_potential, config.padded_shape
                )
                return (
                    in_plane_potential
                    if outputs_real_space
                    else rfftn(in_plane_potential)
                )


def _compute_projected_potential_from_atoms(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    atom_positions: Float[Array, "n_atoms 3"],
    a: Float[Array, "n_atoms n_gaussians_per_atom"],
    b: Float[Array, "n_atoms n_gaussians_per_atom"],
    use_error_functions: bool,
    n_batches: int,
) -> Float[Array, "dim_y dim_x"]:
    # Make the grid on which to evaluate the result
    grid_x = make_1d_coordinate_grid(shape[1], pixel_size)
    grid_y = make_1d_coordinate_grid(shape[0], pixel_size)
    # Get function and pytree to compute potential over a batch of atoms
    xs = (atom_positions, a, b)
    compute_potential_for_atom_group = (
        lambda xs: _compute_projected_potential_from_atom_group(
            grid_x,
            grid_y,
            pixel_size,
            xs[0],
            xs[1],
            xs[2],
            use_error_functions,
        )
    )
    # Compute projection with a call to `jax.lax.map` in batches
    if n_batches > atom_positions.shape[0]:
        raise ValueError(
            "The `n_batches` when computing a projection must "
            "be an integer less than or equal to the number of atoms, "
            f"which is equal to {atom_positions.shape[0]}. Got "
            f"`n_batches = {n_batches}`."
        )
    elif n_batches == 1:
        projection = compute_potential_for_atom_group(xs)
    elif n_batches > 1:
        projection = jnp.sum(
            _batched_map_with_contraction(
                compute_potential_for_atom_group, xs, n_batches
            ),
            axis=0,
        )
    else:
        raise ValueError(
            "The `n_batches` when building a voxel grid must be an "
            "integer greater than or equal to 1."
        )
    return projection


def _compute_projected_potential_from_atom_group(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    pixel_size: Float[Array, ""],
    atom_positions: Float[Array, "n_atoms 3"],
    a: Float[Array, "n_atoms n_gaussians_per_atom"],
    b: Float[Array, "n_atoms n_gaussians_per_atom"],
    use_error_functions: bool,
) -> Float[Array, "dim_y dim_x"]:
    # Evaluate 1D gaussian integrals for each of x, y, and z dimensions

    if use_error_functions:
        result = _compute_gaussian_integrals_for_all_atoms(
            grid_x, grid_y, atom_positions, a, b, pixel_size
        )
    else:
        result = _compute_gaussians_for_all_atoms(grid_x, grid_y, atom_positions, a, b)
    gaussians_times_prefactor_x, gaussians_y = result
    projection = _compute_projected_potential_from_gaussians(
        gaussians_times_prefactor_x, gaussians_y
    )

    return projection


def _compute_projected_potential_from_gaussians(
    gaussians_per_interval_per_atom_x: Float[Array, "dim_x n_atoms n_gaussians_per_atom"],
    gaussians_per_interval_per_atom_y: Float[Array, "dim_y n_atoms n_gaussians_per_atom"],
) -> Float[Array, "dim_y dim_x"]:
    # Prepare matrices with dimensions of the number of atoms and the number of grid
    # points. There are as many matrices as number of gaussians per atom
    gauss_x = jnp.transpose(gaussians_per_interval_per_atom_x, (2, 1, 0))
    gauss_y = jnp.transpose(gaussians_per_interval_per_atom_y, (2, 0, 1))
    # Compute matrix multiplication then sum over the number of gaussians per atom
    return jnp.sum(jnp.matmul(gauss_y, gauss_x), axis=0)


def _compute_gaussian_integrals_for_all_atoms(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    atom_positions: Float[Array, "n_atoms 3"],
    a: Float[Array, "n_atoms n_gaussians_per_atom"],
    b: Float[Array, "n_atoms n_gaussians_per_atom"],
    pixel_size: Float[Array, ""],
) -> tuple[
    Float[Array, "dim_x n_atoms n_gaussians_per_atom"],
    Float[Array, "dim_y n_atoms n_gaussians_per_atom"],
]:
    """Evaluate 1D averaged gaussians in x, y, and z dimensions
    for each atom and each gaussian per atom.
    """
    # Define function to compute integrals for each dimension
    scaling = 2 * jnp.pi / jnp.sqrt(b)
    integration_kernel = lambda delta: (
        jsp.special.erf(scaling[None, :, :] * (delta + pixel_size)[:, :, None])
        - jsp.special.erf(scaling[None, :, :] * delta[:, :, None])
    )
    # Compute outer product of left edge of grid points minus atomic positions
    left_edge_grid_x, left_edge_grid_y = (
        grid_x - pixel_size / 2,
        grid_y - pixel_size / 2,
    )
    delta_x, delta_y = (
        left_edge_grid_x[:, None] - atom_positions[:, 0],
        left_edge_grid_y[:, None] - atom_positions[:, 1],
    )
    # Compute gaussian integrals for each grid point, each atom, and
    # each gaussian per atom
    gauss_x, gauss_y = (integration_kernel(delta_x), integration_kernel(delta_y))
    # Compute the prefactors for each atom and each gaussian per atom
    # for the potential
    prefactor = (4 * jnp.pi * a) / (2 * pixel_size) ** 2
    # Multiply the prefactor onto one of the gaussians for efficiency
    return prefactor * gauss_x, gauss_y


def _compute_gaussians_for_all_atoms(
    grid_x: Float[Array, " x_dim"],
    grid_y: Float[Array, " y_dim"],
    atom_positions: Float[Array, "n_atoms 3"],
    a: Float[Array, "n_atoms n_gaussians_per_atom"],
    b: Float[Array, "n_atoms n_gaussians_per_atom"],
) -> tuple[
    Float[Array, "dim_x n_atoms n_gaussians_per_atom"],
    Float[Array, "dim_y n_atoms n_gaussians_per_atom"],
]:
    b_inverse = 4.0 * jnp.pi / b
    gauss_x = jnp.exp(
        -jnp.pi
        * b_inverse[None, :, :]
        * ((grid_x[:, None] - atom_positions.T[0, :]) ** 2)[:, :, None]
    )
    gauss_y = jnp.exp(
        -jnp.pi
        * b_inverse[None, :, :]
        * ((grid_y[:, None] - atom_positions.T[1, :]) ** 2)[:, :, None]
    )
    prefactor = 4 * jnp.pi * a[None, :, :] * b_inverse[None, :, :]

    return prefactor * gauss_x, gauss_y


def _batched_map_with_contraction(fun, xs, n_batches):
    # ... reshape into an iterative dimension and a batching dimension
    batch_dim = jax.tree.leaves(xs)[0].shape[0]
    batch_size = batch_dim // n_batches
    xs_per_batch = jax.tree.map(
        lambda x: x[: batch_dim - batch_dim % batch_size, ...].reshape(
            (n_batches, batch_size, *x.shape[1:])
        ),
        xs,
    )
    # .. compute the result and reshape back into one leading dimension
    result = jax.lax.map(fun, xs_per_batch)
    # ... if the batch dimension is not divisible by the batch size, need
    # to take care of the remainder
    if batch_dim % batch_size != 0:
        remainder = fun(
            jax.tree.map(lambda x: x[batch_dim - batch_dim % batch_size :, ...], xs)
        )[None, ...]
        result = jnp.concatenate([result, remainder], axis=0)
    return result
