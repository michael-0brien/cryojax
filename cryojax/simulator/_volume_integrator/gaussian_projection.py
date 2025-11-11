import warnings
from typing import ClassVar, Literal
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Complex, Float

from ...constants import variance_to_b_factor
from ...coordinates import make_1d_coordinate_grid
from ...ndimage import resize_with_crop_or_pad, rfftn
from .._image_config import AbstractImageConfig
from .._volume import GaussianMixtureVolume
from .base_integrator import AbstractVolumeIntegrator


class GaussianMixtureProjection(
    AbstractVolumeIntegrator[GaussianMixtureVolume],
    strict=True,
):
    sampling_mode: Literal["average", "point"]
    shape: tuple[int, int] | None
    n_batches: int

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        upsampling_factor: int | None = None,
        shape: tuple[int, int] | None = None,
        sampling_mode: Literal["average", "point"] = "average",
        use_error_functions: bool = False,
        n_batches: int = 1,
    ):
        """**Arguments:**

        - `shape`:
            The shape of the plane on which projections are computed before padding or
            cropping to the `AbstractImageConfig.padded_shape`. This argument is particularly
            useful if the `AbstractImageConfig.padded_shape` is much larger than the protein.
        - `sampling_mode`:
            If `'average'`, use error functions to sample the projected volume at
            a pixel to be the average value using gaussian
            integrals. If `'point'`, the volume at a pixel will
            be evaluated by evaluating the gaussian at a point.
        - `n_batches`:
            The number of batches over groups of positions
            used to evaluate the projection. By default, `n_batches = 1`,
            which computes a projection for all positions at once.
            This is useful to decrease GPU memory usage.
        """  # noqa: E501
        if use_error_functions:
            warnings.warn(
                "`use_error_functions` in `GaussianMixtureProjection` has "
                "been deprecated and will be removed in cryoJAX 0.6.0. "
                "This has been renamed to `sampling_mode = 'average'`.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        if upsampling_factor is not None:
            raise ValueError(
                "`upsampling_factor` in `GaussianMixtureProjection` "
                "has been deprecated as of cryoJAX 0.5.1. The "
                "functionality this implemented was not as intended."
            )
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `GaussianMixtureProjection` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        self.shape = shape
        self.sampling_mode = sampling_mode
        self.n_batches = n_batches

    @override
    def integrate(
        self,
        volume_representation: GaussianMixtureVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ]
        | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    ):
        """Compute a projection from gaussians.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The configuration of the resulting image.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in fourier.

        **Returns:**

        The integrated volume in real or fourier space at the
        `AbstractImageConfig.padded_shape`.
        """  # noqa: E501
        # Grab the image configuration
        shape = image_config.padded_shape if self.shape is None else self.shape
        pixel_size = image_config.pixel_size
        # Grab the gaussian amplitudes and widths
        positions = volume_representation.positions
        amplitudes = volume_representation.amplitudes
        b_factors = variance_to_b_factor(volume_representation.variances)
        # Compute the projection
        use_erf = True if self.sampling_mode == "average" else False
        projection_integral = _gaussians_to_projection(
            shape,
            pixel_size,
            positions,
            amplitudes,
            b_factors,
            use_erf,
            self.n_batches,
        )
        if self.shape is None:
            return (
                projection_integral if outputs_real_space else rfftn(projection_integral)
            )
        else:
            projection_integral = resize_with_crop_or_pad(
                projection_integral, image_config.padded_shape
            )
            return (
                projection_integral if outputs_real_space else rfftn(projection_integral)
            )


def _gaussians_to_projection(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "n_positions 3"],
    a: Float[Array, "n_positions n_gaussians_per_position"],
    b: Float[Array, "n_positions n_gaussians_per_position"],
    use_erf: bool,
    n_batches: int,
) -> Float[Array, "dim_y dim_x"]:
    # Make the grid on which to evaluate the result
    grid_x = make_1d_coordinate_grid(shape[1], pixel_size)
    grid_y = make_1d_coordinate_grid(shape[0], pixel_size)
    # Get function and pytree to compute volume over a batch of positions
    xs = (positions, a, b)
    kernel_fn = lambda xs: _gaussians_to_projection_kernel(
        grid_x,
        grid_y,
        pixel_size,
        xs[0],
        xs[1],
        xs[2],
        use_erf,
    )
    # Compute projection with a call to `jax.lax.map` in batches
    if n_batches > positions.shape[0]:
        raise ValueError(
            "The `n_batches` when computing a projection must "
            "be an integer less than or equal to the number of positions, "
            f"which is equal to {positions.shape[0]}. Got "
            f"`n_batches = {n_batches}`."
        )
    elif n_batches == 1:
        projection = kernel_fn(xs)
    elif n_batches > 1:
        projection = jnp.sum(
            _batched_map_with_contraction(kernel_fn, xs, n_batches),
            axis=0,
        )
    else:
        raise ValueError(
            "The `n_batches` argument for `GaussianMixtureProjection` must be an "
            "integer greater than or equal to 1."
        )
    return projection


def _gaussians_to_projection_kernel(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "n_positions 3"],
    a: Float[Array, "n_positions n_gaussians_per_position"],
    b: Float[Array, "n_positions n_gaussians_per_position"],
    use_erf: bool,
) -> Float[Array, "dim_y dim_x"]:
    # Evaluate 1D gaussian integrals for each of x, y, and z dimensions

    if use_erf:
        gaussians_times_prefactor_x, gaussians_y = _evaluate_gaussian_integrals(
            grid_x, grid_y, positions, a, b, pixel_size
        )
    else:
        gaussians_times_prefactor_x, gaussians_y = _evaluate_gaussians(
            grid_x, grid_y, positions, a, b
        )
    projection = _evaluate_multivariate_gaussian(gaussians_times_prefactor_x, gaussians_y)

    return projection


def _evaluate_multivariate_gaussian(
    gaussians_per_interval_per_position_x: Float[
        Array, "dim_x n_positions n_gaussians_per_position"
    ],
    gaussians_per_interval_per_position_y: Float[
        Array, "dim_y n_positions n_gaussians_per_position"
    ],
) -> Float[Array, "dim_y dim_x"]:
    # Prepare matrices with dimensions of the number of positions and the number of grid
    # points. There are as many matrices as number of gaussians per position
    gauss_x = jnp.transpose(gaussians_per_interval_per_position_x, (2, 1, 0))
    gauss_y = jnp.transpose(gaussians_per_interval_per_position_y, (2, 0, 1))
    # Compute matrix multiplication then sum over the number of gaussians per position
    return jnp.sum(jnp.matmul(gauss_y, gauss_x), axis=0)


def _evaluate_gaussian_integrals(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    positions: Float[Array, "n_positions 3"],
    a: Float[Array, "n_positions n_gaussians_per_position"],
    b: Float[Array, "n_positions n_gaussians_per_position"],
    pixel_size: Float[Array, ""],
) -> tuple[
    Float[Array, "dim_x n_positions n_gaussians_per_position"],
    Float[Array, "dim_y n_positions n_gaussians_per_position"],
]:
    """Evaluate 1D averaged gaussians in x, y, and z dimensions
    for each position and each gaussian per position.
    """
    # Define function to compute integrals for each dimension
    scaling = 2 * jnp.pi / jnp.sqrt(b)
    integration_kernel = lambda delta: (
        jsp.special.erf(scaling[None, :, :] * (delta + pixel_size)[:, :, None])
        - jsp.special.erf(scaling[None, :, :] * delta[:, :, None])
    )
    # Compute outer product of left edge of grid points minus positions
    left_edge_grid_x, left_edge_grid_y = (
        grid_x - pixel_size / 2,
        grid_y - pixel_size / 2,
    )
    delta_x, delta_y = (
        left_edge_grid_x[:, None] - positions[:, 0],
        left_edge_grid_y[:, None] - positions[:, 1],
    )
    # Compute gaussian integrals for each grid point, each position, and
    # each gaussian per position
    gauss_x, gauss_y = (integration_kernel(delta_x), integration_kernel(delta_y))
    # Compute the prefactors for each position and each gaussian per position
    # for the volume
    prefactor = a / (2 * pixel_size) ** 2
    # Multiply the prefactor onto one of the gaussians for efficiency
    return prefactor * gauss_x, gauss_y


def _evaluate_gaussians(
    grid_x: Float[Array, " x_dim"],
    grid_y: Float[Array, " y_dim"],
    positions: Float[Array, "n_positions 3"],
    a: Float[Array, "n_positions n_gaussians_per_position"],
    b: Float[Array, "n_positions n_gaussians_per_position"],
) -> tuple[
    Float[Array, "dim_x n_positions n_gaussians_per_position"],
    Float[Array, "dim_y n_positions n_gaussians_per_position"],
]:
    b_inverse = 4.0 * jnp.pi / b
    gauss_x = jnp.exp(
        -jnp.pi
        * b_inverse[None, :, :]
        * ((grid_x[:, None] - positions.T[0, :]) ** 2)[:, :, None]
    )
    gauss_y = jnp.exp(
        -jnp.pi
        * b_inverse[None, :, :]
        * ((grid_y[:, None] - positions.T[1, :]) ** 2)[:, :, None]
    )
    prefactor = a[None, :, :] * b_inverse[None, :, :]

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
