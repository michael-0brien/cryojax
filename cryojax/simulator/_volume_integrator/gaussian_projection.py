import math
from typing import ClassVar, Optional
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Complex, Float

from ...constants import variance_to_b_factor
from ...coordinates import make_1d_coordinate_grid
from ...ndimage import (
    downsample_to_shape_with_fourier_cropping,
    resize_with_crop_or_pad,
    rfftn,
)
from .._image_config import AbstractImageConfig
from .._volume import GaussianMixtureVolume, PengAtomicVolume
from .base_integrator import AbstractVolumeIntegrator


class GaussianMixtureProjection(
    AbstractVolumeIntegrator[GaussianMixtureVolume | PengAtomicVolume],
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
        use_error_functions: bool = True,
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
            cropping to the `AbstractImageConfig.padded_shape`. This argument is particularly
            useful if the `AbstractImageConfig.padded_shape` is much larger than the protein.
        - `use_error_functions`:
            If `True`, use error functions to evaluate the projected volume at
            a pixel to be the average value within the pixel using gaussian
            integrals. If `False`, the volume at a pixel will simply be evaluated
            as a gaussian.
        - `n_batches`:
            The number of batches over groups of positions
            used to evaluate the projection.
            This is useful if GPU memory is exhausted. By default,
            `1`, which computes a projection for all positions at once.
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
        volume_representation: GaussianMixtureVolume | PengAtomicVolume,
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

        - `volume_representation`: The volume representation to project.
        - `image_config`: The configuration of the imaging instrument.

        **Returns:**

        The integrated volume in real or fourier space at the
        `AbstractImageConfig.padded_shape`.
        """  # noqa: E501
        # Grab the image configuration
        shape = image_config.padded_shape if self.shape is None else self.shape
        pixel_size = image_config.pixel_size
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
        if isinstance(volume_representation, PengAtomicVolume):
            positions = volume_representation.atom_positions
            amplitudes = volume_representation.amplitudes
            b_factors = volume_representation.b_factors
        elif isinstance(volume_representation, GaussianMixtureVolume):
            positions = volume_representation.positions
            amplitudes = volume_representation.amplitudes
            b_factors = variance_to_b_factor(volume_representation.variances)
        else:
            raise ValueError(
                "Supported types for `volume_representation` are "
                "`PengAtomicVolume` and `GaussianMixtureVolume`."
            )
        # Compute the projection
        projection_integral = _gaussians_to_projection(
            upsampled_shape,
            upsampled_pixel_size,
            positions,
            amplitudes,
            b_factors,
            self.use_error_functions,
            self.n_batches,
        )
        if self.upsampling_factor is not None:
            # Downsample back to the original pixel size, rescaling so that the
            # downsampling produces an average in a given region, not a sum
            n_pixels, upsampled_n_pixels = math.prod(shape), math.prod(upsampled_shape)
            if self.shape is None:
                return downsample_to_shape_with_fourier_cropping(
                    projection_integral * (n_pixels / upsampled_n_pixels),
                    downsampled_shape=shape,
                    outputs_real_space=outputs_real_space,
                )
            else:
                projection_integral = downsample_to_shape_with_fourier_cropping(
                    projection_integral * (n_pixels / upsampled_n_pixels),
                    downsampled_shape=shape,
                    outputs_real_space=True,
                )
                projection_integral = resize_with_crop_or_pad(
                    projection_integral, image_config.padded_shape
                )
                return (
                    projection_integral
                    if outputs_real_space
                    else rfftn(projection_integral)
                )
        else:
            if self.shape is None:
                return (
                    projection_integral
                    if outputs_real_space
                    else rfftn(projection_integral)
                )
            else:
                projection_integral = resize_with_crop_or_pad(
                    projection_integral, image_config.padded_shape
                )
                return (
                    projection_integral
                    if outputs_real_space
                    else rfftn(projection_integral)
                )


def _gaussians_to_projection(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "n_positions 3"],
    a: Float[Array, "n_positions n_gaussians_per_position"],
    b: Float[Array, "n_positions n_gaussians_per_position"],
    use_error_functions: bool,
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
        use_error_functions,
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
    use_error_functions: bool,
) -> Float[Array, "dim_y dim_x"]:
    # Evaluate 1D gaussian integrals for each of x, y, and z dimensions

    if use_error_functions:
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
    prefactor = (4 * jnp.pi * a) / (2 * pixel_size) ** 2
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
