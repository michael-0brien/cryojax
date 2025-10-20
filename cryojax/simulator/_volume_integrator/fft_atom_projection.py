import math
from typing import Any, ClassVar
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...coordinates import make_frequency_grid
from ...ndimage import (
    convert_fftn_to_rfftn,
    crop_to_shape,
    irfftn,
    operators as op,
)
from .._image_config import AbstractImageConfig
from .._volume import IndependentAtomVolume
from .base_integrator import AbstractVolumeIntegrator


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


class FFTAtomProjection(
    AbstractVolumeIntegrator[IndependentAtomVolume],
    strict=True,
):
    """Integrate atomic parametrization of a volume onto
    the exit plane using non-uniform FFTs plus convolution.
    """

    upsampling_factor: int | None
    eps: float
    opts: Any

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self, *, upsampling_factor: int | None = None, eps: float = 1e-6, opts: Any = None
    ):
        """**Arguments:**

        - `upsampling_factor`:
            The factor by which to upsample the computation of the images.
            If `upsampling_factor` is greater than 1, the images will be computed
            at a higher resolution and then downsampled to the original resolution.
            This can be useful for reducing aliasing artifacts in the images.
        - `eps`:
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        - `opts`:
            A `jax_finufft.options.Opts` or `jax_finufft.options.NestedOpts`
            dataclass.
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        """
        if jnufft is None:
            raise RuntimeError(
                "Tried to use the `FFTAtomProjection` "
                "class, but `jax-finufft` is not installed. "
                "See https://github.com/flatironinstitute/jax-finufft "
                "for installation instructions."
            ) from JAX_FINUFFT_IMPORT_ERROR
        self.upsampling_factor = upsampling_factor
        self.eps = eps
        self.opts = opts

    @override
    def integrate(
        self,
        volume_representation: IndependentAtomVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> (
        Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
        ]
        | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    ):
        pixel_size, shape = image_config.pixel_size, image_config.padded_shape
        if self.upsampling_factor is not None:
            u = self.upsampling_factor
            pixel_size_u, shape_u = (
                pixel_size / u,
                (
                    shape[0] * u,
                    shape[1] * u,
                ),
            )
            frequency_grid_u = make_frequency_grid(
                shape_u, pixel_size_u, outputs_rfftfreqs=False
            )
        else:
            pixel_size_u, shape_u = pixel_size, shape
            frequency_grid_u = image_config.padded_full_frequency_grid_in_angstroms
        frequency_grid_u = jnp.fft.fftshift(frequency_grid_u, axes=(0, 1))
        proj_kernel = lambda pos, kernel: _project_with_nufft(
            shape_u,
            pixel_size_u,
            pos,
            kernel,
            frequency_grid_u,
            eps=self.eps,
            opts=self.opts,
        )
        # Compute projection over atom types
        fourier_projection = jax.tree.reduce(
            lambda x, y: x + y,
            jax.tree.map(
                proj_kernel,
                volume_representation.position_pytree,
                volume_representation.scattering_factor_pytree,
                is_leaf=lambda x: isinstance(x, op.AbstractFourierOperator),
            ),
        )
        if self.upsampling_factor is not None:
            # Downsample back to the original pixel size, rescaling so that the
            # downsampling produces an average in a given region, not a sum
            n_pix, n_pix_u = math.prod(shape), math.prod(shape_u)
            fourier_projection = (n_pix / n_pix_u) * crop_to_shape(
                fourier_projection, shape
            )

        # Shift zero frequency component to corner
        fourier_projection = convert_fftn_to_rfftn(
            jnp.fft.ifftshift(fourier_projection), mode="real"
        )
        return (
            irfftn(fourier_projection, s=shape)
            if outputs_real_space
            else fourier_projection
        )


def _project_with_nufft(shape, ps, pos, kernel, freqs, eps=1e-6, opts=None):
    assert jnufft is not None
    # Get x and y coordinates
    positions_xy = pos[:, :2]
    # Normalize coordinates betweeen -pi and pi
    ny, nx = shape
    box_xy = ps * jnp.asarray((nx, ny))
    positions_periodic = 2 * jnp.pi * positions_xy / box_xy
    # Unpack and compute
    x, y = positions_periodic[:, 0], positions_periodic[:, 1]
    n_atoms = x.size
    area_element = ps**2
    fourier_projection = (
        jnufft.nufft1(
            shape,
            jnp.full((n_atoms,), 1.0 + 0.0j),
            y,
            x,
            eps=eps,
            opts=opts,
            iflag=-1,
        )
        / area_element
    )
    # Evaluate kernel, multiply, and return
    fourier_projection *= kernel(freqs)

    return fourier_projection
