from typing import Any, ClassVar
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...ndimage import convert_fftn_to_rfftn, irfftn, operators as op
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

    eps: float
    opts: Any

    is_projection_approximation: ClassVar[bool] = True

    def __init__(self, *, eps: float = 1e-6, opts: Any = None):
        """**Arguments:**

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
        shape, pixel_size = image_config.padded_shape, image_config.pixel_size
        frequency_grid = image_config.padded_frequency_grid_in_angstroms
        proj_kernel = lambda pos, kernel: _project_with_nufft(
            shape, pixel_size, pos, kernel, frequency_grid, eps=self.eps, opts=self.opts
        )

        fourier_projection = jax.tree.reduce(
            lambda x, y: x + y,
            jax.tree.map(
                proj_kernel,
                volume_representation.position_pytree,
                volume_representation.scattering_factor_pytree,
                is_leaf=lambda x: isinstance(x, op.AbstractFourierOperator),
            ),
        )

        if outputs_real_space:
            return irfftn(fourier_projection, s=shape)
        else:
            return fourier_projection


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
    # Shift zero frequency component to corner and convert to half-space
    fourier_projection = convert_fftn_to_rfftn(
        jnp.fft.ifftshift(fourier_projection), mode="real"
    )
    # Evaluate kernel, multiply, and return
    fourier_projection *= kernel(freqs)

    return fourier_projection
