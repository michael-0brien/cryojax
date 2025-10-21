from typing import Any, ClassVar
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from cryojax.ndimage._edges import resize_with_crop_or_pad

from ...coordinates import make_frequency_grid
from ...ndimage import (
    convert_fftn_to_rfftn,
    irfftn,
    operators as op,
    rfftn,
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

    antialias: bool
    shape: tuple[int, int] | None
    eps: float
    opts: Any

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        antialias: bool = True,
        shape: tuple[int, int] | None = None,
        eps: float = 1e-6,
        opts: Any = None,
    ):
        """**Arguments:**

        - `antialias`:
            If `True`, apply an anti-aliasing filter to more accurately
            sample the volume.
        - `shape`:
            If given, first compute the image at `shape`, then
            pad or crop to `image_config.padded_shape`.
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
        self.antialias = antialias
        self.shape = shape
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
        pixel_size = image_config.pixel_size
        if self.shape is not None:
            shape = self.shape
            frequency_grid = make_frequency_grid(
                shape, pixel_size, outputs_rfftfreqs=False
            )
        else:
            shape = image_config.padded_shape
            frequency_grid = image_config.padded_full_frequency_grid_in_angstroms
        frequency_grid = jnp.fft.fftshift(frequency_grid, axes=(0, 1))
        proj_kernel = lambda pos, kernel: _project_with_nufft(
            shape, pixel_size, pos, kernel, frequency_grid, eps=self.eps, opts=self.opts
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
        if self.antialias:
            antialias_fn = op.FourierSinc(box_width=pixel_size)
            fourier_projection *= antialias_fn(frequency_grid)

        # Shift zero frequency component to corner
        fourier_projection = convert_fftn_to_rfftn(
            jnp.fft.ifftshift(fourier_projection), mode="real"
        )
        if self.shape is None:
            return (
                irfftn(fourier_projection, s=shape)
                if outputs_real_space
                else fourier_projection
            )
        else:
            projection = resize_with_crop_or_pad(
                irfftn(fourier_projection, s=shape), image_config.padded_shape
            )
            return projection if outputs_real_space else rfftn(projection)


def _project_with_nufft(shape, ps, pos, kernel, freqs, eps=1e-6, opts=None):
    assert jnufft is not None
    # Get x and y coordinates
    positions_xy = pos[:, :2]
    # Normalize coordinates betweeen -pi and pi
    ny, nx = shape
    box_xy = ps * jnp.asarray((nx, ny))
    positions_periodic = 2 * jnp.pi * positions_xy / box_xy
    # Unpack
    x, y = positions_periodic[:, 0], positions_periodic[:, 1]
    n_atoms = x.size
    area_element = ps**2
    # Compute
    fourier_projection = kernel(freqs) * (
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

    return fourier_projection
