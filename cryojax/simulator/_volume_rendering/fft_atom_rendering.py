from typing import Any
from typing_extensions import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...coordinates import make_frequency_grid
from ...jax_util import NDArrayLike, error_if_not_positive
from ...ndimage import convert_fftn_to_rfftn, ifftn, operators as op
from .._volume import IndependentAtomVolume
from .base_rendering import AbstractVolumeRenderFn


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


class GaussianMixtureRenderFn(AbstractVolumeRenderFn[IndependentAtomVolume], strict=True):
    """Render a voxel grid using non-uniform FFTs and convoluton."""

    shape: tuple[int, int, int]
    voxel_size: Float[Array, ""]
    frequency_grid: Float[Array, "_ _ _ 3"] | None
    antialias: bool
    eps: float
    opts: Any

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        frequency_grid: Float[Array, "_ _ _ 3"] | None = None,
        antialias: bool = True,
        eps: float = 1e-6,
        opts: Any = None,
    ):
        """**Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `frequency_grid`:
            An optional frequency grid for rendering the
            volume. If `None`, compute on the fly. The grid
            should be in inverse angstroms and have the zero
            frequency component in the center, i.e.

            ```python
            frequency_grid = jnp.fft.fftshift(
                make_frequency_grid(shape, voxel_size, outputs_rfft=True)
            )
            ```
        - `antialias`:
            If `True`, apply an anti-aliasing filter to more accurately
            sample the volume.
        - `eps`:
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        - `opts`:
            A `jax_finufft.options.Opts` or `jax_finufft.options.NestedOpts`
            dataclass.
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        """
        self.shape = shape
        self.voxel_size = error_if_not_positive(jnp.asarray(voxel_size, dtype=float))
        self.frequency_grid = frequency_grid
        self.antialias = antialias
        self.eps = eps
        self.opts = opts

    @override
    def __call__(
        self,
        volume_representation: IndependentAtomVolume,
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
        if self.frequency_grid is None:
            frequency_grid = jnp.fft.fftshift(
                make_frequency_grid(self.shape, self.voxel_size), axes=(0, 1, 2)
            )
        else:
            frequency_grid = self.frequency_grid
        proj_kernel = lambda pos, kernel: _render_with_nufft(
            self.shape,
            self.voxel_size,
            pos,
            kernel,
            frequency_grid,
            eps=self.eps,
            opts=self.opts,
        )
        # Compute projection over atom types
        fourier_voxel_grid = jax.tree.reduce(
            lambda x, y: x + y,
            jax.tree.map(
                proj_kernel,
                volume_representation.position_pytree,
                volume_representation.scattering_factor_pytree,
                is_leaf=lambda x: isinstance(x, op.AbstractFourierOperator),
            ),
        )
        if self.antialias:
            antialias_fn = op.FourierSinc(box_width=self.voxel_size)
            fourier_voxel_grid *= antialias_fn(frequency_grid)

        if outputs_real_space:
            return ifftn(jnp.fft.ifftshift(fourier_voxel_grid)).real
        else:
            if outputs_rfft:
                fourier_voxel_grid = convert_fftn_to_rfftn(
                    jnp.fft.ifftshift(fourier_voxel_grid), mode="real"
                )
                if fftshifted:
                    return jnp.fft.fftshift(fourier_voxel_grid, axes=(0, 1))
                else:
                    return fourier_voxel_grid
            else:
                if fftshifted:
                    return fourier_voxel_grid
                else:
                    return jnp.fft.ifftshift(fourier_voxel_grid)


def _render_with_nufft(shape, ps, pos, kernel, freqs, eps=1e-6, opts=None):
    assert jnufft is not None
    # Get x and y coordinates
    # Normalize coordinates betweeen -pi and pi
    nz, ny, nx = shape
    box_xyz = ps * jnp.asarray((nx, ny, nz))
    pos_periodic = 2 * jnp.pi * pos / box_xyz
    # Unpack
    x, y, z = pos_periodic[:, 0], pos_periodic[:, 1], pos_periodic[:, 2]
    n_atoms = x.size
    volume_element = ps**3
    # Compute
    fourier_projection = kernel(freqs) * (
        jnufft.nufft1(
            shape,
            jnp.full((n_atoms,), 1.0 + 0.0j),
            z,
            y,
            x,
            eps=eps,
            opts=opts,
            iflag=-1,
        )
        / volume_element
    )

    return fourier_projection
