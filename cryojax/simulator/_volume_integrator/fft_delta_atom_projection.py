from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve

from ...ndimage import (
    rfftn,
)
from .._image_config import AbstractImageConfig
from .._volume import IndependentAtomVolume
from .base_integrator import AbstractVolumeIntegrator


class FFTDeltaAtomProjection(
    AbstractVolumeIntegrator[IndependentAtomVolume],
    strict=True,
):
    """Integrate atomic parametrization of a volume onto
    the exit plane using non-uniform FFTs plus convolution.
    """

    n_kernel_truncation_pix: int
    kernel_width_pix: float
    upsample_factor: int | None
    shape: tuple[int, int] | None

    is_projection_approximation: ClassVar[bool] = True

    def __init__(
        self,
        *,
        n_kernel_truncation_pix: int = 5,
        kernel_width_pix: float = 1.0,
        upsample_factor: int | None = None,
        shape: tuple[int, int] | None = None,
    ):
        """**Arguments:**

        - `antialias`:
            If `True`, apply an anti-aliasing filter to more accurately
            sample the volume.
        - `upsample_factor`:
            If provided, first compute an upsampled version of the
            image at pixel size `image_config.pixel_size / upsample_factor`.
            Then, downsample with `cryojax.ndimage.block_reduce_downsample`
            to locally average to the correct pixel size. This is useful
            for reducing aliasing.
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

        self.upsample_factor = upsample_factor
        self.shape = shape

        self.n_kernel_truncation_pix = n_kernel_truncation_pix
        self.kernel_width_pix = kernel_width_pix

    def __check_init__(self):
        if self.upsample_factor is not None:
            if self.upsample_factor % 2 == 0:
                raise ValueError(
                    f"Set `upsample_factor = {self.upsample_factor}` when instantiating "
                    "`FFTAtomProjection`, but only odd `upsample_factor` are supported."
                )

    def integrate(
        self,
        atom_volume: IndependentAtomVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = True,
    ):
        n_kernel_truncation_pix = self.n_kernel_truncation_pix
        kernel_width_pix = self.kernel_width_pix
        n_batch = 1
        n_atoms = atom_volume.position_pytree.shape[0]

        pixel_size = image_config.pixel_size
        shape = image_config.padded_shape if self.shape is None else self.shape
        n_pix = shape[0]

        # Random atom positions
        atom_centers_batch = (
            atom_volume.position_pytree[:, :2].reshape(n_batch, n_atoms, 2) / pixel_size
        )
        amplitudes = jnp.ones((n_batch, n_atoms), dtype=jnp.float32)
        real_space_proj = project_vec_batch_fft_weighted_same_sigma(
            n_pix,
            n_kernel_truncation_pix,
            kernel_width_pix,
            atom_centers_batch,
            amplitudes,
        )
        if outputs_real_space:
            return real_space_proj
        else:
            return rfftn(real_space_proj.reshape(n_pix, n_pix))

    # @override
    # def integrate(
    #     self,
    #     volume_representation: IndependentAtomVolume,
    #     image_config: AbstractImageConfig,
    #     outputs_real_space: bool = False,
    # ) -> (
    #     Complex[
    #         Array,
    #         "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
    #     ]
    #     | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    # ):
    #     """Compute a projection from scattering factors per atom type
    #     from the `IndependentAtomVolume`.

    #     **Arguments:**

    #     - `volume_representation`:
    #         The volume representation.
    #     - `image_config`:
    #         The configuration of the resulting image.
    #     - `outputs_real_space`:
    #         If `True`, return the image in real space. Otherwise,
    #         return in fourier.

    #     **Returns:**

    #     The integrated volume in real or fourier space at the
    #     `AbstractImageConfig.padded_shape`.
    #     """  # noqa: E501
    #     u = self.upsample_factor
    #     pixel_size = image_config.pixel_size
    #     shape = image_config.padded_shape if self.shape is None else self.shape
    #     if u is None:
    #         shape_u, pixel_size_u = shape, pixel_size
    #     else:
    #         shape_u, pixel_size_u = (u * shape[0], u * shape[1]), pixel_size / u
    #     if shape_u == image_config.padded_shape:
    #         frequency_grid = image_config.padded_full_frequency_grid_in_angstroms
    #     else:
    #         frequency_grid = make_frequency_grid(
    #             shape_u, pixel_size_u, outputs_rfftfreqs=False
    #         )
    #     frequency_grid = jnp.fft.fftshift(frequency_grid, axes=(0, 1))
    #     proj_kernel = lambda pos, kernel: _project_with_nufft(
    #         shape_u,
    #         pixel_size_u,
    #         pos,
    #         kernel,
    #         frequency_grid,
    #         eps=self.eps,
    #         opts=self.opts,
    #     )
    #     # Compute projection over atom types
    #     fourier_projection = jax.tree.reduce(
    #         lambda x, y: x + y,
    #         jax.tree.map(
    #             proj_kernel,
    #             volume_representation.position_pytree,
    #             volume_representation.scattering_factor_pytree,
    #             is_leaf=lambda x: isinstance(x, op.AbstractFourierOperator),
    #         ),
    #     )
    #     # Apply anti-aliasing filter
    #     if self.antialias:
    #         antialias_fn = op.FourierSinc(box_width=pixel_size_u)
    #         fourier_projection *= antialias_fn(frequency_grid)
    #     # Shift zero frequency component to corner and convert to
    #     # rfft
    #     fourier_projection = convert_fftn_to_rfftn(
    #         jnp.fft.ifftshift(fourier_projection), mode="real"
    #     )
    #     if self.shape is None:
    #         if u is None:
    #             return (
    #                 irfftn(fourier_projection, s=shape)
    #                 if outputs_real_space
    #                 else fourier_projection
    #             )
    #         else:
    #             projection = _block_average(irfftn(fourier_projection, s=shape_u), u)
    #             return projection if outputs_real_space else rfftn(projection)
    #     else:
    #         projection = irfftn(fourier_projection, s=shape_u)
    #         if u is not None:
    #             projection = _block_average(projection, u)
    #         projection = resize_with_crop_or_pad(projection, image_config.padded_shape)
    #         return projection if outputs_real_space else rfftn(projection)


def precompute_gaussian_kernel(n_trunc, sigma):
    """
    Returns a (n_trunc, n_trunc) Gaussian kernel.
    Centered at 0,0.
    """
    nt_ = (n_trunc - 1) // 2
    x = jnp.arange(-nt_, nt_ + 1)
    mx, my = jnp.meshgrid(x, x, indexing="xy")
    a = -1.0 / (2.0 * sigma**2)
    kernel = jnp.exp(a * (mx**2 + my**2))
    return kernel


@partial(jax.jit, static_argnames=("n_pix", "n_trunc"))
def project_vec_batch_fft_weighted_same_sigma(
    n_pix, n_trunc, sigma, atom_centers_batch, amplitudes
):
    """
    Project Gaussian blobs using delta image + FFT convolution, with per-atom amplitudes.

    Parameters
    ----------
    n_pix : int, image width/height
    n_trunc : int, Gaussian kernel size
    sigma : float
    atom_centers_batch : (n_batch, n_atoms, 2)
    amplitudes : (n_batch, n_atoms ) amplitude per atom

    Returns
    -------
    proj : (n_batch, n_pix*n_pix) flattened dense images
    """

    # Round and shift to image coords
    coords = jnp.round(atom_centers_batch + n_pix // 2).astype(jnp.int32)
    # coords = jnp.mod(coords, n_pix)  # periodic boundary

    # Precompute Gaussian kernel
    kernel = precompute_gaussian_kernel(n_trunc, sigma)

    def project_single_batch(coords_batch, amps_batch):
        """
        coords_batch: (n_atoms, 2)
        amps_batch: (n_atoms,)
        """
        # Create delta image with amplitudes
        img = jnp.zeros((n_pix, n_pix), dtype=jnp.float32)
        img = img.at[coords_batch[:, 0], coords_batch[:, 1]].add(amps_batch)
        # Convolve with Gaussian kernel
        proj = fftconvolve(img, kernel, mode="same")
        return proj.reshape(-1)  # flatten

    # Vectorize over batch
    proj_batch = jax.vmap(project_single_batch)(
        coords, amplitudes
    )  # (n_batch, n_pix*n_pix)
    return proj_batch
