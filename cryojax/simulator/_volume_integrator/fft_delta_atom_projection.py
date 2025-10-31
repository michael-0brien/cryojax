from functools import partial
from typing import ClassVar
from typing_extensions import override

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
    the exit plane using integer rounded coordinates and
    jax.scipy.signal.fftconvolve for Gaussian convolution.
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

        - `n_kernel_truncation_pix`:
            The size of the Gaussian kernel (in pixels) used for convolution.
        - `kernel_width_pix`:
            The width (standard deviation) of the Gaussian kernel (in pixels).
        - `upsample_factor`:
            If not `None`, the factor by which to upsample the image
            to reduce aliasing effects.
        - `shape`:
            If not `None`, the shape of the output projection images.
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

    @override
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
