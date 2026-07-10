from collections.abc import Mapping
from typing import ClassVar, Literal, Self
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, Float, PyTree

from ..._internal import error_if_not_positive, leaf_asarray
from ...constants import PengScatteringFactorParameters, b_factor_to_variance
from ...jax_util import FloatLike, NDArrayLike, filter_bscan
from ...ndimage import (
    make_1d_coordinate_grid,
    resize_with_crop_or_pad,
    spread_gaussians_2d,
    spread_gaussians_3d,
)
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import (
    AbstractAtomVolume,
    AbstractVolumeIntegrator,
    AbstractVolumeRenderFn,
    ProjectionArray,
    VoxelArray,
)
from .common import spread_and_sum_gaussian_components


def _leaf_broadcast_to(x, shape):
    """Broadcast an array leaf to `shape`, preserving its backend (NumPy stays
    on the host, JAX stays on-device)."""
    if isinstance(x, np.ndarray):
        return np.broadcast_to(x, shape)
    return jnp.broadcast_to(x, shape)


class GaussianMixtureVolume(AbstractAtomVolume, strict=True):
    r"""A representation of a volume as a mixture of
    gaussians, with multiple gaussians used per position.

    The convention of allowing multiple gaussians per position
    follows "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996). The
    $a$ and $b$ parameters in this work correspond to
    `amplitudes = a` and `variances = b / 8\pi^2`.

    !!! info
        Use the following to load a `GaussianMixtureVolume`
        from these tabulated electron scattering factors.

        ```python
        from cryojax.constants import PengScatteringFactorParameters
        from cryojax.io import read_atoms_from_pdb
        from cryojax.simulator import GaussianMixtureVolume

        # Load positions of atoms and one-hot encoded atom names
        atom_positions, atom_types = read_atoms_from_pdb(...)
        parameters = PengScatteringFactorParameters(atom_types)
        potential = GaussianMixtureVolume.from_tabulated_parameters(
            atom_positions, parameters
        )
        ```
    """

    positions: Float[NDArrayLike, "M 3"]
    amplitudes: Float[NDArrayLike, "M K"]
    variances: Float[NDArrayLike, " M K"]

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        positions: Float[NDArrayLike, "M 3"],
        amplitudes: (
            float
            | Float[NDArrayLike, ""]
            | Float[NDArrayLike, " M"]
            | Float[NDArrayLike, "M K"]
        ),
        variances: (
            float
            | Float[NDArrayLike, ""]
            | Float[NDArrayLike, " M"]
            | Float[NDArrayLike, "M K"]
        ),
    ):
        """**Arguments:**

        - `positions`:
            The coordinates of the gaussians in units of angstroms.
        - `amplitudes`:
            The amplitude for each gaussian.
            To simulate in physical units of a scattering potential,
            this should have units of angstroms.
        - `variances`:
            The variance for each gaussian. This has units of angstroms
            squared.
        """
        # Convert inputs to array leaves, preserving their backend (JAX arrays
        # stay on-device, NumPy arrays and Python scalars stay on the host).
        positions = leaf_asarray(positions, dtype=float)
        amplitudes = leaf_asarray(amplitudes, dtype=float)
        variances = leaf_asarray(variances, dtype=float)
        M = positions.shape[0]
        K = 1
        if amplitudes.ndim == 2:
            K = amplitudes.shape[-1]
        elif amplitudes.ndim == 1:
            amplitudes = amplitudes[:, None]
        elif amplitudes.ndim == 0:
            amplitudes = amplitudes[None, None]
        else:
            raise ValueError(
                "Passed `amplitudes` to `GaussianMixtureVolume` "
                f"with shape {amplitudes.shape}, but must be of "
                "shape `()`, `(M,)`, or "
                "`(M, K)`."
            )
        if variances.ndim == 2:
            K = variances.shape[-1]
        elif variances.ndim == 1:
            variances = variances[:, None]
        elif variances.ndim == 0:
            variances = variances[None, None]
        else:
            raise ValueError(
                "Passed `variances` to `GaussianMixtureVolume` "
                f"with shape {variances.shape}, but must be of "
                "shape `()`, `(M,)`, or "
                "`(M, K)`."
            )

        self.positions = positions
        self.amplitudes = _leaf_broadcast_to(amplitudes, (M, K))
        self.variances = _leaf_broadcast_to(variances, (M, K))

    def __check_init__(self):
        if not (
            self.positions.shape[0] == self.amplitudes.shape[0] == self.variances.shape[0]
        ):
            raise ValueError(
                "The number of positions in `GaussianMixtureVolume` was "
                f"{self.positions.shape[0]}, but `amplitudes` shape was "
                f"{self.amplitudes.shape} and `variances` shape was "
                f"{self.variances.shape}. The first dimension must be equal "
                "to the number of positions."
            )
        if not (self.amplitudes.shape == self.variances.shape):
            raise ValueError(
                "In `GaussianMixtureVolume`, `amplitudes` and "
                f"`variances` shape must be equal. Found shapes "
                f"{self.amplitudes.shape} and {self.variances.shape}, "
                "respectively."
            )

    @classmethod
    def from_tabulated_parameters(
        cls,
        atom_positions: Float[NDArrayLike, "n_atoms 3"],
        parameters: PengScatteringFactorParameters,
        extra_b_factors: FloatLike | Float[NDArrayLike, " n_atoms"] | None = None,
    ) -> Self:
        """Initialize a `GaussianMixtureVolume` from tabulated electron
        scattering factor parameters (Peng et al. 1996). This treats
        the scattering potential as a mixture of five gaussians
        per atom.

        **References:**

        - Peng, L-M. "Electron atomic scattering factors and scattering potentials of crystals."
            Micron 30.6 (1999): 625-648.
        - Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic
            scattering factors." Acta Crystallographica Section A: Foundations of Crystallography
            52.2 (1996): 257-276.

        **Arguments:**

        - `atom_positions`:
            The coordinates of the atoms in units of angstroms.
        - `parameters`:
            A pytree for the scattering factor parameters from
            Peng et al. (1996).
        - `extra_b_factors`:
            Additional per-atom B-factors that are added to
            the values in `scattering_parameters.b`.
        """  # noqa: E501
        amplitudes = jnp.asarray(parameters.a, dtype=float)
        b_factors = jnp.asarray(parameters.b, dtype=float)
        if extra_b_factors is not None:
            extra_b_factors = jnp.asarray(extra_b_factors, dtype=float)
            if extra_b_factors.ndim == 1:
                extra_b_factors = extra_b_factors[:, None]
            b_factors += extra_b_factors
        return cls(atom_positions, amplitudes, b_factor_to_variance(b_factors))

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        positions = jnp.asarray(self.positions)
        return eqx.tree_at(
            lambda d: d.positions,
            self,
            pose.rotate_coordinates(positions, inverse=self.is_frame_rotation),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        offset_in_angstroms = jnp.asarray(pose.offset_in_angstroms)
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        return eqx.tree_at(
            lambda d: d.positions, self, jnp.asarray(self.positions) + offset_in_angstroms
        )


class GaussianMixtureProjection(
    AbstractVolumeIntegrator[GaussianMixtureVolume],
    strict=True,
):
    """
    !!! example "Speed up gradients with Pallas"
        ```python
        integrator = cxs.GaussianMixtureProjection(
            n_spread=7, enable_pallas={"bwd": True}
        )
        ```
    """

    sampling_mode: Literal["average", "point"]
    shape: tuple[int, int] | None
    n_batches: int
    n_spread: int | tuple[int, ...] | None
    enable_pallas: bool | Mapping[str, bool] | None

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        shape: tuple[int, int] | None = None,
        sampling_mode: Literal["average", "point"] = "average",
        n_batches: int = 1,
        n_spread: int | tuple[int, ...] | None = None,
        enable_pallas: bool | Mapping[str, bool] | None = None,
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
            This is useful to decrease GPU memory usage. Applies to both
            the dense (`n_spread=None`) and spreading (`n_spread` set) backends.
        - `n_spread`:
            If `None` (default), compute the projection with dense gaussian
            integrals evaluated over the whole grid. If an `int`, instead
            directly spread each gaussian onto only the `n_spread` nearest
            grid points (per dimension), trading accuracy for speed.
            If a `tuple` of `int`s (one value per gaussian component, i.e. of
            length `GaussianMixtureVolume.amplitudes.shape[-1]`), spread each
            gaussian component with its own width instead of one shared
            width -- useful when a volume's gaussian components have widths
            spanning an order of magnitude or more (e.g. X-ray/electron
            scattering factors written as a sum of 5 gaussians), where a
            single `n_spread` would either truncate the widest components or
            waste computation spreading the narrowest ones too widely. See
            [`cryojax.simulator.suggest_n_spread`][] to choose these values
            from `volume_representation.variances`.
        - `enable_pallas`:
            Use the Pallas/Triton GPU backend instead of pure-JAX for the
            `n_spread` spreading backend (ignored if `n_spread` is `None`).
            [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) is
            JAX's framework for writing custom GPU/TPU kernels. This is most
            advantageous for the *backward* pass -- `{"bwd": True}`. See
            [`cryojax.ndimage.spread_gaussians_2d`][]'s `enable_pallas` for
            the full picture. `None` (default) defers to `CRYOJAX_ENABLE_PALLAS`.
        """  # noqa: E501
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
        self.n_spread = n_spread
        self.enable_pallas = enable_pallas

    @override
    def integrate(
        self,
        volume_representation: GaussianMixtureVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionArray:
        """Compute a projection from gaussians.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The image configuration.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in Fourier.

        **Returns:**

        The volume projection in real or Fourier space at the
        `AbstractImageConfig.padded_shape` and the `image_config.pixel_size`.
        """  # noqa: E501
        # Grab the image configuration
        shape = image_config.padded_shape if self.shape is None else self.shape
        pixel_size = jnp.asarray(image_config.pixel_size)
        # Grab the gaussian amplitudes and widths, casting to JAX arrays
        positions = jnp.asarray(volume_representation.positions)
        amplitudes = jnp.asarray(volume_representation.amplitudes)
        variances = error_if_not_positive(jnp.asarray(volume_representation.variances))
        use_erf = self.sampling_mode == "average"
        context = (
            f"Error during projection using `{type(self).__name__}(..., n_batches=...)`"
        )
        # Compute the projection
        if self.n_spread is None:
            projection_integral = _gaussians_to_projection_dense(
                shape,
                pixel_size,
                positions,
                amplitudes,
                variances,
                use_erf,
                self.n_batches,
                context,
            )
        else:
            projection_integral = _gaussians_to_projection_spread(
                shape,
                pixel_size,
                positions,
                amplitudes,
                variances,
                use_erf,
                self.n_spread,
                self.n_batches,
                context,
                self.enable_pallas,
            )
        if self.shape is None:
            return (
                projection_integral
                if outputs_real_space
                else jnp.fft.rfftn(projection_integral)
            )
        else:
            projection_integral = resize_with_crop_or_pad(
                projection_integral, image_config.padded_shape
            )
            return (
                projection_integral
                if outputs_real_space
                else jnp.fft.rfftn(projection_integral)
            )


class GaussianMixtureRenderFn(AbstractVolumeRenderFn[GaussianMixtureVolume], strict=True):
    """Render a voxel grid from the `GaussianMixtureVolume`.

    If `GaussianMixtureVolume` is instantiated from electron scattering
    factors via `from_tabulated_parameters`, this renders an electrostatic
    potential as tabulated in Peng et al. 1996. The elastic electron
    scattering factors defined in this work are

    $$f^{(e)}(\\mathbf{q}) = \\sum\\limits_{i = 1}^5 a_i \\exp(- b_i |\\mathbf{q}|^2),$$

    where $a_i$ is stored as `GaussianMixtureVolume.amplitudes`,
    $b_i / 8 \\pi^2$ are the `GaussianMixtureVolume.variances`, and
    $\\mathbf{q}$ is the scattering vector.

    Under usual scattering approximations (i.e. the first-born approximation),
    the rescaled electrostatic potential energy $U(\\mathbf{r})$ for a given atom type is
    $\\mathcal{F}^{-1}[f^{(e)}(\\boldsymbol{\\xi} / 2)](\\mathbf{r})$, which is computed
    analytically as

    $$U(\\mathbf{r}) = \\sum\\limits_{i = 1}^5 \\frac{a_i}{(2\\pi (b_i / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{r} - \\mathbf{r}'|^2}{2 (b_i / 8 \\pi^2)}),$$

    where $\\mathbf{r}'$ is the position of the atom. Including an additional B-factor (denoted by
    $B$) gives the expression for the potential
    $U(\\mathbf{r})$ of a single atom type and its fourier transform pair $\\tilde{U}(\\boldsymbol{\\xi}) \\equiv \\mathcal{F}[U](\\boldsymbol{\\xi})$,

    $$U(\\mathbf{r}) = \\sum\\limits_{i = 1}^5 \\frac{a_i}{(2\\pi ((b_i + B) / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{r} - \\mathbf{r}'|^2}{2 ((b_i + B) / 8 \\pi^2)}),$$

    $$\\tilde{U}(\\boldsymbol{\\xi}) = \\sum\\limits_{i = 1}^5 a_i \\exp(- (b_i + B) |\\boldsymbol{\\xi}|^2 / 4) \\exp(2 \\pi i \\boldsymbol{\\xi}\\cdot\\mathbf{r}'),$$

    where $\\mathbf{q} = \\boldsymbol{\\xi} / 2$ gives the relationship between the wave vector and the
    scattering vector.

    In practice, for a discretization on a grid with voxel size $\\Delta r$ and grid point $\\mathbf{r}_{\\ell}$,
    the potential is evaluated as the average value inside the voxel

    $$U_{\\ell} = \\frac{1}{\\Delta r^3} \\sum\\limits_{i = 1}^5 a_i \\prod\\limits_{j = 1}^3 \\int_{r^{\\ell}_j-\\Delta r/2}^{r^{\\ell}_j+\\Delta r/2} dr_j \\ \\frac{1}{{\\sqrt{2\\pi ((b_i + B) / 8 \\pi^2)}}} \\exp(- \\frac{(r_j - r'_j)^2}{2 ((b_i + B) / 8 \\pi^2)}),$$

    where $j$ indexes the components of the spatial coordinate vector $\\mathbf{r}$. The above expression is evaluated using the error function as

    $$U_{\\ell} = \\frac{1}{(2 \\Delta r)^3} \\sum\\limits_{i = 1}^5 a_i \\prod\\limits_{j = 1}^3 \\textrm{erf}(\\frac{r_j^{\\ell} - r'_j + \\Delta r / 2}{\\sqrt{2 ((b_i + B) / 8\\pi^2)}}) - \\textrm{erf}(\\frac{r_j^{\\ell} - r'_j - \\Delta r / 2}{\\sqrt{2 ((b_i + B) / 8\\pi^2)}}).$$

    !!! example "Speed up gradients with Pallas"
        ```python
        render_fn = cxs.GaussianMixtureRenderFn(
            shape, voxel_size, n_spread=7, enable_pallas={"bwd": True}
        )
        ```
    """  # noqa: E501

    shape: tuple[int, int, int]
    voxel_size: Float[NDArrayLike, "..."]
    n_batches: int
    n_spread: int | tuple[int, ...] | None
    enable_pallas: bool | Mapping[str, bool] | None

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        n_batches: int = 1,
        n_spread: int | tuple[int, ...] | None = None,
        enable_pallas: bool | Mapping[str, bool] | None = None,
    ):
        """**Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `n_batches`:
            The number of batches over groups of positions used to render
            the voxel grid. By default, `n_batches = 1`, which renders the
            voxel grid for all positions at once. This is useful to decrease
            GPU memory usage. Applies to both the dense (`n_spread=None`) and
            spreading (`n_spread` set) backends.
        - `n_spread`:
            If `None` (default), render the voxel grid with dense gaussian
            integrals evaluated over the whole grid. If an `int`, instead
            directly spread each gaussian onto only the `n_spread` nearest
            grid points (per dimension), trading accuracy for speed.
            If a `tuple` of `int`s (one value per gaussian component, i.e. of
            length `GaussianMixtureVolume.amplitudes.shape[-1]`), spread each
            gaussian component with its own width instead of one shared
            width -- useful when a volume's gaussian components have widths
            spanning an order of magnitude or more (e.g. X-ray/electron
            scattering factors written as a sum of 5 gaussians), where a
            single `n_spread` would either truncate the widest components or
            waste computation spreading the narrowest ones too widely. See
            [`cryojax.simulator.suggest_n_spread`][] to choose these values
            from `volume_representation.variances`.
        - `enable_pallas`:
            Use the Pallas/Triton GPU backend instead of pure-JAX for the
            `n_spread` spreading backend (ignored if `n_spread` is `None`).
            [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) is
            JAX's framework for writing custom GPU/TPU kernels. This is most
            advantageous for the *backward* pass -- `{"bwd": True}`. See
            [`cryojax.ndimage.spread_gaussians_3d`][]'s `enable_pallas` for
            the full picture. `None` (default) defers to `CRYOJAX_ENABLE_PALLAS`.
        """  # noqa: E501
        self.shape = shape
        self.voxel_size = leaf_asarray(voxel_size, dtype=float)
        self.n_batches = n_batches
        self.n_spread = n_spread
        self.enable_pallas = enable_pallas

    @override
    def __call__(
        self,
        volume_representation: GaussianMixtureVolume,
        *,
        outputs_real_space: bool = True,
        outputs_rfft: bool = False,
        fftshifted: bool = False,
    ) -> VoxelArray:
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
        voxel_size = error_if_not_positive(jnp.asarray(self.voxel_size))
        positions = jnp.asarray(volume_representation.positions)
        amplitudes = jnp.asarray(volume_representation.amplitudes)
        variances = error_if_not_positive(jnp.asarray(volume_representation.variances))
        context = (
            f"Error during rendering using `{type(self).__name__}(..., n_batches=...)`"
        )
        if self.n_spread is None:
            real_voxel_grid = _gaussians_to_real_voxels_dense(
                self.shape,
                voxel_size,
                positions,
                amplitudes,
                variances,
                self.n_batches,
                context,
            )
        else:
            real_voxel_grid = _gaussians_to_real_voxels_spread(
                self.shape,
                voxel_size,
                positions,
                amplitudes,
                variances,
                self.n_spread,
                self.n_batches,
                context,
                self.enable_pallas,
            )
        if outputs_real_space:
            return real_voxel_grid
        else:
            if outputs_rfft:
                return (
                    jnp.fft.fftshift(jnp.fft.rfftn(real_voxel_grid), axes=(0, 1))
                    if fftshifted
                    else jnp.fft.rfftn(real_voxel_grid)
                )
            else:
                return (
                    jnp.fft.fftshift(jnp.fft.fftn(real_voxel_grid))
                    if fftshifted
                    else jnp.fft.fftn(real_voxel_grid)
                )


# ============================================================================
# Shared atom-batching
# ============================================================================
#
# Both the dense and spreading backends, for both projection and rendering,
# reduce over atoms (and gaussian components) to produce a single
# image/voxel-grid contribution. `_sum_over_atom_batches` computes that
# reduction in `n_batches` chunks via a summing `filter_bscan` carry, which
# bounds peak memory to one chunk's worth of intermediates rather than all
# atoms at once. `n_batches = 1` (the default) skips batching/`scan` entirely.


def _sum_over_atom_batches(
    kernel_fn,
    xs: PyTree[Array],
    n_batches: int,
    output_shape: tuple[int, ...],
    *,
    context: str,
) -> Array:
    M = jax.tree.leaves(xs)[0].shape[0]
    if n_batches > M:
        raise ValueError(
            f"{context}: `n_batches` must be an integer less than or equal "
            f"to the number of positions, which is equal to {M}. Got "
            f"`n_batches = {n_batches}`."
        )
    if n_batches < 1:
        raise ValueError(
            f"{context}: `n_batches` must be an integer greater than or equal to 1."
        )
    if n_batches == 1:
        return kernel_fn(xs)

    batch_size = M // n_batches

    def f_scan(carry, xs_chunk):
        return carry + kernel_fn(xs_chunk), None

    total, _ = filter_bscan(f_scan, jnp.zeros(output_shape), xs, batch_size=batch_size)
    return total


def _gaussian_weight(r: Array, variance: Array) -> Array:
    """Normalized isotropic Gaussian marginal at physical offset `r`."""
    return jnp.exp(-0.5 * r**2 / variance) / jnp.sqrt(2 * jnp.pi * variance)


def _erf_weight(r: Array, variance: Array, width: Float[Array, ""]) -> Array:
    """Average of `_gaussian_weight` over a pixel/voxel of `width` centered at
    physical offset `r`."""
    scaling = 1.0 / jnp.sqrt(2 * variance)
    left, right = scaling * (r - width / 2), scaling * (r + width / 2)
    return (jsp.special.erf(right) - jsp.special.erf(left)) / (2 * width)


def _dense_axis_values(
    grid: Float[Array, " dim"],
    positions_1d: Float[Array, " M"],
    variance: Float[Array, "M K"],
    width: Float[Array, ""],
    *,
    use_erf: bool,
) -> Float[Array, "dim M K"]:
    """Evaluate the normalized 1D marginal kernel (see `cryojax.simulator
    ._volume.common`) at every grid point, for every position and gaussian
    component."""
    r = grid[:, None, None] - positions_1d[None, :, None]
    variance = variance[None, :, :]
    return _erf_weight(r, variance, width) if use_erf else _gaussian_weight(r, variance)


#
# Projection: dense backend
#
def _gaussians_to_projection_dense(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "M 3"],
    amplitudes: Float[Array, "M K"],
    variances: Float[Array, "M K"],
    use_erf: bool,
    n_batches: int,
    context: str,
) -> Float[Array, "dim_y dim_x"]:
    grid_x = make_1d_coordinate_grid(shape[1], pixel_size)
    grid_y = make_1d_coordinate_grid(shape[0], pixel_size)

    def kernel_fn(xs):
        _positions, _amplitudes, _variances = xs
        return _gaussians_to_projection_dense_kernel(
            grid_x, grid_y, pixel_size, _positions, _amplitudes, _variances, use_erf
        )

    return _sum_over_atom_batches(
        kernel_fn, (positions, amplitudes, variances), n_batches, shape, context=context
    )


def _gaussians_to_projection_dense_kernel(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "M 3"],
    amplitudes: Float[Array, "M K"],
    variances: Float[Array, "M K"],
    use_erf: bool,
) -> Float[Array, "dim_y dim_x"]:
    values_x = amplitudes[None, :, :] * _dense_axis_values(
        grid_x, positions[:, 0], variances, pixel_size, use_erf=use_erf
    )
    values_y = _dense_axis_values(
        grid_y, positions[:, 1], variances, pixel_size, use_erf=use_erf
    )
    return jnp.einsum("ikl, jkl -> ij", values_y, values_x)


#
# Projection: spreading backend
#
def _gaussians_to_projection_spread(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "M 3"],
    amplitudes: Float[Array, "M K"],
    variances: Float[Array, "M K"],
    use_erf: bool,
    n_spread: int | tuple[int, ...],
    n_batches: int,
    context: str,
    enable_pallas: bool | Mapping[str, bool] | None,
) -> Float[Array, "dim_y dim_x"]:
    """Spread each gaussian component directly onto the `n_spread` nearest
    grid points (see `cryojax.simulator._volume.common`), rather than
    evaluating dense gaussian integrals over the whole grid.
    """

    def kernel_fn(xs):
        _positions, _amplitudes, _variances = xs

        def spread_one_gaussian(_amplitude, _variance, _n_spread):
            return spread_gaussians_2d(
                _positions[:, 0],
                _positions[:, 1],
                _amplitude,
                _variance,
                shape,
                pixel_size=pixel_size,
                n_spread=_n_spread,
                use_erf=use_erf,
                enable_pallas=enable_pallas,
            )

        return spread_and_sum_gaussian_components(
            spread_one_gaussian, _amplitudes, _variances, n_spread
        )

    return _sum_over_atom_batches(
        kernel_fn, (positions, amplitudes, variances), n_batches, shape, context=context
    )


#
# Voxel rendering: dense backend
#
def _gaussians_to_real_voxels_dense(
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "M 3"],
    amplitudes: Float[Array, "M K"],
    variances: Float[Array, "M K"],
    n_batches: int,
    context: str,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    z_dim, y_dim, x_dim = shape
    grid_x, grid_y, grid_z = [
        make_1d_coordinate_grid(dim, voxel_size) for dim in [x_dim, y_dim, z_dim]
    ]

    def kernel_fn(xs):
        _positions, _amplitudes, _variances = xs
        return _gaussians_to_real_voxels_dense_kernel(
            grid_x, grid_y, grid_z, voxel_size, _positions, _amplitudes, _variances
        )

    return _sum_over_atom_batches(
        kernel_fn, (positions, amplitudes, variances), n_batches, shape, context=context
    )


def _gaussians_to_real_voxels_dense_kernel(
    grid_x: Float[Array, " dim_x"],
    grid_y: Float[Array, " dim_y"],
    grid_z: Float[Array, " dim_z"],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "M 3"],
    amplitudes: Float[Array, "M K"],
    variances: Float[Array, "M K"],
) -> Float[Array, "dim_z dim_y dim_x"]:
    values_x = amplitudes[None, :, :] * _dense_axis_values(
        grid_x, positions[:, 0], variances, voxel_size, use_erf=True
    )
    values_y = _dense_axis_values(
        grid_y, positions[:, 1], variances, voxel_size, use_erf=True
    )
    values_z = _dense_axis_values(
        grid_z, positions[:, 2], variances, voxel_size, use_erf=True
    )
    # Compute one z-plane at a time to avoid materializing the full 3D
    # correlation tensor at once.
    render_at_z_plane = lambda values_z_row: jnp.einsum(
        "ikl, jkl -> ij", values_y * values_z_row[None, :, :], values_x
    )
    return jax.lax.map(render_at_z_plane, values_z)


#
# Voxel rendering: spreading backend
#
def _gaussians_to_real_voxels_spread(
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "M 3"],
    amplitudes: Float[Array, "M K"],
    variances: Float[Array, "M K"],
    n_spread: int | tuple[int, ...],
    n_batches: int,
    context: str,
    enable_pallas: bool | Mapping[str, bool] | None,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    """Spread each gaussian component directly onto the `n_spread` nearest
    grid points (see `cryojax.simulator._volume.common`), rather than
    evaluating dense gaussian integrals over the whole grid.
    """

    def kernel_fn(xs):
        _positions, _amplitudes, _variances = xs

        def spread_one_gaussian(_amplitude, _variance, _n_spread):
            return spread_gaussians_3d(
                _positions[:, 0],
                _positions[:, 1],
                _positions[:, 2],
                _amplitude,
                _variance,
                shape,
                voxel_size=voxel_size,
                n_spread=_n_spread,
                use_erf=True,
                enable_pallas=enable_pallas,
            )

        return spread_and_sum_gaussian_components(
            spread_one_gaussian, _amplitudes, _variances, n_spread
        )

    return _sum_over_atom_batches(
        kernel_fn, (positions, amplitudes, variances), n_batches, shape, context=context
    )
