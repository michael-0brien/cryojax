from typing import ClassVar, Literal, Self
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from ..._internal import error_if_not_positive
from ...constants import PengScatteringFactorParameters, b_factor_to_variance
from ...jax_util import FloatLike, NDArrayLike, filter_bscan
from ...ndimage import fftn, make_1d_coordinate_grid, resize_with_crop_or_pad, rfftn
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose
from .base_volume import (
    AbstractAtomVolume,
    AbstractVolumeIntegrator,
    AbstractVolumeRenderFn,
    ProjectionArray,
    VoxelArray,
)
from .common import (
    _erf_shape_and_grad,
    _gaussian_shape_and_grad,
    eps_to_nspread,
    normalize_positions_to_grid,
    spread_2d as _grid_spread_2d,
    spread_3d as _grid_spread_3d,
)


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

    positions: Float[Array, "n_positions 3"]
    amplitudes: Float[Array, "n_positions n_gaussians"]
    variances: Float[Array, " n_positions n_gaussians"]

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        positions: Float[NDArrayLike, "n_positions 3"],
        amplitudes: (
            float
            | Float[NDArrayLike, ""]
            | Float[NDArrayLike, " n_positions"]
            | Float[NDArrayLike, "n_positions n_gaussians"]
        ),
        variances: (
            float
            | Float[NDArrayLike, ""]
            | Float[NDArrayLike, " n_positions"]
            | Float[NDArrayLike, "n_positions n_gaussians"]
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
        n_positions = positions.shape[0]
        if isinstance(amplitudes, NDArrayLike):
            if amplitudes.ndim == 2:
                n_gaussians = amplitudes.shape[-1]
            elif amplitudes.ndim == 1:
                n_gaussians = 1
                amplitudes = amplitudes[:, None]
            elif amplitudes.ndim == 0:
                n_gaussians = 1
                amplitudes = amplitudes[None, None]
            else:
                raise ValueError(
                    "Passed `amplitudes` to `GaussianMixtureVolume` "
                    f"with shape {amplitudes.shape}, but must be of "
                    "shape `()`, `(n_positions,)`, or "
                    "`(n_positions, n_gaussians)`."
                )
        else:
            n_gaussians = 1
        if isinstance(variances, NDArrayLike):
            if variances.ndim == 2:
                n_gaussians = variances.shape[-1]
            elif variances.ndim == 1:
                variances = variances[:, None]
            elif variances.ndim == 0:
                variances = variances[None, None]
            else:
                raise ValueError(
                    "Passed `variances` to `GaussianMixtureVolume` "
                    f"with shape {variances.shape}, but must be of "
                    "shape `()`, `(n_positions,)`, or "
                    "`(n_positions, n_gaussians)`."
                )

        self.positions = jnp.asarray(positions, dtype=float)
        self.amplitudes = jnp.broadcast_to(
            jnp.asarray(amplitudes, dtype=float), (n_positions, n_gaussians)
        )
        self.variances = jnp.broadcast_to(
            jnp.asarray(variances, dtype=float), (n_positions, n_gaussians)
        )

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
        return eqx.tree_at(
            lambda d: d.positions,
            self,
            pose.rotate_coordinates(self.positions, inverse=self.is_frame_rotation),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        offset_in_angstroms = pose.offset_in_angstroms
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        return eqx.tree_at(
            lambda d: d.positions, self, self.positions + offset_in_angstroms
        )


class GaussianMixtureProjection(
    AbstractVolumeIntegrator[GaussianMixtureVolume],
    strict=True,
):
    sampling_mode: Literal["average", "point"]
    shape: tuple[int, int] | None
    n_batches: int
    eps: float | None

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        upsampling_factor: int | None = None,
        shape: tuple[int, int] | None = None,
        sampling_mode: Literal["average", "point"] = "average",
        n_batches: int = 1,
        eps: float | None = None,
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
            the dense (`eps=None`) and spreading (`eps` set) backends.
        - `eps`:
            If `None` (default), compute the projection with dense gaussian
            integrals evaluated over the whole grid. If a `float`, instead
            directly spread each gaussian onto only the `nspread` nearest grid
            points (chosen from `eps`, trading accuracy for speed), using the
            same backend as [`cryojax.simulator.IndependentAtomProjection`][].
        """  # noqa: E501
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
        self.eps = eps

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
        pixel_size = image_config.pixel_size
        # Grab the gaussian amplitudes and widths
        positions = volume_representation.positions
        amplitudes = volume_representation.amplitudes
        variances = error_if_not_positive(volume_representation.variances)
        use_erf = self.sampling_mode == "average"
        context = (
            f"Error during projection using `{type(self).__name__}(..., n_batches=...)`"
        )
        # Compute the projection
        if self.eps is None:
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
                self.eps,
                self.n_batches,
                context,
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
    """  # noqa: E501

    shape: tuple[int, int, int]
    voxel_size: Float[Array, ""]
    n_batches: int
    eps: float | None

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        n_batches: int = 1,
        eps: float | None = None,
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
            GPU memory usage. Applies to both the dense (`eps=None`) and
            spreading (`eps` set) backends.
        - `eps`:
            If `None` (default), render the voxel grid with dense gaussian
            integrals evaluated over the whole grid. If a `float`, instead
            directly spread each gaussian onto only the `nspread` nearest grid
            points (chosen from `eps`, trading accuracy for speed), using the
            same backend as [`cryojax.simulator.IndependentAtomRenderFn`][].
        """  # noqa: E501
        self.shape = shape
        self.voxel_size = jnp.asarray(voxel_size, dtype=float)
        self.n_batches = n_batches
        self.eps = eps

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
        voxel_size = error_if_not_positive(self.voxel_size)
        variances = error_if_not_positive(volume_representation.variances)
        context = (
            f"Error during rendering using `{type(self).__name__}(..., n_batches=...)`"
        )
        if self.eps is None:
            real_voxel_grid = _gaussians_to_real_voxels_dense(
                self.shape,
                voxel_size,
                volume_representation.positions,
                volume_representation.amplitudes,
                variances,
                self.n_batches,
                context,
            )
        else:
            real_voxel_grid = _gaussians_to_real_voxels_spread(
                self.shape,
                voxel_size,
                volume_representation.positions,
                volume_representation.amplitudes,
                variances,
                self.eps,
                self.n_batches,
                context,
            )
        if outputs_real_space:
            return real_voxel_grid
        else:
            if outputs_rfft:
                return (
                    jnp.fft.fftshift(rfftn(real_voxel_grid), axes=(0, 1))
                    if fftshifted
                    else rfftn(real_voxel_grid)
                )
            else:
                return (
                    jnp.fft.fftshift(fftn(real_voxel_grid))
                    if fftshifted
                    else fftn(real_voxel_grid)
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
    n_positions = jax.tree.leaves(xs)[0].shape[0]
    if n_batches > n_positions:
        raise ValueError(
            f"{context}: `n_batches` must be an integer less than or equal "
            f"to the number of positions, which is equal to {n_positions}. Got "
            f"`n_batches = {n_batches}`."
        )
    if n_batches < 1:
        raise ValueError(
            f"{context}: `n_batches` must be an integer greater than or equal to 1."
        )
    if n_batches == 1:
        return kernel_fn(xs)

    batch_size = n_positions // n_batches

    def f_scan(carry, xs_chunk):
        return carry + kernel_fn(xs_chunk), None

    total, _ = filter_bscan(f_scan, jnp.zeros(output_shape), xs, batch_size=batch_size)
    return total


def _dense_axis_values(
    grid: Float[Array, " dim"],
    positions_1d: Float[Array, " n_positions"],
    variance: Float[Array, "n_positions n_gaussians"],
    width: Float[Array, ""],
    *,
    use_erf: bool,
) -> Float[Array, "dim n_positions n_gaussians"]:
    """Evaluate the normalized 1D marginal kernel (see `cryojax.simulator
    ._volume.common`) at every grid point, for every position and gaussian
    component."""
    r = grid[:, None, None] - positions_1d[None, :, None]
    variance = variance[None, :, :]
    values, _ = (
        _erf_shape_and_grad(r, variance, width)
        if use_erf
        else _gaussian_shape_and_grad(r, variance)
    )
    return values


#
# Projection: dense backend
#
def _gaussians_to_projection_dense(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians"],
    variances: Float[Array, "n_positions n_gaussians"],
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
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians"],
    variances: Float[Array, "n_positions n_gaussians"],
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
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians"],
    variances: Float[Array, "n_positions n_gaussians"],
    use_erf: bool,
    eps: float,
    n_batches: int,
    context: str,
) -> Float[Array, "dim_y dim_x"]:
    """Spread each gaussian component directly onto the `nspread` nearest
    grid points (see `cryojax.simulator._volume.common`), rather than
    evaluating dense gaussian integrals over the whole grid.
    """
    nspread = eps_to_nspread(eps)
    xy = normalize_positions_to_grid(positions[:, :2], shape, pixel_size)

    def kernel_fn(xs):
        _xy, _amplitudes, _variances = xs

        def spread_one_gaussian(_amplitude, _variance):
            return _grid_spread_2d(
                _xy[:, 0],
                _xy[:, 1],
                _amplitude,
                shape,
                variance=_variance,
                pixel_size=pixel_size,
                nspread=nspread,
                use_erf=use_erf,
            )

        contributions = jax.vmap(spread_one_gaussian, in_axes=(1, 1))(
            _amplitudes, _variances
        )
        return jnp.sum(contributions, axis=0)

    return _sum_over_atom_batches(
        kernel_fn, (xy, amplitudes, variances), n_batches, shape, context=context
    )


#
# Voxel rendering: dense backend
#
def _gaussians_to_real_voxels_dense(
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians"],
    variances: Float[Array, "n_positions n_gaussians"],
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
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians"],
    variances: Float[Array, "n_positions n_gaussians"],
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
    positions: Float[Array, "n_positions 3"],
    amplitudes: Float[Array, "n_positions n_gaussians"],
    variances: Float[Array, "n_positions n_gaussians"],
    eps: float,
    n_batches: int,
    context: str,
) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
    """Spread each gaussian component directly onto the `nspread` nearest
    grid points (see `cryojax.simulator._volume.common`), rather than
    evaluating dense gaussian integrals over the whole grid.
    """
    nspread = eps_to_nspread(eps)
    xyz = normalize_positions_to_grid(positions, shape, voxel_size)

    def kernel_fn(xs):
        _xyz, _amplitudes, _variances = xs

        def spread_one_gaussian(_amplitude, _variance):
            return _grid_spread_3d(
                _xyz[:, 0],
                _xyz[:, 1],
                _xyz[:, 2],
                _amplitude,
                shape,
                variance=_variance,
                voxel_size=voxel_size,
                nspread=nspread,
                use_erf=True,
            )

        contributions = jax.vmap(spread_one_gaussian, in_axes=(1, 1))(
            _amplitudes, _variances
        )
        return jnp.sum(contributions, axis=0)

    return _sum_over_atom_batches(
        kernel_fn, (xyz, amplitudes, variances), n_batches, shape, context=context
    )
