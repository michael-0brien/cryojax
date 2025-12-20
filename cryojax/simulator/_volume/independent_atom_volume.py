from collections.abc import Sequence
from typing import Any, ClassVar, Literal, TypeVar
from typing_extensions import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from ...constants import LobatoScatteringFactorParameters, PengScatteringFactorParameters
from ...jax_util import FloatLike, NDArrayLike, error_if_not_positive
from ...ndimage import (
    AbstractFourierOperator,
    FourierSinc,
    block_reduce_downsample,
    convert_fftn_to_rfftn,
    ifftn,
    irfftn,
    make_frequency_grid,
    resize_with_crop_or_pad,
    rfftn,
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


try:
    import jax_finufft as jnufft

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jnufft = None
    JAX_FINUFFT_IMPORT_ERROR = err


T = TypeVar("T")


class PengScatteringFactor(AbstractFourierOperator, strict=True):
    a: Float[Array, " n"]
    b: Float[Array, " n"]
    b_factor: Float[Array, ""] | None

    def __init__(
        self,
        a: Float[NDArrayLike, " n"],
        b: Float[NDArrayLike, " n"],
        b_factor: FloatLike | None = None,
    ):
        self.a = jnp.asarray(a, dtype=float)
        self.b = jnp.asarray(b, dtype=float)
        self.b_factor = None if b_factor is None else jnp.asarray(b_factor, dtype=float)

    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ):
        q_squared = jnp.sum(frequency_grid**2, axis=-1)
        b_factor = 0.0 if self.b_factor is None else self.b_factor
        gaussian_fn = lambda _a, _b: _a * jnp.exp(-0.25 * (_b + b_factor) * q_squared)
        return jnp.sum(jax.vmap(gaussian_fn)(self.a, self.b), axis=0)


class LobatoScatteringFactor(AbstractFourierOperator, strict=True):
    a: Float[Array, " n"]
    b: Float[Array, " n"]
    b_factor: Float[Array, ""] | None

    def __init__(
        self,
        a: Float[NDArrayLike, " n"],
        b: Float[NDArrayLike, " n"],
        b_factor: FloatLike | None = None,
    ):
        self.a = jnp.asarray(a, dtype=float)
        self.b = jnp.asarray(b, dtype=float)
        self.b_factor = None if b_factor is None else jnp.asarray(b_factor, dtype=float)

    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ):
        q_squared = jnp.sum(frequency_grid**2, axis=-1)
        hydrogenic_fn = (
            lambda _a, _b: _a * (2 + _b * q_squared) / (1 + _b * q_squared) ** 2
        )
        scattering_factor = jnp.sum(jax.vmap(hydrogenic_fn)(self.a, self.b), axis=0)
        if self.b_factor is not None:
            scattering_factor *= jnp.exp(-0.25 * self.b_factor * q_squared)
        return scattering_factor


class IndependentAtomVolume(AbstractAtomVolume, strict=True):
    positions: PyTree[Float[Array, "_ 3"]]
    scattering_factors: PyTree[AbstractFourierOperator]

    def __init__(
        self,
        positions: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        scattering_factors: PyTree[AbstractFourierOperator, "T"],
    ):
        """**Arguments:**

        - `positions`:
            A pytree of atom positions.
        - `scattering_factors`:
            A pytree of scattering factors with the same tree structure
            as `positions`, where each leaf is a
            [`cryojax.ndimage.AbstractFourierOperator`][].
        """
        if jax.tree.structure(positions) != jax.tree.structure(
            scattering_factors,
            is_leaf=lambda x: isinstance(x, AbstractFourierOperator),
        ):
            raise ValueError(
                "When instantiating an `IndependentAtomVolume`, found "
                "that the pytree structures of `positions` and "
                "`scattering_factors` were not equal."
            )
        self.positions = jax.tree.map(lambda x: jnp.asarray(x, dtype=float), positions)
        self.scattering_factors = scattering_factors

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new potential with rotated `positions`."""
        rotate_fn = lambda pos: pose.rotate_coordinates(pos, inverse=inverse)
        return eqx.tree_at(
            lambda x: x.positions,
            self,
            jax.tree.map(rotate_fn, self.positions),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        offset_in_angstroms = pose.offset_in_angstroms
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        translate_fn = lambda pos: pos + offset_in_angstroms
        return eqx.tree_at(
            lambda x: x.positions,
            self,
            jax.tree.map(translate_fn, self.positions),
        )

    @classmethod
    def from_tabulated_parameters(
        cls,
        positions_by_element: tuple[Float[NDArrayLike, "_ 3"], ...],
        parameters: PengScatteringFactorParameters | LobatoScatteringFactorParameters,
        *,
        b_factor_by_element: FloatLike | tuple[FloatLike, ...] | None = None,
    ) -> Self:
        def make_scattering_factor(a, b, b_factor):
            if isinstance(parameters, PengScatteringFactorParameters):
                return PengScatteringFactor(a, b, b_factor)
            elif isinstance(parameters, LobatoScatteringFactorParameters):
                return LobatoScatteringFactor(a, b, b_factor)
            else:
                raise ValueError(
                    "Unrecognized argument `parameters` when "
                    "calling `IndependentAtomVolume.from_tabulated_parameters`. "
                    "Should be either `cryojax.constants.PengScatteringFactorParameters` "
                    "or `cryojax.constants.LobatoScatteringFactorParameters`, but got "
                    f"type {parameters.__class__.__name__}."
                )

        n_elements = len(positions_by_element)
        a, b = parameters.a, parameters.b
        if a.shape[0] != n_elements or b.shape[0] != n_elements:
            raise ValueError(
                "When constructing an `IndependentAtomVolume` via "
                "`from_tabulated_parameters`, found that "
                "`parameters.a.shape[0] != len(positions_by_element)` "
                "or `parameters.b.shape[0] != len(positions_by_element)`. "
                "Make sure that `a` and `b` correspond to the element types "
                "in `positions_by_element.`"
            )
        if b_factor_by_element is not None:
            if isinstance(b_factor_by_element, Sequence):
                if len(b_factor_by_element) != n_elements:
                    raise ValueError(
                        "When constructing an `IndependentAtomVolume` via "
                        "`from_tabulated_parameters`, found that "
                        "`len(b_factor_by_element) != len(positions_by_element)`. "
                        "Make sure that `b_factor_by_element` is a tuple with "
                        "length matching the number of atom types."
                    )
            else:
                b_factor_by_element = tuple(
                    b_factor_by_element for _ in range(n_elements)
                )
            scattering_factors_by_element = tuple(
                make_scattering_factor(a_i, b_i, b_factor)
                for a_i, b_i, b_factor in zip(
                    parameters.a, parameters.b, b_factor_by_element
                )
            )
        else:
            scattering_factors_by_element = tuple(
                make_scattering_factor(a_i, b_i, b_factor=None)
                for a_i, b_i in zip(parameters.a, parameters.b)
            )
        return cls(positions_by_element, scattering_factors_by_element)


class FFTAtomRenderFn(AbstractVolumeRenderFn[IndependentAtomVolume], strict=True):
    """Render a voxel grid using non-uniform FFTs and convolution."""

    shape: tuple[int, int, int]
    voxel_size: Float[Array, ""]
    frequency_grid: Float[Array, "_ _ _ 3"] | None
    sampling_mode: Literal["average", "point"]
    eps: float
    opts: Any

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        frequency_grid: Float[Array, "_ _ _ 3"] | None = None,
        sampling_mode: Literal["average", "point"] = "average",
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
                make_frequency_grid(shape, voxel_size, outputs_rfft=False),
                axes=(0, 1, 2),
            )
            ```

        - `sampling_mode`:
            If `'average'`, convolve with a box function to sample the
            projected volume at a pixel to be the average value of the
            underlying continuous function. If `'point'`, the volume at
            a pixel will be point sampled.
        - `eps`:
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        - `opts`:
            A `jax_finufft.options.Opts` or `jax_finufft.options.NestedOpts`
            dataclass.
            See [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft)
            for documentation.
        """
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `FFTAtomRenderFn` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        self.shape = shape
        self.voxel_size = error_if_not_positive(jnp.asarray(voxel_size, dtype=float))
        self.frequency_grid = frequency_grid
        self.sampling_mode = sampling_mode
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
        if self.frequency_grid is None:
            frequency_grid = jnp.fft.fftshift(
                make_frequency_grid(self.shape, self.voxel_size, outputs_rfftfreqs=False),
                axes=(0, 1, 2),
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
                volume_representation.positions,
                volume_representation.scattering_factors,
                is_leaf=lambda x: isinstance(x, AbstractFourierOperator),
            ),
        )
        if self.sampling_mode == "average":
            antialias_fn = FourierSinc(box_width=self.voxel_size)
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


class FFTAtomProjection(
    AbstractVolumeIntegrator[IndependentAtomVolume],
    strict=True,
):
    """Integrate atomic parametrization of a volume onto
    the exit plane using non-uniform FFTs plus convolution.
    """

    sampling_mode: Literal["average", "point"]
    upsample_factor: int | None
    shape: tuple[int, int] | None
    eps: float
    opts: Any

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        sampling_mode: Literal["average", "point"] = "average",
        upsample_factor: int | None = None,
        shape: tuple[int, int] | None = None,
        eps: float = 1e-6,
        opts: Any = None,
    ):
        """**Arguments:**

        - `sampling_mode`:
            If `'average'`, convolve with a box function to sample the
            projected volume at a pixel to be the average value of the
            underlying continuous function. If `'point'`, the volume at
            a pixel will be point sampled.
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
        if jnufft is None:
            raise RuntimeError(
                "Tried to use the `FFTAtomProjection` "
                "class, but `jax-finufft` is not installed. "
                "See https://github.com/flatironinstitute/jax-finufft "
                "for installation instructions."
            ) from JAX_FINUFFT_IMPORT_ERROR
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `FFTAtomProjection` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        self.sampling_mode = sampling_mode
        self.upsample_factor = upsample_factor
        self.shape = shape
        self.eps = eps
        self.opts = opts

    @override
    def integrate(
        self,
        volume_representation: IndependentAtomVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionArray:
        """Compute a projection from scattering factors per atom type
        from the `IndependentAtomVolume`.

        **Arguments:**

        - `volume_representation`:
            The volume representation.
        - `image_config`:
            The configuration of the resulting image.
        - `outputs_real_space`:
            If `True`, return the image in real space. Otherwise,
            return in fourier.

        **Returns:**

        The volume projection in real or Fourier space at the
        `AbstractImageConfig.padded_shape` and the `image_config.pixel_size`.
        """  # noqa: E501
        u = self.upsample_factor
        pixel_size = image_config.pixel_size
        shape = image_config.padded_shape if self.shape is None else self.shape
        if u is None:
            shape_u, pixel_size_u = shape, pixel_size
        else:
            shape_u, pixel_size_u = (u * shape[0], u * shape[1]), pixel_size / u
        if shape_u == image_config.padded_shape:
            frequency_grid = image_config.padded_full_frequency_grid_in_angstroms
        else:
            frequency_grid = make_frequency_grid(
                shape_u, pixel_size_u, outputs_rfftfreqs=False
            )
        frequency_grid = jnp.fft.fftshift(frequency_grid, axes=(0, 1))
        proj_kernel = lambda pos, kernel: _project_with_nufft(
            shape_u,
            pixel_size_u,
            pos,
            kernel,
            frequency_grid,
            eps=self.eps,
            opts=self.opts,
        )
        # Compute projection over atom types
        fourier_projection = jax.tree.reduce(
            lambda x, y: x + y,
            jax.tree.map(
                proj_kernel,
                volume_representation.positions,
                volume_representation.scattering_factors,
                is_leaf=lambda x: isinstance(x, AbstractFourierOperator),
            ),
        )
        # Apply anti-aliasing filter
        if self.sampling_mode == "average":
            antialias_fn = FourierSinc(box_width=pixel_size_u)
            fourier_projection *= antialias_fn(frequency_grid)
        # If upsample factor is even, need subpixel center correction
        if u is not None and u % 2 == 0:
            fourier_projection = _apply_subpixel_shift(
                shape,
                fourier_projection,
                pixel_size_u * frequency_grid,
                u,
            )
        # Shift zero frequency component to corner and convert to
        # rfft
        fourier_projection = convert_fftn_to_rfftn(
            jnp.fft.ifftshift(fourier_projection), mode="real"
        )
        if self.shape is None:
            if u is None:
                return (
                    irfftn(fourier_projection, s=shape)
                    if outputs_real_space
                    else fourier_projection
                )
            else:
                projection = _block_average(irfftn(fourier_projection, s=shape_u), u)
                return projection if outputs_real_space else rfftn(projection)
        else:
            projection = irfftn(fourier_projection, s=shape_u)
            if u is not None:
                projection = _block_average(projection, u)
            projection = resize_with_crop_or_pad(projection, image_config.padded_shape)
            return projection if outputs_real_space else rfftn(projection)


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


def _block_average(x, factor):
    return (
        block_reduce_downsample(x, factor, jax.lax.add, center_correct=False)
        / factor**x.ndim
    )


def _apply_subpixel_shift(target_shape, fourier_image, frequency_grid, k):
    if len(set(target_shape)) > 1:
        raise NotImplementedError(
            "Even `upsample_factor` and non-square image shape not "
            f"supported in `FFTAtomProjection`. Got `upsample_factor = {k}` "
            f"and `shape = {target_shape}`."
        )
    dim = target_shape[0]
    if dim % 2 == 0:
        shift = jnp.full((2,), (k - 1) / 2)
    else:
        shift = jnp.full((2,), -0.5)
    return (
        jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid, shift))) * fourier_image
    )
