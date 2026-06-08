import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal, Self, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import nufftax
from jaxtyping import Array, Float, Inexact, PyTree
from nufftax.core import Kernel, spread_2d, spread_3d

from ..._internal import error_if_not_positive
from ...constants import (
    LobatoScatteringFactorParameters,
    PengScatteringFactorParameters,
)
from ...jax_util import FloatLike, NDArrayLike
from ...ndimage import (
    AbstractFourierOperator,
    AbstractRealOperator,
    FourierSinc,
    RealGaussian,
    block_reduce_downsample,
    convert_fftn_to_rfftn,
    fftn,
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
    import jax_finufft
    from jax_finufft.options import NestedOpts, Opts

    JAX_FINUFFT_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    jax_finufft, Opts, NestedOpts = None, None, None
    JAX_FINUFFT_IMPORT_ERROR = err


T = TypeVar("T")


class PengScatteringFactor(AbstractFourierOperator, strict=True):
    a: Float[Array, " n"]
    b: Float[Array, " n"]
    b_factor: Float[Array, ""] | None = None

    spatial_dims: ClassVar[list[int]] = [2, 3]

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
        b_factor = 0.0 if self.b_factor is None else error_if_not_positive(self.b_factor)
        gaussian_fn = lambda _a, _b: _a * jnp.exp(-0.25 * (_b + b_factor) * q_squared)
        return jnp.sum(
            jax.vmap(gaussian_fn)(self.a, error_if_not_positive(self.b)), axis=0
        )


class LobatoScatteringFactor(AbstractFourierOperator, strict=True):
    a: Float[Array, " n"]
    b: Float[Array, " n"]
    b_factor: Float[Array, ""] | None = None

    spatial_dims: ClassVar[list[int]] = [2, 3]

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
        hydrogenic_fn = lambda _a, _b: (
            _a * (2 + _b * q_squared) / (1 + _b * q_squared) ** 2
        )
        scattering_factor = jnp.sum(jax.vmap(hydrogenic_fn)(self.a, self.b), axis=0)
        if self.b_factor is not None:
            scattering_factor *= jnp.exp(
                -0.25 * error_if_not_positive(self.b_factor) * q_squared
            )
        return scattering_factor


class IndependentAtomVolume(AbstractAtomVolume, strict=True):
    """A representation of a volume that accepts an array of
    atom positions and an electron scattering factor for these
    atoms.

    !!! example "A Gaussian at each atom"
        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage as im

        positions = ... # load atom positions
        b_factor = ...  # ... and a B-factor
        volume = cxs.IndependentAtomVolume(
            positions=positions, kernel_fns=im.FourierGaussian(b_factor=b_factor)
        )
        ```

    The arguments `positions` and `kernel_fns` may also be
    pytrees of arrays and scattering factors, where each tree leaf represents
    a different atom type.

    !!! example "Multiple atom types"
        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage as im

        positions_1, positions_2 = ...
        b_factor_1, b_factor_2 = ...
        volume = cxs.IndependentAtomVolume(
            positions=(positions_1, positions_2),
            kernel_fns=(im.FourierGaussian(b_factor=b_factor_1), im.FourierGaussian(b_factor=b_factor_2))
        )
        ```

    See [`cryojax.simulator.IndependentAtomVolume.from_tabulated_parameters`][] for
    loading a volume from tabulated electron scattering factors.
    """  # noqa: E501

    positions: PyTree[Float[Array, "_ 3"]]
    kernel_fns: PyTree[AbstractFourierOperator] | PyTree[AbstractRealOperator]

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        positions: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        kernel_fns: (
            PyTree[AbstractFourierOperator, "T"] | PyTree[AbstractRealOperator, "T"]
        ),
    ):
        """**Arguments:**

        - `positions`:
            A pytree of atom positions.
        - `kernel_fns`:
            A pytree of functions with the same tree structure
            as `positions`, where each leaf is a
            [`cryojax.ndimage.AbstractRealOperator`][] or a
            [`cryojax.ndimage.AbstractFourierOperator`][].
            Real-space represents the scattering potential, while
            fourier-space represents the scattering factor.
        """
        if jax.tree.structure(positions) != jax.tree.structure(
            kernel_fns,
            is_leaf=lambda x: isinstance(
                x, (AbstractFourierOperator, AbstractRealOperator)
            ),
        ):
            raise ValueError(
                "When instantiating an `IndependentAtomVolume`, found "
                "that the pytree structures of `positions` and "
                "`kernel_fns` were not equal."
            )
        self.positions = jax.tree.map(lambda x: jnp.asarray(x, dtype=float), positions)
        self.kernel_fns = kernel_fns

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        rotate_fn = lambda pos: pose.rotate_coordinates(
            pos, inverse=self.is_frame_rotation
        )
        return eqx.tree_at(
            lambda x: x.positions,
            self,
            jax.tree.map(rotate_fn, self.positions),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with translated `positions`."""
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
        def make_kernel_fn(a, b, b_factor):
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
            kernel_fns_by_element = tuple(
                make_kernel_fn(a_i, b_i, b_factor)
                for a_i, b_i, b_factor in zip(
                    parameters.a, parameters.b, b_factor_by_element
                )
            )
        else:
            kernel_fns_by_element = tuple(
                make_kernel_fn(a_i, b_i, b_factor=None)
                for a_i, b_i in zip(parameters.a, parameters.b)
            )
        return cls(positions_by_element, kernel_fns_by_element)


_REAL_VS_FOURIER_DOC = """The underlying algorithm depends on if
`IndependentAtomVolume.kernel_fns` are in real-space or
fourier-space via the [`cryojax.ndimage.AbstractFourierOperator`][]
or [`cryojax.ndimage.AbstractRealOperator`][] classes.

- If [`cryojax.ndimage.AbstractFourierOperator`][]:
    Use non-uniform FFTs and convolution. This is good
    when kernels span at least a couple pixels.
- If [`cryojax.ndimage.AbstractRealOperator`][]:
    Directly spread atoms into a volume. This should be
    preferred in most cases.
"""


class IndependentAtomRenderFn(AbstractVolumeRenderFn[IndependentAtomVolume], strict=True):
    shape: tuple[int, int, int]
    voxel_size: Float[Array, ""]

    backend: Literal["nufftax", "jax-finufft"]
    sampling_mode: Literal["average", "point"]
    upsample_factor: float
    eps: float
    options: dict[str, Any]

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        backend: Literal["nufftax", "jax-finufft"] = "nufftax",
        sampling_mode: Literal["average", "point"] = "average",
        upsample_factor: int | float = 2.0,
        eps: float = 1e-6,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `sampling_mode`:
            If `'average'`, convolve with a box function to sample the
            projected volume at a pixel to be the average value of the
            underlying continuous function. If `'point'`, the volume at
            a pixel will be point sampled.
        - `backend`:
            The backend for non-uniform FFT computation. This is either
            [`nufftax`](https://github.com/GragasLab/nufftax/tree/custom-kernel-spread)
            for a pure-JAX implementation of the
            [`finufft`](https://finufft.readthedocs.io) algorithm,
            or [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft) for
            calling `finufft` directly via `jax.ffi`.
            Used only when `IndependentAtomVolume.kernel_fns` are type
            `AbstractFourierOperator`.
        - `upsample_factor`:
            How much to upsample the grid on which atoms are spread onto.
            See [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)
            for documentation. If a 2-tuple is passed, `upsample_factor[0]` is
            used for the forward pass and `upsample_factor[1]` is used for the
            backward pass.
        - `eps`:
            Controls speed / accuracy tradeoff.
            See [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)
            for documentation.
        - `options`:
            A dictionary of options for advanced usage. This is passed directly to the underlying
            non-uniform FFT implementation if kernels are in fourier-space, or to the `nufftax`
            spreading function if kernels are in real-space.
        """  # noqa: E501
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `IndependentAtomRenderFn` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        if backend not in ["jax-finufft", "nufftax"]:
            raise ValueError(
                "`backend` in `IndependentAtomRenderFn` "
                "must be either 'jax-finufft' or 'nufftax'. Got "
                f"`backend = {backend}`."
            )
        self.shape = shape
        self.voxel_size = jnp.asarray(voxel_size, dtype=float)
        self.backend = backend
        self.sampling_mode = sampling_mode
        self.upsample_factor = float(upsample_factor)
        self.eps = eps
        self.options = options

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
        frequency_grid = jnp.fft.fftshift(
            make_frequency_grid(self.shape, self.voxel_size, outputs_rfftfreqs=False),
            axes=(0, 1, 2),
        )
        return render_impl(
            volume_representation.positions,
            volume_representation.kernel_fns,
            shape=self.shape,
            voxel_size=error_if_not_positive(self.voxel_size),
            backend=self.backend,
            frequency_grid=frequency_grid,
            sampling_mode=self.sampling_mode,
            eps=self.eps,
            upsampfac=self.upsample_factor,
            outputs_real_space=outputs_real_space,
            outputs_rfft=outputs_rfft,
            fftshifted=fftshifted,
            options=self.options,
        )


IndependentAtomRenderFn.__doc__ = (
    f"""Render a voxel grid from an `IndependentAtomVolume`. {_REAL_VS_FOURIER_DOC}"""
)


class IndependentAtomProjection(
    AbstractVolumeIntegrator[IndependentAtomVolume],
    strict=True,
):
    backend: Literal["nufftax", "jax-finufft"]
    sampling_mode: Literal["average", "point"]
    upsample_factor: float
    block_size: int
    eps: float
    shape: tuple[int, int] | None
    options: dict[str, Any]

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        backend: Literal["jax-finufft", "nufftax"] = "nufftax",
        sampling_mode: Literal["average", "point"] = "average",
        block_size: int = 1,
        upsample_factor: int | float = 2.0,
        eps: float = 1e-6,
        shape: tuple[int, int] | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `backend`:
            The backend for non-uniform FFT computation. This is either
            [`nufftax`](https://github.com/GragasLab/nufftax/tree/custom-kernel-spread)
            for a pure-JAX implementation of the
            [`finufft`](https://finufft.readthedocs.io) algorithm,
            or [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft) for
            calling `finufft` directly via `jax.ffi`.
            Used only when `IndependentAtomVolume.kernel_fns` are type
            `AbstractFourierOperator`.
        - `sampling_mode`:
            If `'average'`, convolve with a box function to sample the
            projected volume at a pixel to be the average value of the
            underlying continuous function. If `'point'`, the volume at
            a pixel will be point sampled.
        - `block_size`:
            If provided, first compute an image at `voxel_size / block_size`.
            Then, call [`cryojax.ndimage.block_reduce_downsample`][]
            to locally average to the correct pixel size. This is useful
            for reducing aliasing.
        - `upsample_factor`:
            How much to upsample the grid on which atoms are spread onto.
            See [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)
            for documentation. If a 2-tuple is passed, `upsample_factor[0]` is
            used for the forward pass and `upsample_factor[1]` is used for the
            backward pass.
        - `eps`:
            Controls speed / accuracy tradeoff.
            See [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)
            for documentation.
        - `shape`:
            If given, first compute the image at `shape`, then
            pad or crop to `image_config.padded_shape`.
        - `options`:
            A dictionary of options for advanced usage. This is passed directly to the underlying
            non-uniform FFT implementation if kernels are in fourier-space, or to the `nufftax`
            spreading function if kernels are in real-space.
        """  # noqa: E501
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `IndependentAtomProjection` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        if backend not in ["jax-finufft", "nufftax"]:
            raise ValueError(
                "`backend` in `IndependentAtomRenderFn` "
                "must be either 'jax-finufft' or 'nufftax'. Got "
                f"`backend = {backend}`."
            )
        self.backend = backend
        self.sampling_mode = sampling_mode
        self.block_size = block_size
        self.shape = shape
        self.upsample_factor = float(upsample_factor)
        self.eps = eps
        self.options = options

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
        # Prepare arguments
        pixel_size = image_config.pixel_size
        shape = image_config.padded_shape if self.shape is None else self.shape
        if self.block_size > 1:
            shape_b, pixel_size_b = (
                (self.block_size * shape[0], self.block_size * shape[1]),
                pixel_size / self.block_size,
            )
            frequency_grid = make_frequency_grid(
                shape_b, pixel_size_b, outputs_rfftfreqs=False
            )
        else:
            shape_b, pixel_size_b = shape, pixel_size
            frequency_grid = (
                image_config.get_frequency_grid(padding=True, physical=True, full=True)
                if shape == image_config.padded_shape
                else make_frequency_grid(shape, pixel_size, outputs_rfftfreqs=False)
            )
        # Compute projection
        return project_impl(
            volume_representation.positions,
            volume_representation.kernel_fns,
            shape=shape_b,
            pixel_size=pixel_size_b,
            backend=self.backend,
            frequency_grid=frequency_grid,
            sampling_mode=self.sampling_mode,
            eps=self.eps,
            upsampfac=self.upsample_factor,
            block_size=self.block_size,
            output_shape=image_config.padded_shape,
            outputs_real_space=outputs_real_space,
            options=self.options,
        )


IndependentAtomProjection.__doc__ = f"""Integrate atomic parametrization of a volume "
"onto the exit plane from an `IndependentAtomVolume`. {_REAL_VS_FOURIER_DOC}"""


def _check_kernel_fns(
    kernel_pytree: PyTree[AbstractFourierOperator] | PyTree[AbstractRealOperator],
    *,
    spatial_dim: int,
) -> tuple[bool, Callable]:
    kernel_list = jax.tree.leaves(
        kernel_pytree,
        is_leaf=lambda x: isinstance(x, (AbstractFourierOperator, AbstractRealOperator)),
    )
    if all(isinstance(kernel, AbstractRealOperator) for kernel in kernel_list):
        is_real_space = True
        is_leaf = lambda x: isinstance(x, AbstractRealOperator)
        for kernel in kernel_list:
            if 1 not in kernel.spatial_dims:
                raise ValueError(
                    "Found that `IndependentAtomVolume.kernel_fns` were "
                    "`AbstractRealOperator`s, but "
                    "one or more kernel did not support 1-D arrays as input. "
                    "The `AbstractRealOperator.spatial_dims` list must include `1` "
                    "to indicate support for 1D arrays."
                )
    elif all(isinstance(kernel, AbstractFourierOperator) for kernel in kernel_list):
        is_real_space = False
        is_leaf = lambda x: isinstance(x, AbstractFourierOperator)
        for kernel in kernel_list:
            if spatial_dim not in kernel.spatial_dims:
                raise ValueError(
                    "Found that `IndependentAtomVolume.kernel_fns` were "
                    "`AbstractFourierOperator`s, but "
                    f"one or more kernel did not support {spatial_dim}-D arrays as "
                    "input. The `AbstractFourierOperator.spatial_dims` list must "
                    f"include `{spatial_dim}` to indicate support for {spatial_dim}-D "
                    "arrays."
                )
    else:
        raise ValueError(
            "Found that `IndependentAtomVolume.kernel_fns` was not a "
            "PyTree containing only `AbstractFourierOperator`s or "
            "`AbstractRealOperator`s."
        )

    return is_real_space, is_leaf


def _maybe_gaussians_to_erf(
    kernel_pytree: PyTree[AbstractRealOperator],
    pixel_size: Float[Array, ""],
    sampling_mode: Literal["average", "point"],
    upsampfac: float,
) -> tuple[PyTree[AbstractRealOperator], Literal["average", "point"]]:
    """Modify gaussian kernels at runtime to the error function."""
    kernel_list = jax.tree.leaves(
        kernel_pytree,
        is_leaf=lambda x: isinstance(x, AbstractRealOperator),
    )
    if sampling_mode == "point":
        return kernel_pytree, "point"
    else:
        if all(isinstance(kernel, RealGaussian) for kernel in kernel_list):
            return (
                jax.tree.map(
                    lambda x: _IntegratedRealGaussian.from_gaussians(
                        x, pixel_size / upsampfac
                    ),
                    kernel_pytree,
                    is_leaf=lambda x: isinstance(x, AbstractRealOperator),
                ),
                "point",
            )
        return kernel_pytree, "average"


def _project_postprocess(
    projection_out: Inexact[Array, "_ _"],
    *,
    is_real_space: bool,
    compute_shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    frequency_grid: Float[Array, "_ _ 2"],
    sampling_mode: Literal["average", "point"],
    block_size: int,
    output_shape: tuple[int, int],
    outputs_real_space: bool,
) -> Inexact[Array, "_ _"]:
    reduce_shape = tuple(s // block_size for s in projection_out.shape)
    assert len(reduce_shape) == 2
    # Code path is the same for real-space and fourier-space
    del is_real_space
    projection_fft = projection_out
    if sampling_mode == "average":
        antialias_fn = FourierSinc(box_width=pixel_size)
        projection_fft *= antialias_fn(frequency_grid)
    if block_size % 2 == 0:
        projection_fft = _phase_shift_correct(
            projection_fft,
            block_size=block_size,
            target_shape=reduce_shape,
            pixel_size=pixel_size,
            frequency_grid=frequency_grid,
        )
    projection_fft = convert_fftn_to_rfftn(projection_fft, mode="real")
    # Block average and return
    block_average_fn = lambda x, factor: (
        block_reduce_downsample(x, factor, jax.lax.add, center_correct=False)
        / factor**x.ndim
    )
    if output_shape == reduce_shape:
        if block_size > 1:
            projection = block_average_fn(
                irfftn(projection_fft, s=compute_shape), block_size
            )
            return projection if outputs_real_space else rfftn(projection)
        else:
            return (
                irfftn(projection_fft, s=output_shape)
                if outputs_real_space
                else projection_fft
            )
    else:
        projection = irfftn(projection_fft, s=compute_shape)
        if block_size > 1:
            projection = block_average_fn(projection, block_size)
        projection = resize_with_crop_or_pad(projection, output_shape)
        return projection if outputs_real_space else rfftn(projection)


def project_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[AbstractRealOperator] | PyTree[AbstractFourierOperator],
    *,
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    backend: Literal["jax-finufft", "nufftax"],
    frequency_grid: Float[Array, "_ _ 2"],
    sampling_mode: Literal["average", "point"],
    eps: float,
    upsampfac: float,
    block_size: int,
    output_shape: tuple[int, int],
    outputs_real_space: bool,
    options: dict[str, Any],
) -> Array:
    is_real_space, is_leaf = _check_kernel_fns(
        kernel_fns,
        spatial_dim=2,
    )
    if is_real_space:
        kernel_fns, sampling_mode = _maybe_gaussians_to_erf(
            kernel_fns, pixel_size, sampling_mode, upsampfac
        )

    def real_impl(
        _shape: tuple[int, int],
        _ps: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractRealOperator,
    ) -> Array:
        u = upsampfac
        nspread = _eps_to_nspread(eps)
        shape_u = (int(u * _shape[0]), int(u * _shape[1]))
        (ny, nx), num_atoms = _shape, _positions.shape[0]
        box_xy = _ps * jnp.asarray((nx, ny), dtype=float)
        xy = 2 * jnp.pi * _positions[:, :2] / box_xy
        projection = spread_2d(
            x=xy[:, 0],
            y=xy[:, 1],
            c=jnp.ones(num_atoms, dtype=float),
            nf1=shape_u[1],
            nf2=shape_u[0],
            kernel_params=Kernel(
                nspread=nspread,
                phi=lambda _z: eqx.filter_vmap(_kernel_fn)((_ps / u) * _z),
            ),
            **options,
        )
        (indices_y, indices_x), fac = _build_extraction_mesh(shape_u, _shape)
        return fac * fftn(projection)[indices_y, indices_x]

    def fourier_impl(
        _shape: tuple[int, int],
        _ps: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractFourierOperator,
        _frequency_grid: Array,
    ) -> Array:
        (ny, nx), num_atoms = _shape, _positions.shape[0]
        box_xy = _ps * jnp.asarray((nx, ny))
        # Compute
        return (
            _kernel_fn(_frequency_grid)
            * jnp.fft.ifftshift(
                _nufft2d1(
                    _shape,  # type: ignore
                    source=jnp.full((num_atoms,), 1.0 + 0.0j),
                    xy=2 * jnp.pi * _positions[:, :2] / box_xy,
                    backend=backend,
                    eps=eps,
                    upsampfac=upsampfac,
                    options=options,
                )
                / _ps**2
            )
        )

    if is_real_space:
        project_impl, project_args = real_impl, ()
    else:
        project_impl, project_args = fourier_impl, (frequency_grid,)

    # Project and sum over kernels
    project_dispatch = lambda _positions, _kernel_fn: project_impl(
        shape, pixel_size, _positions, _kernel_fn, *project_args
    )
    projection_out = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(project_dispatch, positions, kernel_fns, is_leaf=is_leaf),
    )
    return _project_postprocess(
        projection_out,
        is_real_space=is_real_space,
        compute_shape=shape,
        pixel_size=pixel_size,
        frequency_grid=frequency_grid,
        sampling_mode=sampling_mode,
        block_size=block_size,
        output_shape=output_shape,
        outputs_real_space=outputs_real_space,
    )


def _render_postprocess(
    render_out: Inexact[Array, "_ _ _"],
    voxel_size: Float[Array, ""],
    *,
    is_real_space: bool,
    frequency_grid: Float[Array, "_ _ _ 3"],
    sampling_mode: Literal["average", "point"],
    outputs_real_space: bool,
    outputs_rfft: bool,
    fftshifted: bool,
) -> Inexact[Array, "_ _ _"]:
    # Code path is the same for real-space and fourier-space
    del is_real_space
    render_fft = render_out
    if sampling_mode == "average":
        antialias_fn = FourierSinc(box_width=voxel_size)
        render_fft *= antialias_fn(frequency_grid)
    if outputs_real_space:
        return ifftn(jnp.fft.ifftshift(render_fft)).real
    else:
        if outputs_rfft:
            render_fft = convert_fftn_to_rfftn(jnp.fft.ifftshift(render_fft), mode="real")
            if fftshifted:
                return jnp.fft.fftshift(render_fft, axes=(0, 1))
            else:
                return render_fft
        else:
            if fftshifted:
                return render_fft
            else:
                return jnp.fft.ifftshift(render_fft)


def render_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[AbstractRealOperator] | PyTree[AbstractFourierOperator],
    *,
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    backend: Literal["jax-finufft", "nufftax"],
    frequency_grid: Float[Array, "_ _ _ 3"],
    sampling_mode: Literal["average", "point"],
    eps: float,
    upsampfac: float,
    outputs_real_space: bool,
    outputs_rfft: bool,
    fftshifted: bool,
    options: dict[str, Any],
) -> Array:
    is_real_space, is_leaf = _check_kernel_fns(kernel_fns, spatial_dim=3)
    if is_real_space:
        kernel_fns, sampling_mode = _maybe_gaussians_to_erf(
            kernel_fns, voxel_size, sampling_mode, upsampfac
        )

    def real_impl(
        _shape: tuple[int, int, int],
        _vs: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractRealOperator,
    ) -> Array:
        nspread = _eps_to_nspread(eps)
        u = upsampfac
        shape_u = (int(u * _shape[0]), int(u * _shape[1]), int(u * _shape[2]))
        (nz, ny, nx), num_atoms = _shape, _positions.shape[0]
        box_xyz = _vs * jnp.asarray((nx, ny, nz), dtype=float)
        xyz = 2 * jnp.pi * _positions / box_xyz
        projection = spread_3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            c=jnp.ones(num_atoms, dtype=float),
            nf1=shape_u[2],
            nf2=shape_u[1],
            nf3=shape_u[0],
            kernel_params=Kernel(
                nspread=nspread,
                phi=lambda _z: eqx.filter_vmap(_kernel_fn)((_vs / u) * _z),
            ),
            **options,
        )
        (indices_z, indices_y, indices_x), fac = _build_extraction_mesh(shape_u, _shape)
        return fac * fftn(projection)[indices_z, indices_y, indices_x]

    def fourier_impl(
        _shape: tuple[int, int, int],
        _vs: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractFourierOperator,
        _frequency_grid: Array,
    ) -> Array:
        (nz, ny, nx), num_atoms = _shape, _positions.shape[0]
        box_xyz = _vs * jnp.asarray((nx, ny, nz))
        # Compute
        return _kernel_fn(_frequency_grid) * (
            _nufft3d1(
                _shape,  # type: ignore
                source=jnp.full((num_atoms,), 1.0 + 0.0j),
                xyz=2 * jnp.pi * _positions / box_xyz,
                backend=backend,
                eps=eps,
                upsampfac=upsampfac,
                options=options,
            )
            / _vs**3
        )

    if is_real_space:
        render_impl, render_args = real_impl, ()
    else:
        render_impl, render_args = fourier_impl, (frequency_grid,)

    render_dispatch = lambda _positions, _kernel_fn: render_impl(
        shape, voxel_size, _positions, _kernel_fn, *render_args
    )
    render_out = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(render_dispatch, positions, kernel_fns, is_leaf=is_leaf),
    )

    return _render_postprocess(
        render_out,
        is_real_space=is_real_space,
        voxel_size=voxel_size,
        frequency_grid=frequency_grid,
        sampling_mode=sampling_mode,
        outputs_real_space=outputs_real_space,
        outputs_rfft=outputs_rfft,
        fftshifted=fftshifted,
    )


def _phase_shift_correct(
    projection_fft: Array,
    *,
    block_size: int,
    target_shape: tuple[int, int],
    pixel_size: Array,
    frequency_grid: Array,
):
    if len(set(target_shape)) > 1:
        raise NotImplementedError(
            "Even `block_size` and non-square images not "
            "supported in `IndependentAtomProjection`. "
            f"Got `block_size = {block_size}` "
            f"and `shape = {target_shape}`."
        )
    dim = target_shape[0]
    if dim % 2 == 0:
        shift = jnp.full((2,), (block_size - 1) / 2)
    else:
        shift = jnp.full((2,), -0.5)
    return (
        jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(frequency_grid, pixel_size * shift)))
        * projection_fft
    )


def _make_jax_finufft_opts(upsampfac: float):
    assert NestedOpts is not None
    assert Opts is not None
    return NestedOpts(
        forward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
        backward=Opts(upsampfac=upsampfac, gpu_upsampfac=upsampfac),
    )


def _nufft2d1(
    shape: tuple[int, int],
    source: Array,
    xy: Array,
    *,
    backend: Literal["jax-finufft", "nufftax"],
    eps: float,
    upsampfac: float,
    options: dict[str, Any],
):
    if backend == "jax-finufft":
        if jax_finufft is None:
            raise RuntimeError(
                "Tried to use "
                "`IndependentAtomProjection(..., backend='jax-finufft')`, "
                "but `jax-finufft` is not installed. "
                "See https://github.com/flatironinstitute/jax-finufft "
                "for installation instructions."
            ) from JAX_FINUFFT_IMPORT_ERROR
        opts = (
            options.pop("opts")
            if "opts" in options
            else _make_jax_finufft_opts(upsampfac)
        )
        return jax_finufft.nufft1(
            shape, source, xy[:, 1], xy[:, 0], eps=eps, iflag=-1, opts=opts, **options
        )
    else:
        return nufftax.nufft2d1(
            n_modes=shape[::-1],
            c=source,
            x=xy[:, 0],
            y=xy[:, 1],
            eps=eps,
            isign=-1,
            **options,
        )


def _nufft3d1(
    shape: tuple[int, int, int],
    source: Array,
    xyz: Array,
    *,
    backend: Literal["jax-finufft", "nufftax"],
    eps: float,
    upsampfac: float,
    options: dict[str, Any],
):
    if backend == "jax-finufft":
        if jax_finufft is None:
            raise RuntimeError(
                "Tried to use "
                "`IndependentAtomRenderFn(..., backend='jax-finufft')`, "
                "but `jax-finufft` is not installed. "
                "See https://github.com/flatironinstitute/jax-finufft "
                "for installation instructions."
            ) from JAX_FINUFFT_IMPORT_ERROR
        opts = (
            options.pop("opts")
            if "opts" in options
            else _make_jax_finufft_opts(upsampfac)
        )
        return jax_finufft.nufft1(
            shape,
            source,
            xyz[:, 2],
            xyz[:, 1],
            xyz[:, 0],
            eps=eps,
            iflag=-1,
            opts=opts,
            **options,
        )
    else:
        return nufftax.nufft3d1(
            n_modes=shape[::-1],
            c=source,
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            eps=eps,
            isign=-1,
            **options,
        )


def _eps_to_nspread(eps: float) -> int:
    # FINUFFT heuristic for choosing `nspread` parameter
    # based on desired precision `eps`
    max_nspread = 16
    log_tol = -math.log10(max(eps, 1e-16))
    return max(2, min(int(math.ceil(log_tol + 1)), max_nspread))


def _build_extraction_mesh(shape: tuple[int, ...], shape_ds: tuple[int, ...]):
    """Build indices to extract modes from FFT output."""

    assert len(shape) == len(shape_ds)
    assert len(shape) in [2, 3]
    mean_factor = math.prod(shape_ds) / math.prod(shape)
    indices_1d = tuple(
        _build_extraction_indices_1d(dim, dim_ds) for dim, dim_ds in zip(shape, shape_ds)
    )
    return jnp.meshgrid(*indices_1d, indexing="ij"), mean_factor


def _build_extraction_indices_1d(dim: int, dim_ds: int):
    """Build indices to extract modes from FFT output. Adapted from `nufftax`"""
    q_min = -(dim_ds // 2)
    q_max = (dim_ds - 1) // 2

    idx_pos = jnp.arange(q_max + 1)
    idx_neg = jnp.arange(dim + q_min, dim)

    # `modeord = 1` case
    indices = jnp.concatenate([idx_pos, idx_neg])

    return indices


class _IntegratedRealGaussian(AbstractRealOperator, strict=True):
    amplitude: Float[Array, ""]
    variance: Float[Array, ""]

    pixel_size: Float[Array, ""]

    spatial_dims: ClassVar[list[int]] = [1]

    @classmethod
    def from_gaussians(cls, fn: RealGaussian, pixel_size: Float[Array, ""]):
        return cls(
            amplitude=fn.amplitude,
            variance=fn.variance,
            pixel_size=pixel_size,
        )

    @override
    def __call__(self, coordinate_grid: Float[Array, " x_dim"]) -> Array:
        scaling = 1.0 / jnp.sqrt(2 * self.variance)
        left, right = (
            coordinate_grid - self.pixel_size / 2,
            coordinate_grid + self.pixel_size / 2,
        )
        weight = (1 / (2 * self.pixel_size)) * (
            jsp.special.erf(scaling * right) - jsp.special.erf(scaling * left)
        )
        return self.amplitude * weight
