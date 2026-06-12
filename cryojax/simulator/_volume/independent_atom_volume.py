import functools as ft
import math
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, Self, TypeVar, cast
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
    b_factor_to_variance,
)
from ...jax_util import FloatLike, NDArrayLike
from ...ndimage import (
    AbstractFourierOperator,
    AbstractRealOperator,
    FourierGaussian,
    FourierSinc,
    RealGaussian,
    convert_fftn_to_rfftn,
    fftn,
    ifftn,
    irfftn,
    make_frequency_grid,
    query_efficient_grid_size,
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
from .common import make_frequencies_1d


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

    def __call__(self, frequencies: Float[Array, "... 2"] | Float[Array, "... 3"]):
        q_squared = jnp.sum(frequencies**2, axis=-1)
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
        frequencies: Float[Array, "... 2"] | Float[Array, "... 3"],
    ):
        q_squared = jnp.sum(frequencies**2, axis=-1)
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
    kernel_fns: PyTree[AbstractFourierOperator] | PyTree[RealGaussian]
    amplitudes: PyTree[Inexact[Array, " _"]] | None

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        positions: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        kernel_fns: (PyTree[AbstractFourierOperator, "T"] | PyTree[RealGaussian, "T"]),
        amplitudes: PyTree[Inexact[NDArrayLike, " _"], "T"] | None = None,
    ):
        """**Arguments:**

        - `positions`:
            A pytree of atom positions.
        - `kernel_fns`:
            A pytree of functions with the same tree structure
            as `positions`, where each leaf is a
            [`cryojax.ndimage.RealGaussian`][] or a
            [`cryojax.ndimage.AbstractFourierOperator`][].
            Real-space represents the scattering potential, while
            fourier-space represents the scattering factor.
            [`cryojax.ndimage.RealGaussian`][] classes may have
            amplitudes and variances with a batch dimension.
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
        if amplitudes is None:
            self.amplitudes = None
        else:
            if jax.tree.structure(positions) != jax.tree.structure(amplitudes):
                raise ValueError(
                    "When instantiating an `IndependentAtomVolume`, found "
                    "that the pytree structures of `positions` and "
                    "`amplitudes` were not equal."
                )
            self.amplitudes = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=float), amplitudes
            )

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
        use_real_space: bool = False,
        b_factor_by_element: FloatLike | tuple[FloatLike, ...] | None = None,
    ) -> Self:
        def make_kernel_fn(a, b, b_factor):
            if isinstance(parameters, PengScatteringFactorParameters):
                if use_real_space:
                    b = b + jnp.asarray(b_factor)[None] if b_factor is not None else b
                    return eqx.filter_vmap(
                        lambda _a, _b: RealGaussian(
                            amplitude=_a, variance=b_factor_to_variance(_b)
                        )
                    )(a, b)
                else:
                    return PengScatteringFactor(a, b, b_factor)
            elif isinstance(parameters, LobatoScatteringFactorParameters):
                if use_real_space:
                    raise NotImplementedError(
                        "`IndependentAtomVolume(..., parameters=..., "
                        "use_real_space=True)` does not support "
                        "`parameters = LobatoScatteringFactorParameters(...)`. "
                        "Instead, use `PengScatteringFactorParameters` or set "
                        "`use_real_space = False`."
                    )
                else:
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
    upsample_factor: float | None
    eps: float
    options: dict[str, Any]

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        backend: Literal["nufftax", "jax-finufft"] = "nufftax",
        sampling_mode: Literal["average", "point"] = "average",
        upsample_factor: int | float | None = None,
        eps: float = 1e-6,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
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
            If `IndependentAtomVolume` is instantiated with real-space
            gaussians, then error functions are used in
            `sampling_mode = 'average'`.
        - `upsample_factor`:
            How much to upsample the grid on which atoms are spread onto.
            If equal to `None`, choose a default value at run-time.
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
        self.upsample_factor = None if upsample_factor is None else float(upsample_factor)
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
        # Prepare arguments
        positions, kernel_fns = (
            volume_representation.positions,
            volume_representation.kernel_fns,
        )
        amplitudes = _standardize_amplitudes(volume_representation.amplitudes, positions)
        is_real_space, kernel_fns = _standardize_kernel_fns(kernel_fns, spatial_dim=3)
        (shape_u, voxel_size_u), upsampfac = _prepare_upsample(
            shape=self.shape,
            pixel_size=self.voxel_size,
            upsampfac=self.upsample_factor,
            sampling_mode=self.sampling_mode,
            is_real_space=is_real_space,
        )
        # Modify kernels if using error functions
        sampling_mode = self.sampling_mode
        if is_real_space:
            kernel_fns, sampling_mode = _maybe_use_erf(
                kernel_fns,
                pixel_size=self.voxel_size,
                sampling_mode=sampling_mode,
                upsampfac=upsampfac,
            )
        # Compute
        rendering_out = render_impl(
            positions,
            kernel_fns,
            amplitudes,
            is_real_space=is_real_space,
            shape_u=cast(tuple[int, int, int], shape_u),
            shape_out=cast(tuple[int, int, int], self.shape),
            voxel_size_u=voxel_size_u,
            backend=self.backend,
            eps=self.eps,
            options=self.options,
        )
        if is_real_space:
            # Check case where we can return immediately
            rendering = rendering_out
            if sampling_mode == "point" and self.shape == shape_u:
                return (
                    rendering
                    if outputs_real_space
                    else _prepare_real_to_fft(
                        rendering,
                        outputs_rfft=outputs_rfft,
                        fftshifted=fftshifted,
                    )
                )
            else:
                rendering_fft = fftn(rendering)
        else:
            # Otherwise, postprocess in fourier-domain
            rendering_fft = rendering_out
        # Average within a pixel size
        if sampling_mode == "average":
            box_fn = FourierSinc(box_width=self.voxel_size)
            frequencies_1d = make_frequencies_1d(shape_u, voxel_size_u, modeord=1)
            rendering_fft *= eval_separable_impl(box_fn, frequencies_1d)
        # Downsample by extracting fourier modes
        if self.shape != shape_u:
            (indices_z, indices_y, indices_x), fac = _build_extraction_mesh(
                shape_u, self.shape, modeord=1
            )
            rendering_fft = fac * rendering_fft[indices_z, indices_y, indices_x]
        return (
            ifftn(rendering_fft).real
            if outputs_real_space
            else _prepare_fft_to_fft(
                rendering_fft,
                outputs_rfft=outputs_rfft,
                fftshifted=fftshifted,
            )
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
    upsample_factor: float | None
    eps: float
    shape: tuple[int, int] | None
    options: dict[str, Any]

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        backend: Literal["jax-finufft", "nufftax"] = "nufftax",
        sampling_mode: Literal["average", "point"] = "average",
        upsample_factor: int | float | None = None,
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
            If `IndependentAtomVolume` is instantiated with real-space
            gaussians, then error functions are used in
            `sampling_mode = 'average'`.
        - `upsample_factor`:
            How much to upsample the grid on which atoms are spread onto.
            If equal to `None`, choose a default value at run-time.
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
        self.shape = shape
        self.upsample_factor = None if upsample_factor is None else float(upsample_factor)
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
        output_shape = image_config.padded_shape
        positions, kernel_fns = (
            volume_representation.positions,
            volume_representation.kernel_fns,
        )
        amplitudes = _standardize_amplitudes(volume_representation.amplitudes, positions)
        is_real_space, kernel_fns = _standardize_kernel_fns(kernel_fns, spatial_dim=2)
        (shape_u, pixel_size_u), upsampfac = _prepare_upsample(
            shape=shape,
            pixel_size=pixel_size,
            upsampfac=self.upsample_factor,
            sampling_mode=self.sampling_mode,
            is_real_space=is_real_space,
        )
        # Modify kernels if using error functions
        sampling_mode = self.sampling_mode
        if is_real_space:
            kernel_fns, sampling_mode = _maybe_use_erf(
                kernel_fns,
                pixel_size=pixel_size,
                sampling_mode=sampling_mode,
                upsampfac=upsampfac,
            )
        # Compute projection
        projection_out = project_impl(
            positions,
            kernel_fns,
            amplitudes,
            is_real_space=is_real_space,
            shape_u=cast(tuple[int, int], shape_u),
            pixel_size_u=pixel_size_u,
            shape_out=cast(tuple[int, int], shape),
            backend=self.backend,
            eps=self.eps,
            options=self.options,
        )
        if is_real_space:
            # Check case where we can return immediately
            projection = projection_out
            if sampling_mode == "point" and shape == shape_u:
                if output_shape != shape:
                    projection = resize_with_crop_or_pad(projection, output_shape)
                return projection if outputs_real_space else rfftn(projection)
            else:
                projection_fft = fftn(projection)
        else:
            # Otherwise, postprocess in fourier-domain
            projection_fft = projection_out
        # Average within a pixel size
        if sampling_mode == "average":
            box_fn = FourierSinc(box_width=pixel_size)
            frequencies_1d = make_frequencies_1d(shape_u, pixel_size_u, modeord=1)
            projection_fft *= eval_separable_impl(box_fn, frequencies_1d)
        # Downsample by extracting fourier modes
        if shape != shape_u:
            (indices_y, indices_x), fac = _build_extraction_mesh(
                shape_u, shape, modeord=1
            )
            projection_fft = fac * projection_fft[indices_y, indices_x]
        projection_fft = convert_fftn_to_rfftn(projection_fft, mode="real")
        if output_shape == shape:
            return (
                irfftn(projection_fft, s=output_shape)
                if outputs_real_space
                else projection_fft
            )
        else:
            projection = irfftn(projection_fft, s=shape)
            projection = resize_with_crop_or_pad(projection, output_shape)
            return projection if outputs_real_space else rfftn(projection)


IndependentAtomProjection.__doc__ = f"""Integrate atomic parametrization of a volume "
"onto the exit plane from an `IndependentAtomVolume`. {_REAL_VS_FOURIER_DOC}"""


def _standardize_kernel_fns(
    kernel_pytree: PyTree[AbstractFourierOperator] | PyTree[RealGaussian],
    *,
    spatial_dim: int,
) -> tuple[bool, PyTree[AbstractFourierOperator] | PyTree[RealGaussian]]:
    kernel_list = jax.tree.leaves(
        kernel_pytree,
        is_leaf=lambda x: isinstance(x, (AbstractFourierOperator, AbstractRealOperator)),
    )
    if all(isinstance(kernel, RealGaussian) for kernel in kernel_list):
        # Standardize gaussian kernels for computation
        is_real_space = True
        # ... pytree leaves have a batch dim
        kernel_pytree = jax.tree.map(lambda x: jnp.atleast_1d(x), kernel_pytree)
        # ... amplitude must be spread per dimension
        replace_fn = lambda fn: eqx.tree_at(
            lambda _fn: _fn.amplitude, fn, (fn.amplitude ** (1 / spatial_dim))
        )
        kernel_pytree = jax.tree.map(
            replace_fn, kernel_pytree, is_leaf=lambda x: isinstance(x, RealGaussian)
        )
    elif all(isinstance(kernel, AbstractFourierOperator) for kernel in kernel_list):
        is_real_space = False
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
            "`RealGaussian`s."
        )

    return is_real_space, kernel_pytree


def _maybe_use_erf(
    kernel_pytree: PyTree[RealGaussian],
    pixel_size: Float[Array, ""],
    sampling_mode: Literal["average", "point"],
    upsampfac: float,
) -> tuple[PyTree[RealGaussian], Literal["average", "point"]]:
    """Modify gaussian kernels at runtime to the error function."""
    if sampling_mode == "average":
        if upsampfac == 1.0:
            return (
                jax.tree.map(
                    lambda x: _make_erf(x, pixel_size),
                    kernel_pytree,
                    is_leaf=lambda x: isinstance(x, AbstractRealOperator),
                ),
                "point",
            )
        else:
            return kernel_pytree, "average"
    else:
        return (kernel_pytree, "point")


def project_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[RealGaussian] | PyTree[AbstractFourierOperator],
    amplitudes: PyTree[Inexact[Array, " _"]],
    *,
    is_real_space: bool,
    shape_u: tuple[int, int],
    pixel_size_u: Float[Array, ""],
    shape_out: tuple[int, int],
    backend: Literal["jax-finufft", "nufftax"],
    eps: float,
    options: dict[str, Any],
) -> Array:
    is_leaf = (
        (lambda x: isinstance(x, AbstractRealOperator))
        if is_real_space
        else (lambda x: isinstance(x, AbstractFourierOperator))
    )
    frequency_grid = _maybe_make_frequency_grid(
        shape_u, pixel_size_u, is_real_space=is_real_space, kernel_fns=kernel_fns
    )
    frequencies_1d = make_frequencies_1d(shape_u, pixel_size_u, modeord=0)

    def real_impl(
        _shape: tuple[int, int],
        _ps: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractRealOperator,
        _amplitudes: Inexact[Array, " _"],
    ) -> Array:
        nspread = _eps_to_nspread(eps)
        xy = _normalize_positions(_positions[:, :2], _shape, _ps)
        projection = _spread_2d(
            kernel_fn=_kernel_fn,
            pixel_size=_ps,
            x=xy[:, 0],
            y=xy[:, 1],
            c=_amplitudes.astype(float),
            nf1=_shape[1],
            nf2=_shape[0],
            nspread=nspread,
            options=options,
        )
        return projection

    # Per-dimension NUFFT offset: 2*pi*(N//2)/N maps physical center (x=0) to
    # integer pixel index N//2.  For even N this equals pi; for odd N it is
    # pi*(1 - 1/N).  Must use the original output shape, not the upsampled one.
    _nufft_offsets_2d = jnp.asarray(
        [2 * jnp.pi * (s // 2) / s for s in shape_out[::-1][:2]]
    )

    def fourier_impl(
        _shape: tuple[int, int],
        _ps: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractFourierOperator,
        _amplitudes: Inexact[Array, " _"],
        _f_1d: tuple[Array, ...],
        _f_mesh: Array | None,
    ) -> Array:
        # Scale positions onto the upsampled grid, then apply the shape_out-based
        # center offset.  Do NOT apply the shape_u parity correction here: that
        # correction is for the spread path; the NUFFT offset is entirely
        # determined by shape_out regardless of the parity of shape_u.
        _ns = jnp.asarray(_shape[::-1][:2], dtype=float)
        xy = 2 * jnp.pi * _positions[:, :2] / (_ps * _ns) + _nufft_offsets_2d
        return (
            eval_kernel_impl(_kernel_fn, f_1d=_f_1d, f_mesh=_f_mesh)
            * _nufft2d1(
                _shape,  # type: ignore
                source=_amplitudes.astype(complex),
                xy=xy,
                backend=backend,
                eps=eps,
                options=options,
            )
            / _ps**2
        )

    if is_real_space:
        project_impl, args = real_impl, ()
    else:
        project_impl, args = (
            fourier_impl,
            cast(
                Any,
                (
                    frequencies_1d,
                    frequency_grid,
                ),
            ),
        )

    # Project and sum over kernels
    project_dispatch = lambda _positions, _kernel_fn, _amplitudes: project_impl(
        shape_u, pixel_size_u, _positions, _kernel_fn, _amplitudes, *args
    )
    projection_out = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(
            project_dispatch, positions, kernel_fns, amplitudes, is_leaf=is_leaf
        ),
    )
    if not is_real_space:
        projection_out = jnp.fft.ifftshift(projection_out)

    return projection_out


def render_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[RealGaussian] | PyTree[AbstractFourierOperator],
    amplitudes: PyTree[Inexact[Array, " _"]],
    *,
    is_real_space: bool,
    shape_u: tuple[int, int, int],
    voxel_size_u: Float[Array, ""],
    shape_out: tuple[int, int, int],
    backend: Literal["jax-finufft", "nufftax"],
    eps: float,
    options: dict[str, Any],
) -> Array:
    is_leaf = (
        (lambda x: isinstance(x, AbstractRealOperator))
        if is_real_space
        else (lambda x: isinstance(x, AbstractFourierOperator))
    )
    frequency_grid = _maybe_make_frequency_grid(
        shape_u, voxel_size_u, is_real_space=is_real_space, kernel_fns=kernel_fns
    )
    frequencies_1d = make_frequencies_1d(shape_u, voxel_size_u, modeord=0)

    _nufft_offsets_3d = jnp.asarray([2 * jnp.pi * (s // 2) / s for s in shape_out[::-1]])

    def real_impl(
        _shape: tuple[int, int, int],
        _vs: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractRealOperator,
        _amplitudes: Inexact[Array, " _"],
    ) -> Array:
        nspread = _eps_to_nspread(eps)
        xyz = _normalize_positions(_positions, _shape, _vs)
        rendering = _spread_3d(
            kernel_fn=_kernel_fn,
            voxel_size=_vs,
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            c=_amplitudes.astype(float),
            nf1=_shape[2],
            nf2=_shape[1],
            nf3=_shape[0],
            nspread=nspread,
            options=options,
        )
        return rendering

    def fourier_impl(
        _shape: tuple[int, int, int],
        _vs: Float[Array, ""],
        _positions: Float[Array, "_ 3"],
        _kernel_fn: AbstractFourierOperator,
        _amplitudes: Inexact[Array, " _"],
        _f_1d: tuple[Array, ...],
        _f_mesh: Array | None,
    ) -> Array:
        _ns = jnp.asarray(_shape[::-1], dtype=float)
        xyz = 2 * jnp.pi * _positions / (_vs * _ns) + _nufft_offsets_3d
        return eval_kernel_impl(_kernel_fn, f_1d=_f_1d, f_mesh=_f_mesh) * (
            _nufft3d1(
                _shape,  # type: ignore
                source=_amplitudes.astype(complex),
                xyz=xyz,
                backend=backend,
                eps=eps,
                options=options,
            )
            / _vs**3
        )

    if is_real_space:
        compute_fn, args = real_impl, ()
    else:
        compute_fn, args = cast(Any, (fourier_impl, (frequencies_1d, frequency_grid)))

    render_dispatch = lambda _positions, _kernel_fn, _amplitudes: compute_fn(
        shape_u, voxel_size_u, _positions, _kernel_fn, _amplitudes, *args
    )
    rendering_out = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(render_dispatch, positions, kernel_fns, amplitudes, is_leaf=is_leaf),
    )
    if not is_real_space:
        rendering_out = jnp.fft.ifftshift(rendering_out)

    return rendering_out


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
    options: dict[str, Any],
):
    default_upsampfac = 1.25
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
            else _make_jax_finufft_opts(upsampfac=default_upsampfac)
        )
        return jax_finufft.nufft1(
            shape,
            source,
            xy[:, 1],
            xy[:, 0],
            eps=eps,
            iflag=-1,
            opts=opts,
            **options,
        )
    else:
        upsampfac = (
            options.pop("upsampfac") if "upsampfac" in options else default_upsampfac
        )
        return nufftax.nufft2d1(
            n_modes=shape[::-1],
            c=source,
            x=xy[:, 0],
            y=xy[:, 1],
            eps=eps,
            isign=-1,
            upsampfac=upsampfac,
            **options,
        )


def _nufft3d1(
    shape: tuple[int, int, int],
    source: Array,
    xyz: Array,
    *,
    backend: Literal["jax-finufft", "nufftax"],
    eps: float,
    options: dict[str, Any],
):
    default_upsampfac = 1.25
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
            else _make_jax_finufft_opts(upsampfac=default_upsampfac)
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
        upsampfac = (
            options.pop("upsampfac") if "upsampfac" in options else default_upsampfac
        )
        return nufftax.nufft3d1(
            n_modes=shape[::-1],
            c=source,
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            eps=eps,
            isign=-1,
            upsampfac=upsampfac,
            **options,
        )


def _spread_3d(kernel_fn, voxel_size, x, y, z, c, nf1, nf2, nf3, *, nspread, options):
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, None, None, None))
    def spread_3d_impl(_kernel_fn, _voxel_size, _x, _y, _z, _c):
        return spread_3d(
            x=_x,
            y=_y,
            z=_z,
            c=_c,
            nf1=nf1,
            nf2=nf2,
            nf3=nf3,
            kernel_params=_make_nufftax_kernel(
                _kernel_fn, pixel_size=_voxel_size, nspread=nspread
            ),
            **options,
        )

    return jnp.sum(spread_3d_impl(kernel_fn, voxel_size, x, y, z, c), axis=0)


def _spread_2d(kernel_fn, pixel_size, x, y, c, nf1, nf2, *, nspread, options):
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, None, None))
    def spread_2d_impl(_kernel_fn, _pixel_size, _x, _y, _c):
        return spread_2d(
            x=_x,
            y=_y,
            c=_c,
            nf1=nf1,
            nf2=nf2,
            kernel_params=_make_nufftax_kernel(
                _kernel_fn, pixel_size=_pixel_size, nspread=nspread
            ),
            **options,
        )

    return jnp.sum(spread_2d_impl(kernel_fn, pixel_size, x, y, c), axis=0)


def _eps_to_nspread(eps: float) -> int:
    # FINUFFT heuristic for choosing `nspread` parameter
    # based on desired precision `eps`. In this context it is
    # used for spreading gaussians, so we let it be unbounded
    log_tol = -math.log10(max(eps, 1e-16))
    return max(2, int(math.ceil(log_tol + 1)))


def _build_extraction_mesh(
    shape: tuple[int, ...], shape_ds: tuple[int, ...], *, modeord: int
):
    """Build indices to extract modes from FFT output."""
    assert len(shape) == len(shape_ds)
    assert len(shape) in [2, 3]
    mean_factor = math.prod(shape_ds) / math.prod(shape)
    indices_1d = tuple(
        _build_extraction_indices_1d(dim, dim_ds, modeord=modeord)
        for dim, dim_ds in zip(shape, shape_ds)
    )
    return jnp.meshgrid(*indices_1d, indexing="ij"), mean_factor


def _build_extraction_indices_1d(dim: int, dim_ds: int, *, modeord: int):
    """Build indices to extract modes from FFT output. Adapted from `nufftax`"""
    q_min = -(dim_ds // 2)
    q_max = (dim_ds - 1) // 2

    idx_pos = jnp.arange(q_max + 1)
    idx_neg = jnp.arange(dim + q_min, dim)

    if modeord == 0:
        indices = jnp.concatenate([idx_neg, idx_pos])
    else:
        indices = jnp.concatenate([idx_pos, idx_neg])

    return indices


def _standardize_amplitudes(amplitudes: PyTree[Array] | None, positions: PyTree[Array]):
    if amplitudes is None:
        return jax.tree.map(
            lambda x: jnp.ones((x.shape[0],), dtype=float),
            positions,
        )
    else:
        return amplitudes


def _make_nufftax_kernel(
    kernel_fn: AbstractRealOperator,
    *,
    pixel_size: Float[Array, ""],
    nspread: int,
):
    return Kernel(
        nspread=nspread,
        phi=lambda _z: eqx.filter_vmap(kernel_fn)(pixel_size * _z),
    )


class _Erf(AbstractRealOperator, strict=True):
    amplitude: Float[Array, ""]
    variance: Float[Array, ""]

    pixel_size: Float[Array, ""]

    spatial_dims: ClassVar[list[int]] = [1]

    @override
    def __call__(self, coordinates: Float[Array, " dim"]) -> Float[Array, " dim"]:
        scaling = 1.0 / jnp.sqrt(2 * self.variance)
        left, right = (
            coordinates - self.pixel_size / 2,
            coordinates + self.pixel_size / 2,
        )
        weight = (1 / (2 * self.pixel_size)) * (
            jsp.special.erf(scaling * right) - jsp.special.erf(scaling * left)
        )
        return self.amplitude * weight


@eqx.filter_vmap(in_axes=(0, None))
def _make_erf(gaussian: RealGaussian, pixel_size: Float[Array, ""]):
    return _Erf(
        amplitude=gaussian.amplitude, variance=gaussian.variance, pixel_size=pixel_size
    )


def _prepare_upsample(
    shape: tuple[int, ...],
    pixel_size: Float[Array, ""],
    upsampfac: float | None,
    *,
    is_real_space: bool,
    sampling_mode: Literal["point", "average"],
) -> tuple[tuple[tuple[int, ...], Array], float]:
    """Find the upsampfac that perfectly divides the upsampled
    image shape.
    """
    if upsampfac is None:
        upsampfac = 1.0 if (is_real_space and sampling_mode == "average") else 2.0
    if upsampfac == 1.0:
        return (shape, pixel_size), upsampfac
    else:
        dim = shape[0]
        if not all(s == dim for s in shape):
            # If rectangular image focus on accuracy. Make sure
            # `upsampfac` is mapped directly onto a pixel size
            # dilation
            gcd = ft.reduce(math.gcd, shape)
            upsampfac = round(gcd * upsampfac) / gcd
            shape_u = tuple(s * upsampfac for s in shape)
        else:
            # Otherwise, find an efficient grid for FFTs close to
            # the requested upsampfac and return the corrected upsampfac
            shape_u = query_efficient_grid_size(shape, upsampfac, match_parity=False)
            dim_u = shape_u[0]
            upsampfac = dim_u / dim
        return (
            (tuple(int(upsampfac * s) for s in shape), pixel_size / upsampfac),
            upsampfac,
        )


def _prepare_real_to_fft(a: Array, *, outputs_rfft: bool, fftshifted: bool) -> Array:
    if outputs_rfft:
        f = rfftn(a)
        return jnp.fft.fftshift(f, axes=(0, 1)) if fftshifted else f
    else:
        f = fftn(a)
        return jnp.fft.fftshift(f) if fftshifted else f


def _prepare_fft_to_fft(f: Array, *, outputs_rfft: bool, fftshifted: bool) -> Array:
    if outputs_rfft:
        f = convert_fftn_to_rfftn(f, mode="real")
        return jnp.fft.fftshift(f, axes=(0, 1)) if fftshifted else f
    else:
        return jnp.fft.fftshift(f) if fftshifted else f


def _is_separable(kernel_fn: AbstractFourierOperator):
    return isinstance(kernel_fn, (FourierGaussian, PengScatteringFactor))


def _maybe_make_frequency_grid(
    shape: tuple[int, ...],
    pixel_size: Float[Array, ""],
    *,
    is_real_space: bool,
    kernel_fns: PyTree[AbstractFourierOperator],
):
    is_leaf = lambda x: isinstance(x, AbstractFourierOperator)
    all_separable = jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda x: _is_separable(x), kernel_fns, is_leaf=is_leaf),
    )

    if is_real_space:
        frequency_grid = None
    elif all_separable:
        frequency_grid = None
    else:
        frequency_grid = make_frequency_grid(shape, pixel_size, fftshifted=True)

    return frequency_grid


def eval_kernel_impl(
    kernel_fn: AbstractFourierOperator, *, f_1d: tuple[Array, ...], f_mesh: Array | None
):
    if isinstance(kernel_fn, FourierGaussian):
        assert f_1d is not None
        return eval_separable_impl(kernel_fn, f_1d)
    elif isinstance(kernel_fn, PengScatteringFactor):
        assert f_1d is not None
        return eval_peng_impl(kernel_fn, f_1d)
    else:
        assert f_mesh is not None
        return eval_non_separable_impl(kernel_fn, f_mesh)


def eval_peng_impl(kernel_fn: PengScatteringFactor, frequencies_1d: tuple[Array, ...]):
    a, b = kernel_fn.a, kernel_fn.b
    if kernel_fn.b_factor is not None:
        b = b + kernel_fn.b_factor[None]
    # Split amplitude across dimensions so the separable product recovers the
    # original: (a^(1/ndim))^ndim = a
    make_gaussians = jax.vmap(lambda _a, _b: FourierGaussian(amplitude=_a, b_factor=_b))
    eval_gaussians = jax.vmap(eval_separable_impl, in_axes=(0, None))
    return jnp.sum(eval_gaussians(make_gaussians(a, b), frequencies_1d), axis=0)


def eval_separable_impl(
    kernel_fn: FourierGaussian | FourierSinc, frequencies_1d: tuple[Array, ...]
):
    ndim = len(frequencies_1d)
    assert 1 in kernel_fn.spatial_dims and ndim in [2, 3]
    if isinstance(kernel_fn, FourierGaussian):
        kernel_fn = eqx.tree_at(
            lambda x: x.amplitude, kernel_fn, replace_fn=lambda x: x ** (1 / ndim)
        )
    if len(frequencies_1d) == 2:
        q_x, q_y = frequencies_1d
        return kernel_fn(q_x)[None, :] * kernel_fn(q_y)[:, None]
    else:
        q_x, q_y, q_z = frequencies_1d
        return (
            kernel_fn(q_x)[None, None, :]
            * kernel_fn(q_y)[None, :, None]
            * kernel_fn(q_z)[:, None, None]
        )


def eval_non_separable_impl(kernel_fn: AbstractFourierOperator, frequency_grid: Array):
    assert frequency_grid.shape[-1] in [2, 3]
    return kernel_fn(frequency_grid)


def _normalize_positions(
    positions: Array,
    shape: tuple[int, ...],
    pixel_size: Array,
) -> Array:
    ndim = positions.shape[-1]
    # shape is (z, y, x) or (y, x); reverse so first element is x
    shape_spatial = shape[::-1][:ndim]
    ns = jnp.asarray(shape_spatial, dtype=float)
    # Center is at index N//2 for all N (RELION convention).
    # For even N the offset is 0; for odd N it is -π/N so that x=0 lands
    # exactly on pixel N//2 after fold_rescale.
    offsets = jnp.pi * jnp.asarray([-(s % 2) / s for s in shape_spatial])
    return 2 * jnp.pi * positions / (pixel_size * ns) + offsets
