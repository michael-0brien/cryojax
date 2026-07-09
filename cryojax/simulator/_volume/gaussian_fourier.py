import functools as ft
import math
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, Self, TypeVar, cast
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import nufftax
from jaxtyping import Array, Float, PyTree

from ..._config import CRYOJAX_FINUFFT_BACKEND
from ..._internal import leaf_asarray
from ...constants import (
    PengScatteringFactorParameters,
)
from ...jax_util import FloatLike, NDArrayLike
from ...ndimage import (
    AbstractFourierOperator,
    FourierGaussian,
    FourierSinc,
    convert_fftn_to_rfftn,
    query_efficient_grid_size,
    resize_with_crop_or_pad,
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


class GaussianFourierVolume(AbstractAtomVolume, strict=True):
    """A representation of a volume that accepts an array of
    atom positions and an electron scattering factor for these
    atoms, projected/rendered via non-uniform FFTs (see
    [`cryojax.simulator.GaussianFourierProjection`][]/
    [`cryojax.simulator.GaussianFourierRenderFn`][]).

    !!! example "A Gaussian at each atom"
        ```python
        import cryojax.simulator as cxs
        import cryojax.ndimage as im

        positions = ... # load atom positions
        b_factor = ...  # ... and a B-factor
        volume = cxs.GaussianFourierVolume(
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
        volume = cxs.GaussianFourierVolume(
            positions=(positions_1, positions_2),
            kernel_fns=(im.FourierGaussian(b_factor=b_factor_1), im.FourierGaussian(b_factor=b_factor_2))
        )
        ```

    See [`cryojax.simulator.GaussianFourierVolume.from_tabulated_parameters`][] for
    loading a volume from tabulated electron scattering factors.
    """  # noqa: E501

    positions: PyTree[Float[NDArrayLike, "_ 3"]]
    kernel_fns: PyTree[FourierGaussian]

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        positions: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        kernel_fns: PyTree[FourierGaussian, "T"],
    ):
        """**Arguments:**

        - `positions`:
            A pytree of atom positions.
        - `kernel_fns`:
            A pytree of functions with the same tree structure
            as `positions`, where each leaf is a
            [`cryojax.ndimage.FourierGaussian`][] representing the atom
            type's scattering factor. These may have amplitudes and
            b-factors with a batch dimension to simulate form factors. To
            use a different amplitude for each atom position, use
            [`cryojax.simulator.GaussianMixtureVolume`][] instead.
        """  # noqa: E501
        if jax.tree.structure(positions) != jax.tree.structure(
            kernel_fns,
            is_leaf=lambda x: isinstance(x, AbstractFourierOperator),
        ):
            raise ValueError(
                "When instantiating a `GaussianFourierVolume`, found "
                "that the pytree structures of `positions` and "
                "`kernel_fns` were not equal."
            )
        # Convert positions to array leaves, preserving their backend (JAX
        # arrays stay on-device, NumPy arrays stay on the host).
        self.positions = jax.tree.map(lambda x: leaf_asarray(x, dtype=float), positions)
        self.kernel_fns = kernel_fns

    @override
    def rotate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        rotate_fn = lambda pos: pose.rotate_coordinates(
            jnp.asarray(pos), inverse=self.is_frame_rotation
        )
        return eqx.tree_at(
            lambda x: x.positions,
            self,
            jax.tree.map(rotate_fn, self.positions),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with translated `positions`."""
        offset_in_angstroms = jnp.asarray(pose.offset_in_angstroms)
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        translate_fn = lambda pos: jnp.asarray(pos) + offset_in_angstroms
        return eqx.tree_at(
            lambda x: x.positions,
            self,
            jax.tree.map(translate_fn, self.positions),
        )

    @classmethod
    def from_tabulated_parameters(
        cls,
        positions_by_element: tuple[Float[NDArrayLike, "_ 3"], ...],
        parameters: PengScatteringFactorParameters,
        *,
        b_factor_by_element: FloatLike | tuple[FloatLike, ...] | None = None,
    ) -> Self:
        def make_kernel_fn(a, b, b_factor):
            if isinstance(parameters, PengScatteringFactorParameters):
                b = b + jnp.asarray(b_factor)[None] if b_factor is not None else b
                return eqx.filter_vmap(
                    lambda _a, _b: FourierGaussian(amplitude=_a, b_factor=_b)
                )(a, b)
            else:
                raise ValueError(
                    "Unrecognized argument `parameters` when "
                    "calling `GaussianFourierVolume.from_tabulated_parameters`. "
                    "This should be type "
                    "`cryojax.constants.PengScatteringFactorParameters`, "
                    f"but got type {parameters.__class__.__name__}."
                )

        n_elements = len(positions_by_element)
        a, b = parameters.a, parameters.b
        if a.shape[0] != n_elements or b.shape[0] != n_elements:
            raise ValueError(
                "When constructing a `GaussianFourierVolume` via "
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
                        "When constructing a `GaussianFourierVolume` via "
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


class GaussianFourierRenderFn(AbstractVolumeRenderFn[GaussianFourierVolume], strict=True):
    """Render a voxel grid from a `GaussianFourierVolume` using non-uniform FFTs
    and Fourier-domain convolution. Good when kernels span at least a
    couple pixels; see `cryojax.simulator.GaussianMixtureRenderFn` for an
    alternative that directly spreads narrow kernels onto the grid.

    !!! info
        By default, the non-uniform FFT runs on a pure-JAX backend using
        [`nufftax`](https://github.com/GragasLab/nufftax/tree/custom-kernel-spread).
        Setting the environment variable `CRYOJAX_FINUFFT_BACKEND=jax-finufft` switches to
        [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft), which
        can be more computationally efficient and less memory-demanding, at the
        cost of being trickier to install and having more limited integration
        with multi-GPU JAX.
    """  # noqa: E501

    shape: tuple[int, int, int]
    voxel_size: Float[NDArrayLike, "..."]

    sampling_mode: Literal["average", "point"]
    upsample_factor: float
    eps: float
    options: dict[str, Any]

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        sampling_mode: Literal["average", "point"] = "average",
        upsample_factor: int | float = 1.0,
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
        - `upsample_factor`:
            How much to upsample the grid on which atoms are spread onto.
        - `eps`:
            The precision of the underlying non-uniform FFT implementation.
            See [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)
            for documentation.
        - `options`:
            A dictionary of options for advanced usage, passed directly to
            the underlying non-uniform FFT implementation.
        """  # noqa: E501
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `GaussianFourierRenderFn` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        self.shape = shape
        self.voxel_size = leaf_asarray(voxel_size, dtype=float)
        self.sampling_mode = sampling_mode
        self.upsample_factor = float(upsample_factor)
        self.eps = eps
        self.options = options

    @override
    def __call__(
        self,
        volume_representation: GaussianFourierVolume,
        *,
        outputs_real_space: bool = True,
        outputs_rfft: bool = False,
        fftshifted: bool = False,
    ) -> VoxelArray:
        """**Arguments:**

        - `volume_representation`:
            The `GaussianFourierVolume`.
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
        # Prepare arguments, casting positions and the voxel size to JAX arrays
        voxel_size = jnp.asarray(self.voxel_size)
        positions, kernel_fns = (
            jax.tree.map(lambda x: jnp.asarray(x), volume_representation.positions),
            volume_representation.kernel_fns,
        )
        kernel_fns = _standardize_kernel_fns(kernel_fns, spatial_dim=3)
        (shape_u, voxel_size_u), upsampfac = _prepare_upsample(
            shape=self.shape,
            pixel_size=voxel_size,
            upsampfac=self.upsample_factor,
        )
        # Compute
        rendering_fft = _render_impl(
            positions,
            kernel_fns,
            shape_u=cast(tuple[int, int, int], shape_u),
            shape_out=cast(tuple[int, int, int], self.shape),
            voxel_size_u=voxel_size_u,
            eps=self.eps,
            options=self.options,
        )
        # Average within a pixel size
        if self.sampling_mode == "average":
            box_fn = FourierSinc(box_width=voxel_size)
            frequencies_1d = make_frequencies_1d(shape_u, voxel_size_u, modeord=1)
            rendering_fft *= _eval_separable_impl(box_fn, frequencies_1d)
        # Downsample by extracting fourier modes
        if self.shape != shape_u:
            (indices_z, indices_y, indices_x), fac = _build_extraction_mesh(
                shape_u, self.shape, modeord=1
            )
            rendering_fft = fac * rendering_fft[indices_z, indices_y, indices_x]
        return (
            jnp.fft.ifftn(rendering_fft).real
            if outputs_real_space
            else _prepare_fft_to_fft(
                rendering_fft,
                outputs_rfft=outputs_rfft,
                fftshifted=fftshifted,
            )
        )


class GaussianFourierProjection(
    AbstractVolumeIntegrator[GaussianFourierVolume],
    strict=True,
):
    """Integrate atomic parametrization of a volume onto the exit plane from
    a `GaussianFourierVolume` using non-uniform FFTs and Fourier-domain
    convolution. Good when kernels span at least a couple pixels; see
    `cryojax.simulator.GaussianMixtureProjection` for an alternative that
    directly spreads narrow kernels onto the grid.

    !!! info
        By default, the non-uniform FFT runs on a pure-JAX backend using
        [`nufftax`](https://github.com/GragasLab/nufftax/tree/custom-kernel-spread).
        Setting the environment variable `CRYOJAX_FINUFFT_BACKEND=jax-finufft` switches to
        [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft), which
        can be more computationally efficient and less memory-demanding, at the
        cost of being trickier to install and having more limited integration
        with multi-GPU JAX.
    """  # noqa: E501

    sampling_mode: Literal["average", "point"]
    upsample_factor: float
    eps: float
    shape: tuple[int, int] | None
    options: dict[str, Any]

    outputs_ewald_sphere: ClassVar[bool] = False

    def __init__(
        self,
        *,
        sampling_mode: Literal["average", "point"] = "average",
        upsample_factor: int | float = 1.0,
        eps: float = 1e-6,
        shape: tuple[int, int] | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `sampling_mode`:
            If `'average'`, convolve with a box function to sample the
            projected volume at a pixel to be the average value of the
            underlying continuous function. If `'point'`, the volume at
            a pixel will be point sampled.
        - `upsample_factor`:
            How much to upsample the grid on which atoms are spread onto.
        - `eps`:
            The precision of the underlying non-uniform FFT implementation.
            See [`finufft`](https://finufft.readthedocs.io/en/latest/opts.html#options-parameters-cpu)
            for documentation.
        - `shape`:
            If given, first compute the image at `shape`, then
            pad or crop to `image_config.padded_shape`.
        - `options`:
            A dictionary of options for advanced usage, passed directly to
            the underlying non-uniform FFT implementation.
        """  # noqa: E501
        if sampling_mode not in ["average", "point"]:
            raise ValueError(
                "`sampling_mode` in `GaussianFourierProjection` "
                "must be either 'average' for averaging within a "
                "pixel or 'point' for point sampling. Got "
                f"`sampling_mode = {sampling_mode}`."
            )
        self.sampling_mode = sampling_mode
        self.shape = shape
        self.upsample_factor = float(upsample_factor)
        self.eps = eps
        self.options = options

    @override
    def integrate(
        self,
        volume_representation: GaussianFourierVolume,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionArray:
        """Compute a projection from scattering factors per atom type
        from the `GaussianFourierVolume`.

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
        pixel_size = jnp.asarray(image_config.pixel_size)
        shape = image_config.padded_shape if self.shape is None else self.shape
        output_shape = image_config.padded_shape
        positions, kernel_fns = (
            volume_representation.positions,
            volume_representation.kernel_fns,
        )
        kernel_fns = _standardize_kernel_fns(kernel_fns, spatial_dim=2)
        (shape_u, pixel_size_u), upsampfac = _prepare_upsample(
            shape=shape,
            pixel_size=pixel_size,
            upsampfac=self.upsample_factor,
        )
        # Compute projection
        projection_fft = _project_impl(
            positions,
            kernel_fns,
            shape_u=cast(tuple[int, int], shape_u),
            pixel_size_u=pixel_size_u,
            shape_out=cast(tuple[int, int], shape),
            eps=self.eps,
            options=self.options,
        )
        # Average within a pixel size
        if self.sampling_mode == "average":
            box_fn = FourierSinc(box_width=pixel_size)
            frequencies_1d = make_frequencies_1d(shape_u, pixel_size_u, modeord=1)
            projection_fft *= _eval_separable_impl(box_fn, frequencies_1d)
        # Downsample by extracting fourier modes
        if shape != shape_u:
            (indices_y, indices_x), fac = _build_extraction_mesh(
                shape_u, shape, modeord=1
            )
            projection_fft = fac * projection_fft[indices_y, indices_x]
        projection_fft = convert_fftn_to_rfftn(projection_fft, mode="real")
        if output_shape == shape:
            return (
                jnp.fft.irfftn(projection_fft, s=output_shape)
                if outputs_real_space
                else projection_fft
            )
        else:
            projection = jnp.fft.irfftn(projection_fft, s=shape)
            projection = resize_with_crop_or_pad(projection, output_shape)
            return projection if outputs_real_space else jnp.fft.rfftn(projection)


def _standardize_kernel_fns(
    kernel_pytree: PyTree[FourierGaussian], *, spatial_dim: int
) -> PyTree[FourierGaussian]:
    kernel_list = jax.tree.leaves(
        kernel_pytree,
        is_leaf=lambda x: isinstance(x, AbstractFourierOperator),
    )
    if not all(isinstance(kernel, FourierGaussian) for kernel in kernel_list):
        raise ValueError(
            "Found that `GaussianFourierVolume.kernel_fns` was not a "
            "PyTree containing only `FourierGaussian`s."
        )
    # Standardize gaussian kernels for computation
    # ... pytree leaves have a batch dim
    kernel_pytree = jax.tree.map(lambda x: jnp.atleast_1d(x), kernel_pytree)
    # ... `eval_separable_impl` embeds the kernel's amplitude once per axis
    # (outer product), so it must be pre-split by the `spatial_dim`-th root
    # to recover the correct total amplitude.
    replace_fn = lambda fn: eqx.tree_at(
        lambda _fn: _fn.amplitude, fn, (fn.amplitude ** (1 / spatial_dim))
    )
    kernel_pytree = jax.tree.map(
        replace_fn,
        kernel_pytree,
        is_leaf=lambda x: isinstance(x, FourierGaussian),
    )
    return kernel_pytree


def _project_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[FourierGaussian],
    *,
    shape_u: tuple[int, int],
    pixel_size_u: Float[Array, ""],
    shape_out: tuple[int, int],
    eps: float,
    options: dict[str, Any],
) -> Array:
    is_leaf = lambda x: isinstance(x, AbstractFourierOperator)  # noqa: E731
    frequencies_1d = make_frequencies_1d(shape_u, pixel_size_u, modeord=0)

    # Per-dimension NUFFT offset: 2*pi*(N//2)/N maps physical center (x=0) to
    # integer pixel index N//2.  For even N this equals pi; for odd N it is
    # pi*(1 - 1/N).  Must use the original output shape, not the upsampled one.
    _nufft_offsets_2d = jnp.asarray(
        [2 * jnp.pi * (s // 2) / s for s in shape_out[::-1][:2]]
    )

    def fourier_impl(
        _positions: Float[Array, "_ 3"],
        _kernel_fn: FourierGaussian,
    ) -> Array:
        _ns = jnp.asarray(shape_u[::-1][:2], dtype=float)
        xy = 2 * jnp.pi * _positions[:, :2] / (pixel_size_u * _ns) + _nufft_offsets_2d
        return (
            _eval_kernel_impl(_kernel_fn, frequencies_1d)
            * _nufft2d1(
                shape_u,
                source=jnp.ones(_positions.shape[0], dtype=complex),
                xy=xy,
                eps=eps,
                options=options,
            )
            / pixel_size_u**2
        )

    # Project and sum over kernels
    projection_out = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(fourier_impl, positions, kernel_fns, is_leaf=is_leaf),
    )
    return jnp.fft.ifftshift(projection_out)


def _render_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[FourierGaussian],
    *,
    shape_u: tuple[int, int, int],
    voxel_size_u: Float[Array, ""],
    shape_out: tuple[int, int, int],
    eps: float,
    options: dict[str, Any],
) -> Array:
    is_leaf = lambda x: isinstance(x, AbstractFourierOperator)  # noqa: E731
    frequencies_1d = make_frequencies_1d(shape_u, voxel_size_u, modeord=0)

    _nufft_offsets_3d = jnp.asarray([2 * jnp.pi * (s // 2) / s for s in shape_out[::-1]])

    def fourier_impl(
        _positions: Float[Array, "_ 3"],
        _kernel_fn: FourierGaussian,
    ) -> Array:
        _ns = jnp.asarray(shape_u[::-1], dtype=float)
        xyz = 2 * jnp.pi * _positions / (voxel_size_u * _ns) + _nufft_offsets_3d
        return _eval_kernel_impl(_kernel_fn, frequencies_1d) * (
            _nufft3d1(
                shape_u,
                source=jnp.ones(_positions.shape[0], dtype=complex),
                xyz=xyz,
                eps=eps,
                options=options,
            )
            / voxel_size_u**3
        )

    rendering_out = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(fourier_impl, positions, kernel_fns, is_leaf=is_leaf),
    )
    return jnp.fft.ifftshift(rendering_out)


def _eval_kernel_impl(kernel_fn: FourierGaussian, frequencies_1d: tuple[Array, ...]):
    a, b = kernel_fn.amplitude, kernel_fn.b_factor
    make_gaussians = jax.vmap(lambda _a, _b: FourierGaussian(amplitude=_a, b_factor=_b))
    eval_gaussians = jax.vmap(_eval_separable_impl, in_axes=(0, None))
    return jnp.sum(eval_gaussians(make_gaussians(a, b), frequencies_1d), axis=0)


def _eval_separable_impl(
    kernel_fn: FourierGaussian | FourierSinc, frequencies_1d: tuple[Array, ...]
):
    ndim = len(frequencies_1d)
    assert 1 in kernel_fn.spatial_dims and ndim in [2, 3]
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
    eps: float,
    options: dict[str, Any],
):
    default_upsampfac = 1.25
    if CRYOJAX_FINUFFT_BACKEND == "jax-finufft":
        if jax_finufft is None:
            raise RuntimeError(
                "Tried to use the `jax-finufft` non-uniform FFT backend "
                "(set via the `CRYOJAX_FINUFFT_BACKEND` environment "
                "variable), but `jax-finufft` is not installed. "
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
            n_modes=shape[::-1],  # type: ignore
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
    eps: float,
    options: dict[str, Any],
):
    default_upsampfac = 1.25
    if CRYOJAX_FINUFFT_BACKEND == "jax-finufft":
        if jax_finufft is None:
            raise RuntimeError(
                "Tried to use the `jax-finufft` non-uniform FFT backend "
                "(set via the `CRYOJAX_FINUFFT_BACKEND` environment "
                "variable), but `jax-finufft` is not installed. "
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
            n_modes=shape[::-1],  # type: ignore
            c=source,
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            eps=eps,
            isign=-1,
            upsampfac=upsampfac,
            **options,
        )


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


def _prepare_upsample(
    shape: tuple[int, ...],
    pixel_size: Float[Array, ""],
    upsampfac: float,
) -> tuple[tuple[tuple[int, ...], Array], float]:
    """Find the upsampfac that perfectly divides the upsampled
    image shape.
    """
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
            shape_u = query_efficient_grid_size(shape, upsampfac)
            dim_u = shape_u[0]
            upsampfac = dim_u / dim
        return (
            (tuple(int(upsampfac * s) for s in shape), pixel_size / upsampfac),
            upsampfac,
        )


def _prepare_fft_to_fft(f: Array, *, outputs_rfft: bool, fftshifted: bool) -> Array:
    if outputs_rfft:
        f = convert_fftn_to_rfftn(f, mode="real")
        return jnp.fft.fftshift(f, axes=(0, 1)) if fftshifted else f
    else:
        return jnp.fft.fftshift(f) if fftshifted else f
