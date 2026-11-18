from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, Self, TypeVar, cast
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from ..._internal import leaf_asarray
from ...constants import (
    PengScatteringFactorParameters,
)
from ...jax_util import FloatLike, NDArrayLike
from ...ndimage import (
    AbstractFourierOperator,
    FourierGaussian,
    convert_rfftn_to_fftn,
    dispatch_nufft1,
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
from .internal import check_anisotropy_supported, eval_separable_impl, make_frequencies_1d


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

    The arguments `positions`, `kernel_fns` and `amplitudes` may also be
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
    amplitudes: PyTree[Float[NDArrayLike, " _"]]

    is_frame_rotation: ClassVar[bool] = False

    def __init__(
        self,
        positions: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        kernel_fns: PyTree[FourierGaussian, "T"],
        amplitudes: PyTree[Float[NDArrayLike, " _"], "T"] | None = None,
    ):
        """**Arguments:**

        - `positions`:
            A pytree of atom positions.
        - `kernel_fns`:
            A pytree of functions with the same tree structure
            as `positions`, where each leaf is a
            [`cryojax.ndimage.FourierGaussian`][] representing the atom
            type's scattering factor. These may have amplitudes and
            b-factors with a batch dimension to simulate form factors.
        - `amplitudes`:
            A pytree with the same tree structure as `positions`, each leaf
            of shape `(M,)`: one amplitude per atom, scaling its type's
            scattering factor. Ones if not given.
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
        if amplitudes is not None and (
            jax.tree.structure(positions) != jax.tree.structure(amplitudes)
        ):
            raise ValueError(
                "When instantiating a `GaussianFourierVolume`, found "
                "that the pytree structures of `positions` and "
                "`amplitudes` were not equal."
            )
        # Convert positions to array leaves, preserving their backend (JAX
        # arrays stay on-device, NumPy arrays stay on the host).
        self.positions = jax.tree.map(lambda x: leaf_asarray(x, dtype=float), positions)
        self.kernel_fns = kernel_fns
        if amplitudes is None:
            amplitudes = jax.tree.map(lambda x: np.ones(x.shape[0]), self.positions)
        self.amplitudes = jax.tree.map(lambda x: leaf_asarray(x, dtype=float), amplitudes)
        for pos, amp in zip(
            jax.tree.leaves(self.positions), jax.tree.leaves(self.amplitudes)
        ):
            if amp.shape != (pos.shape[0],):
                raise ValueError(
                    "When instantiating a `GaussianFourierVolume`, found an "
                    f"`amplitudes` leaf of shape {amp.shape} for {pos.shape[0]} "
                    "positions. Each leaf must have shape `(M,)`, where `M` is "
                    "the number of positions in the matching leaf of `positions`."
                )

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
    and Fourier-domain convolution. Computes the band-limited voxel grid, whose
    spectrum is the volume's at every frequency of the grid. Good when kernels span
    at least a couple pixels; see `cryojax.simulator.GaussianMixtureRenderFn` for an
    alternative that directly spreads narrow kernels onto the grid, and for the
    voxel average.

    !!! info
        By default, the non-uniform FFT runs on cryoJAX's native backend, in pure JAX or
        Pallas. Setting the environment variable `CRYOJAX_FINUFFT_BACKEND=jax-finufft`
        switches to [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft).
    """  # noqa: E501

    shape: tuple[int, int, int]
    voxel_size: Float[NDArrayLike, "..."]

    sampling_mode: Literal["point"] = eqx.field(static=True)
    eps: float
    upsample_factor: float
    enable_pallas: bool | Mapping[str, bool] | None
    options: dict[str, Any]

    def __init__(
        self,
        shape: tuple[int, int, int],
        voxel_size: FloatLike,
        *,
        eps: float = 1e-6,
        upsample_factor: float = 1.25,
        enable_pallas: bool | Mapping[str, bool] | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `eps`:
            The requested relative accuracy of the non-uniform FFT.
        - `upsample_factor`:
            The ratio of the fine grid to the grid (finufft's `upsampfac`), greater
            than 1. Smaller values use less memory but a wider kernel.
        - `enable_pallas`:
            Whether the native backend spreads with Pallas kernels. See
            [`cryojax.ndimage.dispatch_nufft1`][].
        - `options`:
            A dictionary of options for the `jax-finufft` backend only.
        """  # noqa: E501
        self.shape = shape
        self.voxel_size = leaf_asarray(voxel_size, dtype=float)
        self.sampling_mode = "point"
        self.eps = eps
        self.upsample_factor = _check_upsample_factor(
            "GaussianFourierRenderFn", upsample_factor
        )
        self.enable_pallas = enable_pallas
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
        # Prepare arguments, casting positions, amplitudes and voxel size to JAX arrays
        voxel_size = jnp.asarray(self.voxel_size)
        positions, kernel_fns, amplitudes = (
            jax.tree.map(lambda x: jnp.asarray(x), volume_representation.positions),
            volume_representation.kernel_fns,
            jax.tree.map(lambda x: jnp.asarray(x), volume_representation.amplitudes),
        )
        kernel_fns = _standardize_kernel_fns(kernel_fns, spatial_dim=3)
        # Compute
        rendering_rfft = _render_impl(
            positions,
            kernel_fns,
            amplitudes,
            shape=self.shape,
            voxel_size=voxel_size,
            eps=self.eps,
            upsample_factor=self.upsample_factor,
            enable_pallas=self.enable_pallas,
            options=self.options,
        )
        if outputs_real_space:
            return jnp.fft.irfftn(rendering_rfft, s=self.shape)
        return _prepare_fft_to_fft(
            rendering_rfft,
            self.shape,
            outputs_rfft=outputs_rfft,
            fftshifted=fftshifted,
        )


class GaussianFourierProjection(
    AbstractVolumeIntegrator[GaussianFourierVolume],
    strict=True,
):
    """Integrate atomic parametrization of a volume onto the exit plane from
    a `GaussianFourierVolume` using non-uniform FFTs and Fourier-domain
    convolution. Computes the band-limited image, whose spectrum is the projected
    volume's at every frequency of the grid. Good when kernels span at least a
    couple pixels; see `cryojax.simulator.GaussianMixtureProjection` for an
    alternative that directly spreads narrow kernels onto the grid, and for the
    pixel average.

    !!! info
        By default, the non-uniform FFT runs on cryoJAX's native backend, in pure JAX or
        Pallas. Setting the environment variable `CRYOJAX_FINUFFT_BACKEND=jax-finufft`
        switches to [`jax-finufft`](https://github.com/flatironinstitute/jax-finufft).
    """  # noqa: E501

    sampling_mode: Literal["point"] = eqx.field(static=True)
    eps: float
    upsample_factor: float
    shape: tuple[int, int] | None
    enable_pallas: bool | Mapping[str, bool] | None
    options: dict[str, Any]

    outputs_ewald_sphere: ClassVar[bool] = False
    supports_anisotropy: ClassVar[bool] = False

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        upsample_factor: float = 1.25,
        shape: tuple[int, int] | None = None,
        enable_pallas: bool | Mapping[str, bool] | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `eps`:
            The requested relative accuracy of the non-uniform FFT.
        - `upsample_factor`:
            The ratio of the fine grid to the grid (finufft's `upsampfac`), greater
            than 1. Smaller values use less memory but a wider kernel.
        - `enable_pallas`:
            Whether the native backend spreads with Pallas kernels. See
            [`cryojax.ndimage.dispatch_nufft1`][].
        - `shape`:
            If given, first compute the image at `shape`, then
            pad or crop to `image_config.padded_shape`.
        - `enable_pallas`:
            Whether the native backend spreads with Pallas kernels. See
            [`cryojax.ndimage.dispatch_nufft1`][].
        - `options`:
            A dictionary of options for the `jax-finufft` backend only.
        """  # noqa: E501
        self.sampling_mode = "point"
        self.shape = shape
        self.eps = eps
        self.upsample_factor = _check_upsample_factor(
            "GaussianFourierProjection", upsample_factor
        )
        self.enable_pallas = enable_pallas
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
        check_anisotropy_supported(self, image_config)
        # Prepare arguments
        pixel_size = jnp.asarray(image_config.pixel_size)
        shape = image_config.padded_shape if self.shape is None else self.shape
        output_shape = image_config.padded_shape
        positions, kernel_fns, amplitudes = (
            volume_representation.positions,
            volume_representation.kernel_fns,
            volume_representation.amplitudes,
        )
        kernel_fns = _standardize_kernel_fns(kernel_fns, spatial_dim=2)
        # Compute projection
        projection_rfft = _project_impl(
            positions,
            kernel_fns,
            amplitudes,
            shape=cast(tuple[int, int], shape),
            pixel_size=pixel_size,
            eps=self.eps,
            upsample_factor=self.upsample_factor,
            enable_pallas=self.enable_pallas,
            options=self.options,
        )
        if output_shape == shape:
            return (
                jnp.fft.irfftn(projection_rfft, s=output_shape)
                if outputs_real_space
                else projection_rfft
            )
        else:
            projection = jnp.fft.irfftn(projection_rfft, s=shape)
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
    amplitudes: PyTree[Float[Array, " _"]],
    *,
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    eps: float,
    upsample_factor: float,
    enable_pallas: bool | Mapping[str, bool] | None,
    options: dict[str, Any],
) -> Array:
    is_leaf = lambda x: isinstance(x, AbstractFourierOperator)  # noqa: E731
    frequencies_1d = make_frequencies_1d(shape, pixel_size, modeord=1, outputs_rfft=True)

    def fourier_impl(
        _positions: Float[Array, "_ 3"],
        _kernel_fn: FourierGaussian,
        _amplitudes: Float[Array, " _"],
    ) -> Array:
        spectrum = dispatch_nufft1(
            _positions[:, :2],
            jnp.asarray(_amplitudes, dtype=float),
            shape,
            pixel_size=pixel_size,
            eps=eps,
            upsample_factor=upsample_factor,
            enable_pallas=enable_pallas,
            options=options,
        )
        return _eval_kernel_impl(_kernel_fn, frequencies_1d) * spectrum / pixel_size**2

    # Project and sum over kernels
    return jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(fourier_impl, positions, kernel_fns, amplitudes, is_leaf=is_leaf),
    )


def _render_impl(
    positions: PyTree[Float[Array, "_ 3"]],
    kernel_fns: PyTree[FourierGaussian],
    amplitudes: PyTree[Float[Array, " _"]],
    *,
    shape: tuple[int, int, int],
    voxel_size: Float[Array, ""],
    eps: float,
    upsample_factor: float,
    enable_pallas: bool | Mapping[str, bool] | None,
    options: dict[str, Any],
) -> Array:
    is_leaf = lambda x: isinstance(x, AbstractFourierOperator)  # noqa: E731
    frequencies_1d = make_frequencies_1d(shape, voxel_size, modeord=1, outputs_rfft=True)

    def fourier_impl(
        _positions: Float[Array, "_ 3"],
        _kernel_fn: FourierGaussian,
        _amplitudes: Float[Array, " _"],
    ) -> Array:
        spectrum = dispatch_nufft1(
            _positions,
            jnp.asarray(_amplitudes, dtype=float),
            shape,
            pixel_size=voxel_size,
            eps=eps,
            upsample_factor=upsample_factor,
            enable_pallas=enable_pallas,
            options=options,
        )
        return _eval_kernel_impl(_kernel_fn, frequencies_1d) * spectrum / voxel_size**3

    return jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(fourier_impl, positions, kernel_fns, amplitudes, is_leaf=is_leaf),
    )


def _check_upsample_factor(name: str, upsample_factor: float) -> float:
    if upsample_factor <= 1:
        raise ValueError(
            f"Found invalid value for `{name}.upsample_factor`. It must be greater "
            f"than 1, but got `upsample_factor = {upsample_factor}`."
        )
    return upsample_factor


def _eval_kernel_impl(kernel_fn: FourierGaussian, frequencies_1d: tuple[Array, ...]):
    a, b = kernel_fn.amplitude, kernel_fn.b_factor
    make_gaussians = jax.vmap(lambda _a, _b: FourierGaussian(amplitude=_a, b_factor=_b))
    eval_gaussians = jax.vmap(eval_separable_impl, in_axes=(0, None))
    return jnp.sum(eval_gaussians(make_gaussians(a, b), frequencies_1d), axis=0)


def _prepare_fft_to_fft(
    f: Array, shape: tuple[int, ...], *, outputs_rfft: bool, fftshifted: bool
) -> Array:
    if outputs_rfft:
        return jnp.fft.fftshift(f, axes=(0, 1)) if fftshifted else f
    f = convert_rfftn_to_fftn(f, shape)
    return jnp.fft.fftshift(f) if fftshifted else f
