import pathlib
from collections.abc import Callable
from typing import Any, Literal, overload

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import mmdf
import numpy as np
import pandas as pd
from jaxtyping import Bool

from ..atom_util import split_atoms_by_element
from ..constants import PengScatteringFactorParameters
from ..io import mmdf_to_atoms
from ..jax_util import FloatLike, NDArrayLike
from ..ndimage import (
    AbstractFourierOperator,
    AbstractImageTransform,
    AbstractRealOperator,
    RealGaussian,
    make_coordinate_grid,
    variance_to_nspread,
)
from ._detector import AbstractDetector
from ._image_config import AbstractImageConfig, DoseImageConfig
from ._image_model import (
    AbstractImageModel,
    ContrastImageModel,
    ElectronCountsImageModel,
    IntensityImageModel,
    LinearImageModel,
    ProjectionImageModel,
)
from ._pose import AbstractPose
from ._scattering_theory import WeakPhaseScatteringTheory
from ._transfer_theory import ContrastTransferTheory
from ._volume import (
    AbstractAtomVolume,
    AbstractVolumeIntegrator,
    AbstractVolumeParametrization,
    AbstractVolumeRenderFn,
    AutoVolumeProjection,
    FourierVoxelGridVolume,
    FourierVoxelSplineVolume,
    GaussianMixtureVolume,
    IndependentAtomVolume,
    RealVoxelCloudVolume,
    RealVoxelGridVolume,
)


identity_fn = eqxi.doc_repr(lambda x, _: x, "identity_fn")


@overload
def make_image_model(
    volume: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: None = None,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    image_transform: AbstractImageTransform | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    signal_centering: Literal["bg", "mean"] = "mean",
    translate_mode: Literal["fft", "atom", "none"] = "fft",
    quantity_mode: Literal["none"] = "none",
) -> ProjectionImageModel: ...


@overload
def make_image_model(  # pyright: ignore[reportOverlappingOverload]
    volume: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    image_transform: AbstractImageTransform | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    signal_centering: Literal["bg", "mean"] = "mean",
    translate_mode: Literal["fft", "atom", "none"] = "fft",
    quantity_mode: Literal["none"] = "none",
) -> LinearImageModel: ...


@overload
def make_image_model(
    volume: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    image_transform: AbstractImageTransform | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    signal_centering: Literal["bg", "mean"] = "mean",
    translate_mode: Literal["fft", "atom", "none"] = "fft",
    quantity_mode: Literal["contrast"] = "contrast",
) -> ContrastImageModel: ...


@overload
def make_image_model(
    volume: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    image_transform: AbstractImageTransform | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    signal_centering: Literal["bg", "mean"] = "mean",
    translate_mode: Literal["fft", "atom", "none"] = "fft",
    quantity_mode: Literal["intensity"] = "intensity",
) -> IntensityImageModel: ...


@overload
def make_image_model(
    volume: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    image_transform: AbstractImageTransform | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    signal_centering: Literal["bg", "mean"] = "mean",
    translate_mode: Literal["fft", "atom", "none"] = "fft",
    quantity_mode: Literal["counts"] = "counts",
) -> ElectronCountsImageModel: ...


def make_image_model(
    volume: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory | None = None,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    image_transform: AbstractImageTransform | None = None,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    signal_centering: Literal["bg", "mean"] = "mean",
    translate_mode: Literal["fft", "atom", "none"] = "fft",
    quantity_mode: Literal["contrast", "intensity", "counts", "none"] = "none",
) -> AbstractImageModel:
    """Construct an [`cryojax.simulator.AbstractImageModel`][] for
    most common use-cases.

    !!! example "Simulate an image"

        ```python
        import cryojax.simulator as cxs

        # Load modeling components
        volume, image_config, pose, transfer_theory = ...
        # Build image formation model
        image_model = cxs.make_image_model(volume, image_config, pose, transfer_theory)
        # Simulate!
        image = image_model.simulate()
        ```

    **Main arguments:**

    - `volume`:
        The volume for imaging. To get started building volumes:
        - See [`cryojax.simulator.load_tabulated_volume`][] for instantiating
        atomistic volumes
        - See [`cryojax.simulator.render_voxel_volume`][] for converting them
        to voxel maps
        - Explore cryoJAX's [`cryojax.simulator.AbstractVolumeRepresentation`][]
        classes; instantiate these directly for more flexibility
        - Advanced users may also be interested in implementing
        [`cryojax.simulator.AbstractVolumeParametrization`][] classes.
    - `image_config`:
        The configuration for the image and imaging instrument. Unless using
        a model that uses the electron dose as a parameter, choose the
        [`cryojax.simulator.BasicImageConfig`][]. Otherwise, choose the
        [`cryojax.simulator.DoseImageConfig`][].
    - `pose`:
        The pose in a particular parameterization convention. Common options
        are the [`cryojax.simulator.EulerAnglePose`][],
        [`cryojax.simulator.QuaternionPose`][], or
        [`cryojax.simulator.AxisAnglePose`][].
    - `transfer_theory`:
        The contrast transfer function and its theory for how it is applied
        to the image. See [`cryojax.simulator.ContrastTransferTheory`][].
    - `volume_integrator`:
        Optionally, pass the method for integrating the electrostatic potential onto
        the plane (e.g. projection via fourier slice extraction). If not provided,
        a default option is chosen using the
        [`cryojax.simulator.AutoVolumeProjection`][] class.
    - `detector`:
        If `quantity_mode = 'counts'` is chosen, then an
        [`cryojax.simulator.AbstractDetector`][] class must be chosen to
        simulate electron counts.

    **Options:**

    - `image_transform`:
        A [`cryojax.ndimage.AbstractImageTransform`][] applied to the
        the output of `image_model.simulate()` as a postprocessing step.
    - `normalizes_signal`:
        Whether or not to normalize the output of `image_model.simulate()`.
        If `True`, see `signal_centering` for options.
    - `signal_region`:
        A boolean array that is 1 where there is signal,
        and 0 otherwise used to normalize the image.
        Must have shape equal to `image_config.shape`.
    - `signal_centering`:
        How to calculate the offset for normalization when
        `normalizes_signal = True`
        (and ignored if `normalizes_signal = False`).
        Options are
        - 'mean':
            Normalize the image to be mean 0
            within `signal_region`. This normalizes the image
            to be a z-score.
        - 'bg':
            Subtract mean value at the image edges.
            This makes the image fade to a background with values
            equal to zero. Requires that `image_config.padded_shape`
            is large enough so that the signal sufficiently decays.
    - `translate_mode`:
        How to apply in-plane translation to the volume. Options are
        - 'fft':
            Apply phase shifts in the Fourier domain. This option
            is best for most use cases and is usually faster than
            the `'atom'` option.
        - 'atom':
            Apply translation to atom positions before
            projection. This method is more numerically accurate
            than the `'fft'` option, but it is only supported if
            the `volume` argument yields a
            [`cryojax.simulator.AbstractAtomVolume`][].
        - 'none':
            Do not apply the translation.
    - `quantity_mode`:
        The physical observable to simulate. Options are:
        - 'none':
            Use the [`cryojax.simulator.LinearImageModel`][]. This
            simulates without scaling to physical units.
        - 'contrast':
            Uses the [`cryojax.simulator.ContrastImageModel`][]
            to simulate contrast.
        - 'intensity':
            Uses the [`cryojax.simulator.IntensityImageModel`][]
            to simulate intensity.
        - 'counts':
            Uses the [`cryojax.simulator.ElectronCountsImageModel`][]
            to simulate electron counts.
            If this is passed, a `detector` must also be passed.

    **Returns:**

    An [`cryojax.simulator.AbstractImageModel`][]. This has type:

    - [`cryojax.simulator.ProjectionImageModel`][] if no `transfer_theory`
    is specified.
    - [`cryojax.simulator.LinearImageModel`][] if a `transfer_theory`
    is specified and `quantity_mode = None`.
    - [`cryojax.simulator.ContrastImageModel`][],
    [`cryojax.simulator.IntensityImageModel`][], or
    [`cryojax.simulator.ElectronCountsImageModel`][] depending on
    the value of `quantity_mode`.
    """
    options = dict(
        normalizes_signal=normalizes_signal,
        signal_centering=signal_centering,
        signal_region=signal_region,
        translate_mode=translate_mode,
        image_transform=image_transform,
    )
    if transfer_theory is None:
        # Image model for projections
        image_model = ProjectionImageModel(
            volume,
            pose,
            image_config,
            volume_integrator,
            **options,  # pyright: ignore[reportArgumentType]
        )
    else:
        # Simulate physical observables
        if quantity_mode == "none":
            # Linear image model
            image_model = LinearImageModel(
                volume,
                pose,
                image_config,
                transfer_theory,
                volume_integrator,
                **options,  # pyright: ignore[reportArgumentType]
            )
        else:
            scattering_theory = WeakPhaseScatteringTheory(
                volume_integrator, transfer_theory
            )
            if quantity_mode == "counts":
                if not isinstance(image_config, DoseImageConfig):
                    raise ValueError(
                        "If using `quantity_mode = 'counts'` to simulate electron "
                        "counts, pass `image_config = DoseImageConfig(...)`. Got config "
                        f"{type(image_config).__name__}."
                    )
                if detector is None:
                    raise ValueError(
                        "If using `quantity_mode = 'counts'` to simulate electron "
                        "counts, an `AbstractDetector` must be passed."
                    )
                image_model = ElectronCountsImageModel(
                    volume,
                    pose,
                    image_config,
                    scattering_theory,
                    detector,
                    **options,  # pyright: ignore[reportArgumentType]
                )
            elif quantity_mode == "contrast":
                image_model = ContrastImageModel(
                    volume,
                    pose,
                    image_config,
                    scattering_theory,
                    **options,  # pyright: ignore[reportArgumentType]
                )
            elif quantity_mode == "intensity":
                image_model = IntensityImageModel(
                    volume,
                    pose,
                    image_config,
                    scattering_theory,
                    **options,  # pyright: ignore[reportArgumentType]
                )
            else:
                raise ValueError(
                    f"Found `quantity_mode = {quantity_mode}`, but valid "
                    "values are 'contrast', 'intensity', 'counts', or 'none'."
                )

    return image_model


@overload
def load_tabulated_volume(  # pyright: ignore[reportOverlappingOverload]
    path_or_mmdf: str | pathlib.Path | pd.DataFrame,
    *,
    output_type: type[IndependentAtomVolume] = IndependentAtomVolume,
    tabulation: Literal["peng"] = "peng",
    include_b_factors: bool = True,
    b_factor_fn: Callable[[NDArrayLike, NDArrayLike], NDArrayLike] = identity_fn,
    selection_string: str = "all",
    pdb_options: dict[str, Any] = {},
) -> IndependentAtomVolume: ...


@overload
def load_tabulated_volume(
    path_or_mmdf: str | pathlib.Path | pd.DataFrame,
    *,
    output_type: type[GaussianMixtureVolume] = GaussianMixtureVolume,
    tabulation: Literal["peng"] = "peng",
    include_b_factors: bool = True,
    b_factor_fn: Callable[[NDArrayLike, NDArrayLike], NDArrayLike] = identity_fn,
    selection_string: str = "all",
    pdb_options: dict[str, Any] = {},
) -> GaussianMixtureVolume: ...


def load_tabulated_volume(
    path_or_mmdf: str | pathlib.Path | pd.DataFrame,
    *,
    output_type: type[
        IndependentAtomVolume | GaussianMixtureVolume
    ] = GaussianMixtureVolume,
    tabulation: Literal["peng"] = "peng",
    include_b_factors: bool = False,
    b_factor_fn: Callable[[NDArrayLike, NDArrayLike], NDArrayLike] = identity_fn,
    selection_string: str = "all",
    pdb_options: dict[str, Any] = {},
) -> IndependentAtomVolume | GaussianMixtureVolume:
    """Load an atomistic representation of a volume from
    tabulated electron scattering factors.

    !!! warning
        This function cannot be used with JIT compilation.
        Rather, its output should be passed to JIT-compiled
        functions. For example:

        ```python
        import cryojax.simulator as cxs
        import equinox as eqx

        path_to_pdb = ...
        volume = cxs.load_tabulated_volume(path_to_pdb)

        @eqx.filter_jit
        def simulate_fn(volume, ...):
            image_model = cxs.make_image_model(volume, ...)
            return image_model.simulate()

        image = simulate_fn(volume, ...)
        ```

    **Arguments:**

    - `path_or_mmdf`:
        The path to the PDB/PDBx file or a `pandas.DataFrame` loaded
        from [`mmdf.read`](https://github.com/teamtomo/mmdf).
    - `output_type`:
        Either a [`cryojax.simulator.GaussianMixtureVolume`][] or
        [`cryojax.simulator.IndependentAtomVolume`][] class.
    - `tabulation`:
        Specifies which electron scattering factor tabulation to use.
        The only supported value current is `tabulation = 'peng'`.
        See [`cryojax.constants.PengScatteringFactorParameters`][]
        for more information.
    - `include_b_factors`:
        If `True`, include PDB B-factors in the volume.
    - `b_factor_fn`:
        A function that modulates PDB B-factors before passing to the
        volume. Has signature
        `new_b_factor = b_factor_fn(b_factor, atomic_number)`.
        If `output_type = IndependentAtomVolume`, `b_factor` is
        the mean B-factor for a given atom type.
    - `selection_string`:
        A string for [`mdtraj` atom selection](https://mdtraj.org/1.9.4/examples/atom-selection.html#atom-selection).
        See [`cryojax.io.read_atoms_from_pdb`][] for documentation.
    - `pdb_options`:
        Additional keyword options passed to [`cryojax.io.read_atoms_from_pdb`][],
        not including `selection_string`.

    **Returns:**

    A [`cryojax.simulator.AbstractVoxelVolume`][] with exact type
    equal to `output_type`.
    """  # noqa: E501
    if isinstance(path_or_mmdf, (str, pathlib.Path)):
        atom_data = mmdf.read(pathlib.Path(path_or_mmdf))
    elif isinstance(path_or_mmdf, pd.DataFrame):
        atom_data = path_or_mmdf
    else:
        raise ValueError(
            "Argument `path_or_mmdf` to "
            "`load_tabulated_volume` was an unrecognized "
            "input type. Accepts a path to a PDB/PDBx file, "
            "or a pandas.DataFrame loaded from `mmdf.read`. "
            f"Instead, got type {path_or_mmdf.__class__.__name__}."
        )
    atom_positions, atomic_numbers, atom_properties = mmdf_to_atoms(
        atom_data,
        loads_properties=True,
        selection_string=selection_string,
        **pdb_options,
    )
    if output_type is GaussianMixtureVolume:
        if tabulation != "peng":
            raise ValueError(
                "Passed `output_type = GaussianMixtureVolume` to "
                "`load_tabulated_volume`, but found that "
                f"`tabulation = {tabulation}`, which "
                "is not a mixture of gaussians. Use "
                "`tabulation = 'peng'` instead."
            )
        peng_parameters = PengScatteringFactorParameters(atomic_numbers)
        b_factors = (
            jnp.asarray(
                b_factor_fn(atom_properties["b_factors"], atomic_numbers), dtype=float
            )
            if include_b_factors
            else None
        )
        atom_volume = GaussianMixtureVolume.from_tabulated_parameters(
            atom_positions, peng_parameters, extra_b_factors=b_factors
        )
    elif output_type is IndependentAtomVolume:
        (positions_by_id, b_factor_by_id), atom_ids = split_atoms_by_element(
            atomic_numbers, (atom_positions, atom_properties["b_factors"])
        )
        b_factor_by_id = tuple(
            jnp.asarray(b_factor_fn(jnp.mean(b), atom_ids)) for b in b_factor_by_id
        )
        if tabulation == "peng":
            parameters = PengScatteringFactorParameters(atom_ids)
        else:
            raise ValueError(
                "Only `tabulation` equal to 'peng' is supported in "
                f"`load_tabulated_volume`. Instead, got `tabulation = {tabulation}`."
            )
        atom_volume = IndependentAtomVolume.from_tabulated_parameters(
            positions_by_id, parameters, b_factor_by_element=b_factor_by_id
        )
    else:
        raise ValueError(
            "Only `output_type` equal to `GaussianMixtureVolume` "
            "or `IndependentAtomVolume` are supported."
        )

    return atom_volume


def suggest_n_spread(
    volume: IndependentAtomVolume | GaussianMixtureVolume,
    pixel_size: FloatLike,
    *,
    cutoff_sigma: float = 4.0,
    mode: Literal["global", "termwise"] = "termwise",
    min_n_spread: int = 1,
    max_n_spread: int | None = None,
) -> int | tuple[int, ...]:
    """Suggest `n_spread` value(s) for real-space gaussian spreading, sized
    to `volume`'s variances via [`cryojax.ndimage.variance_to_nspread`][].

    !!! warning "Usage"
        Call this on the concrete `volume`, before it enters `jax.jit`, and
        pass the result to the matching concrete integrator/render-fn --
        not `AutoVolumeProjection`/`AutoVolumeRenderFn`, which can't resolve
        `n_spread` from a traced volume and don't save you anything here
        anyway (you already need to know the volume's type to call this):

        ```python
        import jax
        import cryojax.simulator as cxs

        volume = cxs.load_tabulated_volume(
            path_to_pdb, output_type=cxs.GaussianMixtureVolume
        )
        n_spread = cxs.suggest_n_spread(volume, pixel_size)
        integrator = cxs.GaussianMixtureProjection(n_spread=n_spread)

        image_model = jax.jit(make_image_model)(
            volume=volume, volume_integrator=integrator, ...
        )
        ```

    **Arguments:**

    - `volume`: The volume to suggest `n_spread` for.
    - `pixel_size`: The pixel/voxel size of the grid `volume` is spread onto.
    - `cutoff_sigma`: Truncation width in standard deviations (`n_sigma` in
      `variance_to_nspread`).
    - `mode`: `"termwise"` (default) returns one `n_spread` per gaussian
      component, each sized to that component's largest variance (across
      positions, for `GaussianMixtureVolume`; across atom types, for
      `IndependentAtomVolume`). `"global"` returns a single `int` sized to
      the single widest gaussian in `volume`.
    - `min_n_spread`/`max_n_spread`: Clamp every returned value to this range.

    **Returns:**

    An `int` (`mode="global"`) or `tuple[int, ...]`, one per gaussian
    component (`mode="termwise"`).
    """
    if mode not in ("global", "termwise"):
        raise ValueError(
            f"`cryojax.simulator.suggest_n_spread(..., mode=...)` must be 'global' or "
            f"'termwise', but got `mode = '{mode}'`."
        )

    def _clamp(n: int) -> int:
        n = max(n, min_n_spread)
        if max_n_spread is not None:
            n = min(n, max_n_spread)
        return n

    if isinstance(volume, GaussianMixtureVolume):
        variances = np.asarray(volume.variances)  # (n_positions, n_gaussians)
        if mode == "global":
            return _clamp(
                variance_to_nspread(variances, pixel_size, n_sigma=cutoff_sigma)
            )
        n_gaussians = variances.shape[-1]
        return tuple(
            _clamp(variance_to_nspread(variances[:, i], pixel_size, n_sigma=cutoff_sigma))
            for i in range(n_gaussians)
        )
    elif isinstance(volume, IndependentAtomVolume):
        is_leaf = lambda x: isinstance(  # noqa: E731
            x, (AbstractRealOperator, AbstractFourierOperator)
        )
        kernel_leaves = jax.tree.leaves(volume.kernel_fns, is_leaf=is_leaf)
        # `(n_gaussians,)` variance per leaf, as plain numpy -- avoiding
        # `cryojax.constants.b_factor_to_variance`, which always returns a
        # JAX array, so that `suggest_n_spread` stays numpy/python-only.
        variance_leaves = [
            np.asarray(fn.variance)
            if isinstance(fn, RealGaussian)
            else np.asarray(fn.b_factor) / (8.0 * np.pi**2)
            for fn in kernel_leaves
        ]
        n_gaussians_set = {v.shape[-1] for v in variance_leaves}
        if len(n_gaussians_set) != 1:
            raise ValueError(
                "`suggest_n_spread` requires every `IndependentAtomVolume."
                "kernel_fns` leaf to have the same number of gaussian "
                f"components; got leaves with {sorted(n_gaussians_set)} "
                "gaussian components."
            )
        n_gaussians = n_gaussians_set.pop()
        if mode == "global":
            n_spread_per_leaf = [
                variance_to_nspread(v, pixel_size, n_sigma=cutoff_sigma)
                for v in variance_leaves
            ]
            return _clamp(max(n_spread_per_leaf))
        per_component = []
        for i in range(n_gaussians):
            n_spread_per_leaf_i = [
                variance_to_nspread(v[i], pixel_size, n_sigma=cutoff_sigma)
                for v in variance_leaves
            ]
            per_component.append(_clamp(max(n_spread_per_leaf_i)))
        return tuple(per_component)
    else:
        raise ValueError(
            "`cryojax.simulator.suggest_n_spread` only supports `volume` of type "
            "`GaussianMixtureVolume` or `IndependentAtomVolume`, but got "
            f"type `{type(volume).__name__}`."
        )


@overload
def render_voxel_volume(  # pyright: ignore[reportOverlappingOverload]
    atom_volume: AbstractAtomVolume,
    render_fn: AbstractVolumeRenderFn,
    *,
    output_type: type[FourierVoxelGridVolume] = FourierVoxelGridVolume,
) -> FourierVoxelGridVolume: ...


@overload
def render_voxel_volume(
    atom_volume: AbstractAtomVolume,
    render_fn: AbstractVolumeRenderFn,
    *,
    output_type: type[FourierVoxelSplineVolume] = FourierVoxelSplineVolume,
) -> FourierVoxelSplineVolume: ...


@overload
def render_voxel_volume(
    atom_volume: AbstractAtomVolume,
    render_fn: AbstractVolumeRenderFn,
    *,
    output_type: type[RealVoxelGridVolume] = RealVoxelGridVolume,
) -> RealVoxelGridVolume: ...


@overload
def render_voxel_volume(
    atom_volume: AbstractAtomVolume,
    render_fn: AbstractVolumeRenderFn,
    *,
    output_type: type[RealVoxelCloudVolume] = RealVoxelCloudVolume,
) -> RealVoxelCloudVolume: ...


def render_voxel_volume(
    atom_volume: AbstractAtomVolume,
    render_fn: AbstractVolumeRenderFn,
    *,
    output_type: type[
        FourierVoxelGridVolume
        | FourierVoxelSplineVolume
        | RealVoxelGridVolume
        | RealVoxelCloudVolume
    ] = FourierVoxelGridVolume,
) -> (
    FourierVoxelGridVolume
    | FourierVoxelSplineVolume
    | RealVoxelGridVolume
    | RealVoxelCloudVolume
):
    """Render a voxel volume representation from an atomistic one.

    !!! example "Simulate an image with Fourier slice extraction"

        ```python
        import cryojax.simulator as cxs

        # Simulate an image with Fourier slice extraction
        voxel_volume = cxs.render_voxel_volume(
            atom_volume=cxs.load_tabulated_volume("example.pdb"),
            render_fn=cxs.AutoVolumeRenderFn(shape=(100, 100, 100), voxel_size=1.0),
            output_type=cxs.FourierVoxelGridVolume,
        )
        image_model = cxs.make_image_model(voxel_volume, ...)
        image = image_model.simulate()
        ```

    **Arguments:**

    - `atom_volume`:
        An atomistic volume representation, such as a
        [`cryojax.simulator.GaussianMixtureVolume`][] or a
        [`cryojax.simulator.IndependentAtomVolume`][].
    - `render_fn`:
        A [`cryojax.simulator.AbstractVolumeRenderFn`][] that
        accepts `atom_volume` as input. Choose
        [`cryojax.simulator.AutoVolumeRenderFn`][] to
        auto-select a method from existing cryoJAX
        implementations.
    - `output_type`:
        The [`cryojax.simulator.AbstractVoxelVolume`][]
        implementation to output.
        Either [`cryojax.simulator.FourierVoxelGridVolume`][] /
        [`cryojax.simulator.FourierVoxelSplineVolume`][] for
        fourier-space representations, or
        [`cryojax.simulator.RealVoxelGridVolume`][] /
        [`cryojax.simulator.RealVoxelCloudVolume`][] for real-space.


    **Returns:**

    A [`cryojax.simulator.AbstractVoxelVolume`][] with exact type
    equal to `output_type`.
    """
    if len(set(render_fn.shape)) != 1:
        raise ValueError(
            "Function `render_voxel_volume` only supports "
            "volume rendering for cubic volumes, i.e. "
            "`render_fn.shape = (N, N, N)`. Got "
            f"`render_fn.shape = {render_fn.shape}`."
        )
    if output_type == FourierVoxelGridVolume or output_type == FourierVoxelSplineVolume:
        fourier_voxel_grid = render_fn(
            atom_volume, outputs_real_space=False, outputs_rfft=False
        )
        if output_type == FourierVoxelGridVolume:
            return FourierVoxelGridVolume.from_fourier_voxel_grid(fourier_voxel_grid)
        else:
            return FourierVoxelSplineVolume.from_fourier_voxel_grid(fourier_voxel_grid)
    elif output_type == RealVoxelGridVolume or output_type == RealVoxelCloudVolume:
        coordinate_grid = make_coordinate_grid(render_fn.shape)
        real_voxel_grid = render_fn(atom_volume, outputs_real_space=True)
        if output_type == RealVoxelGridVolume:
            return RealVoxelGridVolume(real_voxel_grid, coordinate_grid)
        else:
            return RealVoxelCloudVolume.from_real_voxel_grid(
                real_voxel_grid, coordinate_grid_in_pixels=coordinate_grid
            )
    else:
        raise ValueError(
            f"Got `output_type = {output_type}`, but this is "
            "not supported by `render_voxel_volume(..., output_type=...)`."
            "Valid values for `output_type` are `FourierVoxelGridVolume`, "
            "`FourierVoxelSplineVolume`, `RealVoxelGridVolume`, or "
            "`RealVoxelCloudVolume`."
        )
