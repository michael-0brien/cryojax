import pathlib
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeVar, overload

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import lineax as lx
import mmdf
import pandas as pd
from jaxtyping import Array, Bool, PyTree

from ..atom_util import split_atoms_by_element
from ..constants import LobatoScatteringFactorParameters, PengScatteringFactorParameters
from ..io import mmdf_to_atoms
from ..jax_util import NDArrayLike, make_filter_spec
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
    AbstractVolumeIntegrator,
    AbstractVolumeParametrization,
    AutoVolumeProjection,
    GaussianMixtureVolume,
    IndependentAtomVolume,
)


Args = TypeVar("Args")

identity_fn = eqxi.doc_repr(lambda x, _: x, "identity_fn")


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: None = None,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: None = None,
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> ProjectionImageModel: ...


@overload
def make_image_model(  # pyright: ignore[reportOverlappingOverload]
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: None = None,
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> LinearImageModel: ...


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["contrast"] = "contrast",
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> ContrastImageModel: ...


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["intensity"] = "intensity",
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> IntensityImageModel: ...


@overload
def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["counts"] = "counts",
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> ElectronCountsImageModel: ...


def make_image_model(
    volume_parametrization: AbstractVolumeParametrization,
    image_config: AbstractImageConfig,
    pose: AbstractPose,
    transfer_theory: ContrastTransferTheory | None = None,
    volume_integrator: AbstractVolumeIntegrator = AutoVolumeProjection(),
    detector: AbstractDetector | None = None,
    *,
    quantity_mode: Literal["contrast", "intensity", "counts"] | None = None,
    applies_translation: bool = True,
    normalizes_signal: bool = False,
    signal_region: Bool[NDArrayLike, "_ _"] | None = None,
    translate_mode: Literal["fft", "atom"] = "fft",
) -> AbstractImageModel:
    """Construct an `AbstractImageModel` for most common use-cases.

    **Arguments:**

    - `volume_parametrization`:
        The representation of the protein volume.
        Common choices are the `FourierVoxelGridVolume`
        for fourier-space voxel grids or the `GaussianMixtureVolume`.
    - `image_config`:
        The configuration for the image and imagining instrument. Unless using
        a model that uses the electron dose as a parameter, choose the
        `BasicImageConfig`. Otherwise, choose the `DoseImageConfig`.
    - `pose`:
        The pose in a particular parameterization convention. Common options
        are the `EulerAnglePose`, `QuaternionPose`, or `AxisAnglePose`.
    - `transfer_theory`:
        The contrast transfer function and its theory for how it is applied
        to the image.
    - `volume_integrator`:
        Optionally pass the method for integrating the electrostatic potential onto
        the plane (e.g. projection via fourier slice extraction). If not provided,
        a default option is chosen.
    - `detector`:
        If `quantity_mode = 'counts'` is chosen, then an `AbstractDetector` class must be
        chosen to simulate electron counts.
    - `applies_translation`:
        If `True`, apply the in-plane translation in the `AbstractPose`
        via phase shifts in fourier space.
    - `normalizes_signal`:
        If `True`, normalizes_signal the image before returning.
    - `signal_region`:
        A boolean array that is 1 where there is signal,
        and 0 otherwise used to normalize the image.
        Must have shape equal to `AbstractImageConfig.shape`.
    - `quantity_mode`:
        The physical observable to simulate. If `None`, simulate without scaling
        to physical units using the `LinearImageModel`.
        Options are
        - 'contrast':
            Uses the `ContrastImageModel` to simulate contrast. This is
            default.
        - 'intensity':
            Uses the `IntensityImageModel` to simulate intensity.
        - 'counts':
            Uses the `ElectronCountsImageModel` to simulate electron counts.
            If this is passed, a `detector` must also be passed.
    - `translate_mode`:
        If `'fft'`, apply in-plane translation via phase
        shifts in the Fourier domain. If `'atoms'` apply translation
        on atom positions before projection. Does nothing if
        `applies_translation = False`.

    **Returns:**

    An `AbstractImageModel`. Simulate an image with syntax

    ```python
    image_model = make_image_model(...)
    image = image_model.simulate()
    ```
    """
    options = dict(
        applies_translation=applies_translation,
        normalizes_signal=normalizes_signal,
        signal_region=signal_region,
        translate_mode=translate_mode,
    )
    if transfer_theory is None:
        # Image model for projections
        image_model = ProjectionImageModel(
            volume_parametrization,
            pose,
            image_config,
            volume_integrator,
            **options,  # pyright: ignore[reportArgumentType]
        )
    else:
        # Simulate physical observables
        if quantity_mode is None:
            # Linear image model
            image_model = LinearImageModel(
                volume_parametrization,
                pose,
                image_config,
                volume_integrator,
                transfer_theory,
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
                    volume_parametrization,
                    pose,
                    image_config,
                    scattering_theory,
                    detector,
                    **options,  # pyright: ignore[reportArgumentType]
                )
            elif quantity_mode == "contrast":
                image_model = ContrastImageModel(
                    volume_parametrization,
                    pose,
                    image_config,
                    scattering_theory,
                    **options,  # pyright: ignore[reportArgumentType]
                )
            elif quantity_mode == "intensity":
                image_model = IntensityImageModel(
                    volume_parametrization,
                    pose,
                    image_config,
                    scattering_theory,
                    **options,  # pyright: ignore[reportArgumentType]
                )
            else:
                raise ValueError(
                    f"`quantity_mode = {quantity_mode}` not supported. Supported "
                    "modes for simulating "
                    "physical quantities are 'contrast', 'intensity', and 'counts'."
                )

    return image_model


@overload
def load_tabulated_volume(  # pyright: ignore[reportOverlappingOverload]
    path_or_mmdf: str | pathlib.Path | pd.DataFrame,
    *,
    output_type: type[IndependentAtomVolume] = IndependentAtomVolume,
    tabulation: Literal["peng", "lobato"] = "peng",
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
    ] = IndependentAtomVolume,
    tabulation: Literal["peng", "lobato"] = "peng",
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
        Either [`cryojax.simulator.GaussianMixtureVolume`][] or
        [`cryojax.simulator.IndependentAtomVolume`][].
    - `tabulation`:
        Specifies which electron scattering factor tabulation to use.
        Supported values are `tabulation = 'peng'` or `tabulation = 'lobato'`.
        See [`cryojax.constants.PengScatteringFactorParameters`][] and
        [`cryojax.constants.LobatoScatteringFactorParameters`][]
        for more information.
    - `include_b_factors`:
        If `True`, include PDB B-factors in the volume.
    - `b_factor_fn`:
        A function that modulates PDB B-factors before passing to the
        volume. Has signature
        `modulated_b_factor = b_factor_fn(pdb_b_factor, atomic_number)`.
        If `output_type = IndependentAtomVolume`, `pdb_b_factor` is
        the mean B-factor for a given atom type.
    - `selection_string`:
        A string for [`mdtraj` atom selection](https://mdtraj.org/1.9.4/examples/atom-selection.html#atom-selection).
        See [`cryojax.io.read_atoms_from_pdb`][] for documentation.
    - `pdb_options`:
        Additional keyword options passed to [`cryojax.io.read_atoms_from_pdb`][],
        not including `selection_string`.

    **Returns:**

    Returns a [`cryojax.simulator.GaussianMixtureVolume`][] or
    a [`cryojax.simulator.IndependentAtomVolume`][] depending on
    `output_type`.
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
        elif tabulation == "lobato":
            parameters = LobatoScatteringFactorParameters(atom_ids)
        else:
            raise ValueError(
                "Only `tabulation` equal to 'peng' or 'lobato' are supported in "
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


def make_linear_operator(
    simulate_fn: Callable[[Args], Array],
    args: Args,
    where_volume: Callable[[Args], Any],
    *,
    tags: object | Iterable[object] = (),
) -> tuple[lx.FunctionLinearOperator, Args]:
    """Convert from a cryoJAX abstraction for image simulation to a
    [`lineax`](https://docs.kidger.site/lineax/)'s matrix-vector multiplication
    abstraction.

    In particular, instantiates a [`lineax.FunctionLinearOperator`](https://docs.kidger.site/lineax/api/operators/#lineax.FunctionLinearOperator)
    to simulate an image.

    !!! example

        ```python
        import cryojax.simulator as cxs

        # Instantiate a linear operator
        volume_representation = cxs.FourierVoxelGridVolume.from_real_voxel_grid(...)
        image_model = cxs.make_image_model(volume_representation, ...)
        operator, volume = cxs.make_linear_operator(
            simulate_fn=lambda x: x.simulate(),
            args=image_model,
            where_volume=lambda x: x.volume_parametrization.fourier_voxel_grid,
        )
        # Simulate an image
        image = operator.mv(volume)
        # Access arguments other than those at `volume`
        args = operator.fn.args
    ```

    !!! warning

        This function promises that `simulate_fn` can be expressed as a
        linear operator with respect to the input arguments at `where_volume`.
        CryoJAX does not explicitly check if this is the case, so JAX will
        throw errors downstream.

    **Arguments:**

    - `simulate_fn`:
        A function with signature `image = simulate_fn(args)`
    - `args`:
        Input arguments to `simulate_fn`
    - `where_volume`:
        A pointer to where the arguments for the volume are in
        `args`.
    - `tags`:
        See `lineax.FunctionLinearOperator` for documentation.

    **Returns:**

    A tuple with first element `lineax.FunctionLinearOperator` and second element
    a pytree with the same structure as `pytree`, partitioned to only include the
    arguments at `where_volume`.
    """  # noqa: E501
    # Extract arguments for the volume corresponding to
    # `where_volume`
    filter_spec = make_filter_spec(args, where_volume)
    volume, other_args = eqx.partition(args, filter_spec)
    volume, static_args = eqx.partition(volume, eqx.is_array)
    other_args = eqx.combine(other_args, static_args)
    # Instantiate the `lineax.FunctionLinearOperator`
    simulate_wrapper = _SimulateFn(simulate_fn, other_args)
    input_structure = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), volume
    )
    linear_operator = lx.FunctionLinearOperator(
        fn=simulate_wrapper, input_structure=input_structure, tags=tags
    )
    return linear_operator, volume


class _SimulateFn(eqx.Module):
    fn: Callable[[PyTree], Array]
    args: PyTree

    def __call__(self, volume_args: PyTree) -> Array:
        pytree = eqx.combine(volume_args, self.args)
        return self.fn(pytree)
