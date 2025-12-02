import pathlib
from collections.abc import Callable
from typing import Any, Literal, overload

import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Bool, ScalarLike

from ..atom_util import split_atoms_by_element
from ..constants import PengScatteringFactorParameters
from ..io import read_atoms_from_pdb
from ..jax_util import NDArrayLike
from ._detector import AbstractDetector
from ._image_config import AbstractImageConfig, DoseImageConfig
from ._image_model import (
    AbstractImageModel,
    ContrastImageModel,
    ElectronCountsImageModel,
    IntensityImageModel,
    LinearImageModel,
    ProjectionImageModel as ProjectionImageModel,
)
from ._pose import AbstractPose
from ._scattering_theory import WeakPhaseScatteringTheory
from ._transfer_theory import ContrastTransferTheory
from ._volume import (
    AbstractAtomVolume,
    AbstractVolumeIntegrator,
    AbstractVolumeParametrization,
    AutoVolumeProjection,
    GaussianMixtureVolume,
    IndependentAtomVolume,
)


identity_fn = eqxi.doc_repr(lambda x: x, "identity_fn")


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
    path_to_pdb: str | pathlib.Path,
    *,
    loads_gmm: Literal[True] = True,
    table: Literal["peng"] = "peng",
    include_b_factors: bool = True,
    b_factor_fn: Callable[[ScalarLike], ScalarLike] = identity_fn,
    selection_string: str = "all",
    pdb_options: dict[str, Any] = {},
) -> GaussianMixtureVolume: ...


@overload
def load_tabulated_volume(
    path_to_pdb: str | pathlib.Path,
    *,
    loads_gmm: Literal[False] = False,
    table: Literal["peng"] = "peng",
    include_b_factors: bool = True,
    b_factor_fn: Callable[[ScalarLike], ScalarLike] = identity_fn,
    selection_string: str = "all",
    pdb_options: dict[str, Any] = {},
) -> IndependentAtomVolume: ...


def load_tabulated_volume(
    path_to_pdb: str | pathlib.Path,
    *,
    loads_gmm: bool = True,
    table: Literal["peng"] = "peng",
    include_b_factors: bool = True,
    b_factor_fn: Callable[[ScalarLike], ScalarLike] = identity_fn,
    selection_string: str = "all",
    pdb_options: dict[str, Any] = {},
) -> AbstractAtomVolume:
    """Load an atomistic representation of a volume from
    tabulated electron scattering factors.

    **Arguments:**

    - `path_to_pdb`:
        The path to the PDB/PDBx file
    - `loads_gmm`:
        If `True`, load a [`cryojax.simulator.GaussianMixtureVolume`][] with
        [`cryojax.constants.PengScatteringFactorParameters`][].
    - `table`:
        Specifies which electron scattering factor table to use.
        For now, only `table = 'peng'` is supported.
    - `include_b_factors`:
        If `True`, include PDB B-factors in the volume.
    - `b_factor_fn`:
        A function that modulates PDB B-factors before passing to the
        volume. Has signature `modulated_b_factor = b_factor_fn(pdb_b_factor)`.
    - `selection_string`:
        See [`cryojax.io.read_atoms_from_pdb`][] for documentation.
    - `pdb_options`:
        Additional keyword options passed to [`cryojax.io.read_atoms_from_pdb`][],
        not including `selection_string`.

    **Returns:**

    If `loads_gmm = True`, returns a [`cryojax.simulator.GaussianMixtureVolume`][].
    Otherwise, returns a [`cryojax.simulator.IndependentAtomVolume`][].
    """
    atom_positions, atomic_numbers, atom_properties = read_atoms_from_pdb(
        path_to_pdb,
        loads_properties=True,
        selection_string=selection_string,
        **pdb_options,
    )
    if loads_gmm:
        # TODO: this is inefficient if this function is called multiple times,
        # as the electron scattering factor parameter table is read on each call
        peng_parameters = PengScatteringFactorParameters(atomic_numbers)
        b_factors = (
            jnp.asarray(b_factor_fn(atom_properties["b_factors"]), dtype=float)
            if include_b_factors
            else None
        )
        atom_volume = GaussianMixtureVolume.from_tabulated_parameters(
            atom_positions, peng_parameters, extra_b_factors=b_factors
        )
    else:
        (positions_by_id, b_factor_by_id), atom_ids = split_atoms_by_element(
            atomic_numbers, (atom_positions, atom_properties["b_factors"])
        )
        b_factor_by_id = tuple(
            jnp.asarray(b_factor_fn(jnp.mean(b))) for b in b_factor_by_id
        )
        if table == "peng":
            scattering_parameters = PengScatteringFactorParameters(atom_ids)
        else:
            raise ValueError(
                "Only `table = 'peng'` is supported in "
                "`load_tabulated_volume`. "
                "Additional tables are not yet implemented."
            )
        atom_volume = IndependentAtomVolume.from_tabulated_parameters(
            positions_by_id, scattering_parameters, b_factor_by_element=b_factor_by_id
        )

    return atom_volume
