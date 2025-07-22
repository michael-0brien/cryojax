"""cryoJAX compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import abc
import pathlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Literal, Optional, TypedDict, cast
from typing_extensions import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Int

from ...internal import NDArrayLike
from ...io import read_starfile, write_image_stack_to_mrc, write_starfile
from ...ndimage.operators import Constant, FourierGaussian
from ...simulator import (
    AberratedAstigmaticCTF,
    BasicConfig,
    ContrastTransferTheory,
    EulerAnglePose,
)
from .._particle_data import (
    AbstractParticleParameterFile,
    AbstractParticleStackDataset,
)


# RELION column entries
RELION_CTF_OPTICS_ENTRIES = [
    ("rlnSphericalAberration", "Float64"),
    ("rlnAmplitudeContrast", "Float64"),
]
RELION_INSTRUMENT_OPTICS_ENTRIES = [
    ("rlnImageSize", "Int64"),
    ("rlnVoltage", "Float64"),
    ("rlnImagePixelSize", "Float64"),
]
RELION_CTF_PARTICLE_ENTRIES = [
    ("rlnDefocusU", "Float64"),
    ("rlnDefocusV", "Float64"),
    ("rlnDefocusAngle", "Float64"),
    ("rlnPhaseShift", "Float64"),
]
RELION_POSE_PARTICLE_ENTRIES = [
    ("rlnOriginXAngst", "Float64"),
    ("rlnOriginYAngst", "Float64"),
    ("rlnAngleRot", "Float64"),
    ("rlnAngleTilt", "Float64"),
    ("rlnAnglePsi", "Float64"),
]
# Default entries for writing
RELION_DEFAULT_OPTICS_ENTRIES = [
    *RELION_INSTRUMENT_OPTICS_ENTRIES,
    *RELION_CTF_OPTICS_ENTRIES,
    ("rlnOpticsGroup", "Int64"),
]
RELION_DEFAULT_PARTICLE_ENTRIES = [
    *RELION_CTF_PARTICLE_ENTRIES,
    *RELION_POSE_PARTICLE_ENTRIES,
    ("rlnOpticsGroup", "Int64"),
]
# Required entries for loading
RELION_REQUIRED_OPTICS_ENTRIES = RELION_DEFAULT_OPTICS_ENTRIES
RELION_REQUIRED_PARTICLE_ENTRIES = [
    *RELION_CTF_PARTICLE_ENTRIES,
    ("rlnOpticsGroup", "Int64"),
]
RELION_SUPPORTED_PARTICLE_ENTRIES = [
    *RELION_REQUIRED_PARTICLE_ENTRIES,
    ("rlnCtfBfactor", "Float64"),
    ("rlnCtfScalefactor", "Float64"),
]


class ParticleParameterInfo(TypedDict):
    """Parameters for a particle stack from RELION."""

    config: BasicConfig
    pose: EulerAnglePose
    transfer_theory: ContrastTransferTheory

    metadata: Optional[pd.DataFrame]


class ParticleStackInfo(TypedDict):
    """Particle stack info from RELION."""

    parameters: ParticleParameterInfo
    images: Float[Array, "... y_dim x_dim"]


ParticleParameterLike = dict[str, Any] | ParticleParameterInfo
ParticleStackLike = dict[str, Any] | ParticleStackInfo


class StarfileData(TypedDict):
    optics: pd.DataFrame
    particles: pd.DataFrame


class MrcfileSettings(TypedDict):
    prefix: str
    output_folder: str | pathlib.Path
    n_characters: int
    delimiter: str
    overwrite: bool
    compression: str | None


def _default_make_config_fn(shape, pixel_size, voltage_in_kilovolts, **kwargs):
    return BasicConfig(shape, pixel_size, voltage_in_kilovolts, **kwargs)


class AbstractParticleStarFile(
    AbstractParticleParameterFile[ParticleParameterInfo, ParticleParameterLike]
):
    @property
    @override
    def path_to_output(self) -> pathlib.Path:
        return self.path_to_starfile

    @path_to_output.setter
    @override
    def path_to_output(self, value: str | pathlib.Path):
        self.path_to_starfile = value

    @property
    @abc.abstractmethod
    def path_to_starfile(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_starfile.setter
    @abc.abstractmethod
    def path_to_starfile(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def starfile_data(self) -> StarfileData:
        raise NotImplementedError

    @starfile_data.setter
    @abc.abstractmethod
    def starfile_data(self, value: dict[str, pd.DataFrame]):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_metadata(self) -> bool:
        raise NotImplementedError

    @loads_metadata.setter
    @abc.abstractmethod
    def loads_metadata(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_envelope(self) -> bool:
        raise NotImplementedError

    @loads_envelope.setter
    @abc.abstractmethod
    def loads_envelope(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def broadcasts_optics_group(self) -> bool:
        raise NotImplementedError

    @broadcasts_optics_group.setter
    @abc.abstractmethod
    def broadcasts_optics_group(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def updates_optics_group(self) -> bool:
        raise NotImplementedError

    @updates_optics_group.setter
    @abc.abstractmethod
    def updates_optics_group(self, value: bool):
        raise NotImplementedError

    def copy(self) -> Self:
        return deepcopy(self)


class RelionParticleParameterFile(AbstractParticleStarFile):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html)
    format.
    """

    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        exists_ok: bool = False,
        selection_filter: Optional[dict[str, Callable]] = None,
        *,
        loads_metadata: bool = False,
        broadcasts_optics_group: bool = True,
        loads_envelope: bool = False,
        updates_optics_group: bool = False,
        make_config_fn: Callable[
            [tuple[int, int], Float[Array, "..."], Float[Array, "..."]],
            BasicConfig,
        ] = _default_make_config_fn,
    ):
        """**Arguments:**

        - `path_to_starfile`:
            The path to the RELION STAR file. If the path does not exist
            and `mode = 'w'`, an empty dataset will be created.
        - `path_to_relion_project`: The path to the RELION project directory.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *parameters*. This is done by storing an empty dataset in
            `RelionParticleParameterFile.starfile_data`. If a STAR file
            already exists at `path_to_starfile`, set `exists_ok = True`.
            - If `mode = 'r'`, the STAR file at `path_to_starfile` is read
            into `RelionParticleParameterFile.starfile_data`.
        - `exists_ok`:
            If the `path_to_starfile` already exists, if `True` and `mode = 'w'`
            nonetheless stores an empty `RelionParticleParameterFile.starfile_data`.
        - `selection_filter`:
            A dictionary used to include only particular dataset elements.
            The keys of this dictionary should be any data entry in the STAR
            file, while the values should be a function that takes in a
            column and returns a boolean mask for the column. For example,
            filter by class using
            `selection_filter["rlnClassNumber"] = lambda x: x == 0`.
        - `loads_metadata`:
            If `True`, the resulting `ParticleParameterInfo` dict loads
            the raw metadata from the STAR file that is not otherwise included
            in the `ParticleParameterInfo` as a `pandas.DataFrame`.
            If this is set to `True`, note that dictionaries cannot pass through
            JIT boundaries without removing the metadata.
        - `broadcasts_optics_group`:
            If `True`, select optics group parameters are broadcasted. If
            there are multiple optics groups in the STAR file, parameters
            are always broadcasted and this option is null.
        - `loads_envelope`:
            If `True`, read in the parameters of the CTF envelope function, i.e.
            "rlnCtfScalefactor" and "rlnCtfBfactor".
        - `updates_optics_group`:
            If `True`, when re-writing STAR file entries via
            `dataset[idx] = parameters` syntax, creates a new optics group entry.
        - `make_config_fn`:
            A function used for `BasicConfig` initialization that returns
            an `BasicConfig`. This is used to customize the metadata of the
            read object.
        """
        # Private attributes
        self._make_config_fn = make_config_fn
        self._mode = _validate_mode(mode)
        # The STAR file data
        self._path_to_starfile = pathlib.Path(path_to_starfile)
        self._starfile_data = _load_starfile_data(
            self._path_to_starfile, selection_filter, mode, exists_ok
        )
        # Properties for loading
        self._loads_metadata = loads_metadata
        self._broadcasts_optics_group = broadcasts_optics_group
        self._loads_envelope = loads_envelope
        # Properties for writing
        self._updates_optics_group = updates_optics_group

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"]
    ) -> ParticleParameterInfo:
        # Validate index
        n_rows = self.starfile_data["particles"].shape[0]
        _validate_dataset_index(type(self), index, n_rows)
        # ... read particle data at the requested indices
        particle_data = self.starfile_data["particles"]
        particle_data_at_index = particle_data.iloc[index]
        # ... read optics group given the particle data
        optics_data = self.starfile_data["optics"]
        optics_group = _get_optics_group_from_particle_data(
            particle_data_at_index, optics_data
        )
        # Load the image stack and STAR file parameters
        config, transfer_theory, pose = _make_pytrees_from_starfile(
            particle_data_at_index,
            optics_group,
            self.broadcasts_optics_group,
            self.loads_envelope,
            self._make_config_fn,
        )
        if self.loads_metadata:
            # ... convert to dataframe for serialization
            if isinstance(particle_data_at_index, pd.Series):
                particle_data_at_index = particle_data_at_index.to_frame().T
            # ... no overlapping keys with loaded pytrees
            redundant_entry_labels, _ = list(zip(*RELION_SUPPORTED_PARTICLE_ENTRIES))
            columns = particle_data_at_index.columns
            remove_columns = [
                column for column in columns if column in redundant_entry_labels
            ]
            metadata = particle_data_at_index.drop(remove_columns, axis="columns")
        else:
            metadata = None

        return ParticleParameterInfo(
            config=config,
            pose=pose,
            transfer_theory=transfer_theory,
            metadata=metadata,
        )

    @override
    def __len__(self) -> int:
        return len(self.starfile_data["particles"])

    @override
    def __setitem__(
        self,
        index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"],
        value: ParticleParameterLike,
    ):
        # Make sure index is valid
        n_rows = self.starfile_data["particles"].shape[0]
        _validate_dataset_index(type(self), index, n_rows)
        # ... also, the parameters too
        _validate_parameters(value, force_keys=False)
        # Grab the current and new optics and particle data
        if self.updates_optics_group:
            optics_group_index = _make_optics_group_index(self.starfile_data["optics"])
            particle_data_for_update = _parameters_to_particle_data(
                value, optics_group_index
            )
            optics_data_to_append = _parameters_to_optics_data(value, optics_group_index)
            optics_data = pd.concat(
                [self.starfile_data["optics"], optics_data_to_append], ignore_index=True
            )
        else:
            particle_data_for_update = _parameters_to_particle_data(value)
            optics_data = self.starfile_data["optics"]
        particle_data = self.starfile_data["particles"]
        # Set new empty columns in the particle data, if the update data includes this
        new_columns = list(
            set(particle_data_for_update.columns) - set(particle_data.columns)
        )
        for column in new_columns:
            dtype = pd.api.types.pandas_dtype(particle_data_for_update[column].dtype)
            particle_data[column] = pd.Series(dtype=dtype)
        # Finally, set the updated data
        if isinstance(index, (int, np.ndarray)):
            index = np.atleast_1d(index)
        particle_data.loc[
            particle_data.index[index], particle_data_for_update.columns
        ] = particle_data_for_update.values
        self._starfile_data = StarfileData(optics=optics_data, particles=particle_data)

    @override
    def append(self, value: ParticleParameterLike):
        """Add an entry or entries to the dataset.

        **Arguments:**

        - `value`:
            A dictionary of parameters to add to the dataset.
        """
        # Make sure parameters are valid
        _validate_parameters(value, force_keys=True)
        # Make new optics group
        optics_group_index = _make_optics_group_index(self.starfile_data["optics"])
        optics_data, optics_data_to_append = (
            self.starfile_data["optics"],
            _parameters_to_optics_data(value, optics_group_index),
        )
        # Make new particle entries
        particle_data, particle_data_to_append = (
            self.starfile_data["particles"],
            _parameters_to_particle_data(value, optics_group_index),
        )
        # Concatenate and set new entries
        optics_data = (
            pd.concat([optics_data, optics_data_to_append], ignore_index=True)
            if len(optics_data) > 0
            else optics_data_to_append
        )
        particle_data = (
            pd.concat([particle_data, particle_data_to_append], ignore_index=True)
            if len(particle_data) > 0
            else particle_data_to_append
        )
        self._starfile_data = StarfileData(optics=optics_data, particles=particle_data)

    @override
    def save(
        self,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        path_to_starfile = self.path_to_starfile
        path_exists = path_to_starfile.exists()
        if path_exists and not overwrite:
            raise FileExistsError(
                f"Tried saving STAR file, but file {str(path_to_starfile)} "
                "already exists. To overwrite existing STAR file, set "
                f"`{type(self).__name__}.overwrite = True`."
            )
        else:
            if not path_to_starfile.parent.exists():
                path_to_starfile.parent.mkdir(parents=True)
            write_starfile(self.starfile_data, path_to_starfile, **kwargs)

    @property
    @override
    def path_to_starfile(self) -> pathlib.Path:
        return self._path_to_starfile

    @path_to_starfile.setter
    @override
    def path_to_starfile(self, value: str | pathlib.Path):
        self._path_to_starfile = pathlib.Path(value)

    @property
    def mode(self) -> Literal["r", "w"]:
        return self._mode  # type: ignore

    @property
    @override
    def starfile_data(self) -> StarfileData:
        return self._starfile_data

    @starfile_data.setter
    @override
    def starfile_data(self, value: dict[str, pd.DataFrame]):
        if "particles" in value and "optics" in value:
            particle_data, optics_data = value["particles"], value["optics"]
            if isinstance(particle_data, pd.DataFrame) and isinstance(
                optics_data, pd.DataFrame
            ):
                self._starfile_data = StarfileData(
                    optics=optics_data, particles=particle_data
                )
            else:
                raise ValueError(
                    "STAR file data must be a dictionary "
                    "of pandas DataFrames, with keys equal to "
                    "'particles' and 'optics'. Found that the "
                    f"particle data was type `{type(particle_data).__name__}`"
                    "and the optics data was type "
                    f"`{type(optics_data).__name__}`."
                )
        else:
            raise ValueError(
                "STAR file data must be a dictionary "
                "of pandas DataFrames, with keys equal to "
                "'particles' and 'optics'. Tried setting "
                f"with a dictionary with keys {list(value.keys())}."
            )

    @property
    @override
    def loads_metadata(self) -> bool:
        return self._loads_metadata

    @loads_metadata.setter
    @override
    def loads_metadata(self, value: bool):
        self._loads_metadata = value

    @property
    @override
    def loads_envelope(self) -> bool:
        return self._loads_envelope

    @loads_envelope.setter
    @override
    def loads_envelope(self, value: bool):
        self._loads_envelope = value

    @property
    @override
    def broadcasts_optics_group(self) -> bool:
        return self._broadcasts_optics_group

    @broadcasts_optics_group.setter
    @override
    def broadcasts_optics_group(self, value: bool):
        self._broadcasts_optics_group = value

    @property
    @override
    def updates_optics_group(self) -> bool:
        return self._updates_optics_group

    @updates_optics_group.setter
    @override
    def updates_optics_group(self, value: bool):
        self._updates_optics_group = value


class RelionParticleStackDataset(
    AbstractParticleStackDataset[ParticleStackInfo, ParticleStackLike]
):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    def __init__(
        self,
        parameter_file: AbstractParticleStarFile,
        path_to_relion_project: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        mrcfile_settings: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `path_to_relion_project`:
            In RELION STAR files, only a relative path is added to the
            'rlnImageName' column. This is relative to the path to the
            "project", which is given by this parameter.
        - `parameter_file`:
            The `RelionParticleParameterFile`.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *images*. This is done by removing 'rlnImageName' from
            `parameter_file.starfile_data`, if it exists at all.
            does not have a column 'rlnImageName' and image files
            are not yet written.
            - If `mode = 'r'`, images are read from the 'rlnImageName'
            stored in the `parameter_file.starfile_data`.
        - `mrcfile_settings`:
            A dictionary with the following keys:
            - 'prefix':
                A `str` which acts as the prefix to the filenames. If this
                is equal to `"f"`, then the filename for image stack 0 will
                be called "f-00000.mrcs", for `delimiter = '-'` and
                `n_characters = 5`.
                are of format "filenam"
            - 'output_folder':
                A `str` or `pathlib.Path` type where to write MRC files,
                relative to the `path_to_relion_project`.
            - 'n_characters':
                An `int` for the number of characters to write the filename
                number string. If this is equal to `5`, then the filename
                for image stack 0 will be called "f-00000.mrcs", for
                `delimiter = '-'` and `prefix = 'f'`.
             - 'delimiter':
                A `str` for the delimiter between the filename prefix
                and number string. If this is equal to `'-'`, then the
                filename for image stack 0 will be called "f-00000.mrcs",
                for `n_characters = 5` and `prefix = 'f'`.
            - 'overwrite':
                If `True`, overwrite existing MRC file path if it exists.
        """
        # Set properties. First, core properties of the dataset, starting
        # with the `mode``
        self._mode = _validate_mode(mode)
        # ... then, the `parameter_file`. If `mode = 'w'` but
        # the images already exist, we should make a copy in case
        # those images are being used elsewhere
        particle_data, optics_data = (
            parameter_file.starfile_data["particles"],
            parameter_file.starfile_data["optics"],
        )
        images_exist = "rlnImageName" in particle_data.columns
        if mode == "w" and images_exist:
            parameter_file = parameter_file.copy()
        self._parameter_file = parameter_file
        # ... next, properties common to reading and writing images
        self._path_to_relion_project = pathlib.Path(path_to_relion_project)
        # ... last, properties for writing images
        self._mrcfile_settings = _dict_to_mrcfile_settings(mrcfile_settings)
        # Now, initialize for `mode = 'r'` vs `mode = 'w'`
        project_exists = self.path_to_relion_project.exists()
        if mode == "w":
            # Write empty "rlnImageName" column (defaults to NaN values)
            particle_data["rlnImageName"] = pd.Series(dtype=str)
            self.parameter_file.starfile_data = dict(
                optics=optics_data, particles=particle_data
            )
            # Make the project directory, if it does not yet exist
            if not project_exists:
                self.path_to_relion_project.mkdir(parents=True, exist_ok=False)
        else:
            if not images_exist:
                raise IOError(
                    "Could not find column 'rlnImageName' in the STAR file. "
                    "When using `mode = 'r'`, the STAR file must have this "
                    "column. To write images in a STAR file, "
                    "set `mode = 'w'`."
                )
            if not project_exists:
                raise FileNotFoundError(
                    "`RelionParticleStackDataset` opened in "
                    "'mode = `r`', but the RELION project directory "
                    "`path_to_relion_project` does not exist. "
                    "To write images in a STAR file in a new RELION project, "
                    "set `mode = 'w'`."
                )

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> ParticleStackInfo:
        # ... make sure particle metadata is being loaded
        loads_metadata = self.parameter_file.loads_metadata
        self.parameter_file.loads_metadata = True
        # ... read parameters
        parameters = self.parameter_file[index]
        # ... and construct dataframe
        particle_dataframe_at_index = cast(pd.DataFrame, parameters["metadata"])
        if "rlnImageName" not in particle_dataframe_at_index.columns:
            raise IOError(
                "Tried to read STAR file for "
                f"`RelionParticleStackDataset` index = {index}, "
                "but no entry found for 'rlnImageName'."
            )
        # ... then, load stack of images
        shape = parameters["config"].shape
        images = _load_image_stack_from_mrc(
            shape, particle_dataframe_at_index, self.path_to_relion_project
        )
        if parameters["pose"].offset_x_in_angstroms.ndim == 0:
            images = jnp.squeeze(images)

        # ... reset boolean
        self.parameter_file.loads_metadata = loads_metadata
        if not loads_metadata:
            parameters["metadata"] = None

        return ParticleStackInfo(parameters=parameters, images=images)

    @override
    def __len__(self) -> int:
        return len(self.parameter_file)

    @override
    def __setitem__(
        self, index: int | slice | Int[np.ndarray, ""], value: ParticleStackLike
    ):
        if isinstance(index, Int[np.ndarray, "_"]):  # type: ignore
            raise ValueError(
                "When setting `dataset[index] = ...`, "
                "it is not supported to pass `index` as a 1D numpy-array."
            )
        if not isinstance(value, dict):
            raise TypeError(
                "When setting `dataset[index] = foo`, "
                "foo must be a dictionary with key "
                "'images' and optionally key 'parameters'."
            )
        images, parameters = _unpack_particle_stack_dict(value)
        if parameters is not None:
            self.parameter_file[index] = parameters
        n_particles = len(self.parameter_file)
        index_array = np.atleast_1d(_index_to_array(index, n_particles))
        self.write_images(index_array, images, parameters=parameters)

    @override
    def append(self, value: ParticleStackLike):
        """Add an entry or entries to the dataset.

        **Arguments:**

        - `value`:

        """
        if not isinstance(value, dict):
            raise TypeError(
                "When appending `dataset.append(foo)`, "
                "`foo` must be a dictionary with keys "
                "'images' and 'parameters'."
            )
        images, parameters = _unpack_particle_stack_dict(value)
        if parameters is None:
            raise ValueError(
                "When appending dictionary `foo` as `dataset.append(foo)`, "
                "`foo` must have key 'parameters'."
            )
        start = len(self.parameter_file.starfile_data["particles"])
        # Append parameters. This automatically sets the 'rlnImageName'
        # column to NaNs
        self.parameter_file.append(parameters)
        # Write images
        stop = len(self.parameter_file.starfile_data["particles"])
        index_array = np.arange(start, stop, dtype=int)
        self.write_images(index_array, images, parameters=parameters)

    @override
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: Optional[ParticleParameterLike] = None,
    ):
        # Get relevant metadata
        particle_data = self.parameter_file.starfile_data["particles"]
        optics_data = self.parameter_file.starfile_data["optics"]
        if parameters is None:
            optics_group = _get_optics_group_from_particle_data(
                particle_data.iloc[index_array], optics_data
            )
            pixel_size, dim = (
                float(optics_group["rlnImagePixelSize"]),
                int(optics_group["rlnImageSize"]),
            )
        else:
            pixel_size, dim = (
                float(np.atleast_1d(parameters["config"].pixel_size)[0]),
                parameters["config"].shape[0],
            )
        if not (images.ndim in [2, 3] and images.shape[-2:] == (dim, dim)):
            raise ValueError(
                "Image(s) must be of "
                "shape `(n_images, dim, dim)` or `(dim, dim)`. "
                f"Tried writing image(s) of "
                f"shape {images.shape}."
            )
        # Prepare to write images
        if images.ndim == 2:
            images = images[None, ...]
        n_images, _ = images.shape[0], images.shape[1]
        n_particles = len(self.parameter_file)

        # Convert index into 1D ascending numpy array
        n_indices = index_array.size
        if n_images != n_indices:
            raise ValueError(
                "Tried to set dataset elements with an inconsistent number "
                f"of images. Found that the number of images was {n_images}, "
                f"while the number of dataset indices was {n_indices}."
            )
        # Get absolute path to the filename, as well as the 'rlnImageName'
        # column
        path_to_filename, rln_image_names = _make_image_filename(
            index_array,
            particle_data,
            n_particles,
            self.mrcfile_settings,
            self.path_to_relion_project,
        )
        # Set the STAR file column
        particle_data.loc[particle_data.index[index_array], "rlnImageName"] = (
            rln_image_names
        )
        self.parameter_file.starfile_data = dict(
            particles=particle_data, optics=optics_data
        )
        # ... and write the images to disk
        try:
            write_image_stack_to_mrc(
                images.astype(jnp.float32),
                pixel_size,
                path_to_filename,
                overwrite=self.mrcfile_settings["overwrite"],
                compression=self.mrcfile_settings["compression"],
            )
        except Exception as err:
            raise IOError(
                "Error occurred when writing image stack to MRC "
                "file. Most likely, the filename the writer "
                f"chose ({str(path_to_filename)}) already "
                "exists. Try changing the "
                "`RelionParticleStackDataset.mrcfile_settings`. "
                f"The error message was:\n{err}"
            )

    @property
    @override
    def parameter_file(self) -> AbstractParticleStarFile:
        return self._parameter_file

    @property
    @override
    def mode(self) -> Literal["r", "w"]:
        return self._mode  # type: ignore

    @property
    def path_to_relion_project(self) -> pathlib.Path:
        return self._path_to_relion_project

    @property
    def mrcfile_settings(self) -> MrcfileSettings:
        return self._mrcfile_settings

    @mrcfile_settings.setter
    def mrcfile_settings(self, value: dict[str, Any]):
        self._mrcfile_settings = _dict_to_mrcfile_settings(value)


def _load_starfile_data(
    path_to_starfile: pathlib.Path,
    selection_filter: dict[str, Callable] | None,
    mode: Literal["r", "w"],
    exists_ok: bool,
) -> StarfileData:
    if mode == "r":
        if path_to_starfile.exists():
            starfile_data = read_starfile(path_to_starfile)
            _validate_starfile_data(starfile_data)
            if selection_filter is not None:
                starfile_data = _select_particles(starfile_data, selection_filter)
        else:
            raise FileNotFoundError(
                f"Set `mode = '{mode}'`, but STAR file {str(path_to_starfile)} does not "
                "exist. To write a new STAR file, set `mode = 'w'`."
            )
    else:
        if path_to_starfile.exists() and not exists_ok:
            raise FileExistsError(
                f"Set `mode = 'w'`, but STAR file {str(path_to_starfile)} already "
                "exists. To read an existing STAR file, set `mode = 'r'` or "
                "to erase an existing STAR file, set `mode = 'w'` and "
                "`exists_ok=True`."
            )
        else:
            if selection_filter is None:
                starfile_data = dict(
                    optics=pd.DataFrame(
                        data={
                            column: pd.Series(dtype=dtype)
                            for column, dtype in RELION_DEFAULT_OPTICS_ENTRIES
                        }
                    ),
                    particles=pd.DataFrame(
                        data={
                            column: pd.Series(dtype=dtype)
                            for column, dtype in RELION_DEFAULT_PARTICLE_ENTRIES
                        }
                    ),
                )
            else:
                raise ValueError(
                    "Initialized a `RelionParticleParameterFile` in `mode = 'w'` "
                    "but also passed a `selection_filter`. Selection is only used "
                    "in `mode = 'r'`."
                )

    return StarfileData(
        optics=starfile_data["optics"], particles=starfile_data["particles"]
    )


def _validate_mode(mode: str) -> Literal["r", "w"]:
    if mode not in ["r", "w"]:
        raise ValueError(
            f"Passed unsupported `mode = {mode}`. Supported modes are 'r' and 'w'."
        )
    return mode  # type: ignore


def _select_particles(
    starfile_data: dict[str, pd.DataFrame], selection_filter: dict[str, Callable]
) -> dict[str, pd.DataFrame]:
    particle_data = starfile_data["particles"]
    boolean_mask = pd.Series(True, index=particle_data.index)
    for key in selection_filter:
        if key in particle_data.columns:
            fn = selection_filter[key]
            column = particle_data[key]
            base_error_message = (
                f"Error filtering key '{key}' in the `selection_filter`. "
                f"To filter the STAR file entries, `selection_filter['{key}']`"
                "must be a function that takes in an array and returns a "
                "boolean mask."
            )
            if isinstance(selection_filter[key], Callable):
                try:
                    mask_at_column = fn(column)
                except Exception as err:
                    raise ValueError(
                        f"{base_error_message} "
                        "When calling the function, caught an error:\n"
                        f"{err}"
                    )
                if not pd.api.types.is_bool_dtype(mask_at_column):
                    raise ValueError(
                        f"{base_error_message} "
                        "Found that the function did not return "
                        "a boolean dtype."
                    )
            else:
                raise ValueError(base_error_message)
            # Update mask
            boolean_mask = mask_at_column & boolean_mask
        else:
            raise ValueError(
                f"Included key '{key}' in the `selection_filter`, "
                "but this entry could not be found in the STAR file. "
                "The `selection_filter` must be a dictionary whose "
                "keys are strings in the STAR file and whose values "
                "are functions that take in columns and return boolean "
                "masks."
            )
    # Select particles using mask
    starfile_data["particles"] = particle_data[boolean_mask]

    return starfile_data


#
# STAR file reading
#
def _make_pytrees_from_starfile(
    starfile_dataframe,
    optics_group,
    broadcasts_optics_group,
    loads_envelope,
    make_config_fn,
) -> tuple[BasicConfig, ContrastTransferTheory, EulerAnglePose]:
    defocus_in_angstroms = (
        np.asarray(starfile_dataframe["rlnDefocusU"], dtype=float)
        + np.asarray(starfile_dataframe["rlnDefocusV"], dtype=float)
    ) / 2
    astigmatism_in_angstroms = np.asarray(
        starfile_dataframe["rlnDefocusU"], dtype=float
    ) - np.asarray(starfile_dataframe["rlnDefocusV"], dtype=float)
    astigmatism_angle = np.asarray(starfile_dataframe["rlnDefocusAngle"], dtype=float)
    phase_shift = np.asarray(starfile_dataframe["rlnPhaseShift"], dtype=float)
    # ... optics group data
    image_size = int(optics_group["rlnImageSize"])
    pixel_size = np.asarray(optics_group["rlnImagePixelSize"], dtype=float)
    voltage_in_kilovolts = np.asarray(optics_group["rlnVoltage"], dtype=float)
    spherical_aberration_in_mm = np.asarray(
        optics_group["rlnSphericalAberration"], dtype=float
    )
    amplitude_contrast_ratio = np.asarray(
        optics_group["rlnAmplitudeContrast"], dtype=float
    )
    # ... create cryojax objects. First, the BasicConfig
    image_shape = (image_size, image_size)
    batch_dim = 0 if defocus_in_angstroms.ndim == 0 else defocus_in_angstroms.shape[0]
    config = _make_config(
        image_shape,
        pixel_size,
        voltage_in_kilovolts,
        batch_dim,
        make_config_fn,
        broadcasts_optics_group,
    )
    # ... now the ContrastTransferTheory
    if loads_envelope:
        b_factor, scale_factor = (
            (
                np.asarray(starfile_dataframe["rlnCtfBfactor"], dtype=float)
                if "rlnCtfBfactor" in starfile_dataframe.keys()
                else None
            ),
            (
                np.asarray(starfile_dataframe["rlnCtfScalefactor"], dtype=float)
                if "rlnCtfScalefactor" in starfile_dataframe.keys()
                else None
            ),
        )
        envelope = _make_envelope_function(scale_factor, b_factor)
    else:
        envelope = None

    transfer_theory = _make_transfer_theory(
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
        spherical_aberration_in_mm,
        amplitude_contrast_ratio,
        phase_shift,
        envelope,
    )
    # ... and finally, the EulerAnglePose
    pose = EulerAnglePose()
    # ... values for the pose are optional, so look to see if
    # each key is present
    particle_keys = starfile_dataframe.keys()
    # Read the pose. first, xy offsets
    rln_origin_x_angst = (
        starfile_dataframe["rlnOriginXAngst"]
        if "rlnOriginXAngst" in particle_keys
        else 0.0
    )
    rln_origin_y_angst = (
        starfile_dataframe["rlnOriginYAngst"]
        if "rlnOriginYAngst" in particle_keys
        else 0.0
    )
    # ... rot angle
    rln_angle_rot = (
        starfile_dataframe["rlnAngleRot"] if "rlnAngleRot" in particle_keys else 0.0
    )
    # ... tilt angle
    if "rlnAngleTilt" in particle_keys:
        rln_angle_tilt = starfile_dataframe["rlnAngleTilt"]
    elif "rlnAngleTiltPrior" in particle_keys:  # support for helices
        rln_angle_tilt = starfile_dataframe["rlnAngleTiltPrior"]
    else:
        rln_angle_tilt = 0.0
    # ... psi angle
    if "rlnAnglePsi" in particle_keys:
        # Relion uses -999.0 as a placeholder for an un-estimated in-plane
        # rotation
        if isinstance(starfile_dataframe["rlnAnglePsi"], pd.Series):
            # ... check if all values are equal to -999.0. If so, just
            # replace the whole pandas.Series with 0.0
            if (
                starfile_dataframe["rlnAnglePsi"].nunique() == 1
                and starfile_dataframe["rlnAnglePsi"].iloc[0] == -999.0
            ):
                rln_angle_psi = 0.0
            # ... otherwise, replace -999.0 values with 0.0
            else:
                rln_angle_psi = starfile_dataframe["rlnAnglePsi"].where(
                    lambda x: x != -999.0, 0.0
                )
        else:
            # ... if the column is just equal to a float, then
            # directly check if it is equal to -999.0
            rln_angle_psi = (
                0.0
                if starfile_dataframe["rlnAnglePsi"] == -999.0
                else starfile_dataframe["rlnAnglePsi"]
            )
    elif "rlnAnglePsiPrior" in particle_keys:  # support for helices
        rln_angle_psi = starfile_dataframe["rlnAnglePsiPrior"]
    else:
        rln_angle_psi = 0.0
    # Now, flip the sign of the translations and transpose rotations.
    # RELION's convention thinks about the translation as "undoing" the translation
    # and rotation in the image
    pose_parameter_names = (
        "offset_x_in_angstroms",
        "offset_y_in_angstroms",
        "phi_angle",
        "theta_angle",
        "psi_angle",
    )
    pose_parameter_values = (
        -np.asarray(rln_origin_x_angst, dtype=float),
        -np.asarray(rln_origin_y_angst, dtype=float),
        -np.asarray(rln_angle_rot, dtype=float),
        -np.asarray(rln_angle_tilt, dtype=float),
        -np.asarray(rln_angle_psi, dtype=float),
    )
    # ... fill the EulerAnglePose will keys that are present. if they are not
    # present, keep the default values in the `pose = EulerAnglePose()`
    # instantiation
    maybe_make_full = lambda param: (
        np.full((batch_dim,), param) if batch_dim > 0 and param.shape == () else param
    )
    pose = eqx.tree_at(
        lambda p: tuple([getattr(p, name) for name in pose_parameter_names]),
        pose,
        tuple(
            [
                jnp.asarray(maybe_make_full(value), dtype=float)
                for value in pose_parameter_values
            ]
        ),
    )

    return config, transfer_theory, pose


def _make_config(
    image_shape,
    pixel_size,
    voltage_in_kilovolts,
    batch_dim,
    make_config_fn,
    broadcasts_optics_group,
):
    make_fn = lambda ps, volt: make_config_fn(image_shape, ps, volt)
    if broadcasts_optics_group:
        make_fn_vmap = eqx.filter_vmap(make_fn)
        return (
            make_fn(pixel_size, voltage_in_kilovolts)
            if batch_dim == 0
            else make_fn_vmap(
                jnp.full((batch_dim,), pixel_size),
                jnp.full((batch_dim,), voltage_in_kilovolts),
            )
        )
    else:
        return make_fn(pixel_size, voltage_in_kilovolts)


def _make_envelope_function(amp, b_factor):
    if b_factor is None and amp is None:
        warnings.warn(
            "loads_envelope was set to True, but no envelope parameters were found. "
            + "Setting envelope as None. "
            + "Make sure your starfile is correctly formatted or set loads_envelope=False"
        )
        return None

    elif b_factor is None and amp is not None:

        def _make_const_env(amp):
            return Constant(amp)

        @eqx.filter_vmap(in_axes=0, out_axes=0)
        def _make_const_env_vmap(amp):
            return _make_const_env(amp)

        return _make_const_env(amp) if amp.ndim == 0 else _make_const_env_vmap(amp)
    else:
        if amp is None:
            amp = jnp.asarray(1.0) if b_factor.ndim == 0 else jnp.ones_like(b_factor)

        def _make_gaussian_env(amp, b):
            return FourierGaussian(amplitude=amp, b_factor=b)

        @eqx.filter_vmap(in_axes=(0, 0), out_axes=0)
        def _make_gaussian_env_vmap(amp, b):
            return _make_gaussian_env(amp, b)

        return (
            _make_gaussian_env(amp, b_factor)
            if b_factor.ndim == 0
            else _make_gaussian_env_vmap(amp, b_factor)
        )


def _make_transfer_theory(defocus, astig, angle, sph, ac, ps, env=None):
    if env is not None:

        def _make_w_env(defocus, astig, angle, sph, ac, ps, env):
            ctf = AberratedAstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=astig,
                astigmatism_angle=angle,
                spherical_aberration_in_mm=sph,
            )
            return ContrastTransferTheory(
                ctf, env, amplitude_contrast_ratio=ac, phase_shift=ps
            )

        @eqx.filter_vmap(in_axes=(0, 0, 0, None, None, 0, 0), out_axes=0)
        def _make_w_env_vmap(defocus, astig, angle, sph, ac, ps, env):
            return _make_w_env(defocus, astig, angle, sph, ac, ps, env)

        return (
            _make_w_env(defocus, astig, angle, sph, ac, ps, env)
            if defocus.ndim == 0
            else _make_w_env_vmap(defocus, astig, angle, sph, ac, ps, env)
        )

    else:

        def _make_wo_env(defocus, astig, angle, sph, ac, ps):
            ctf = AberratedAstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=astig,
                astigmatism_angle=angle,
                spherical_aberration_in_mm=sph,
            )
            return ContrastTransferTheory(
                ctf, envelope=None, amplitude_contrast_ratio=ac, phase_shift=ps
            )

        @eqx.filter_vmap(in_axes=(0, 0, 0, None, None, 0), out_axes=0)
        def _make_wo_env_vmap(defocus, astig, angle, sph, ac, ps):
            return _make_wo_env(defocus, astig, angle, sph, ac, ps)

        return (
            _make_wo_env(defocus, astig, angle, sph, ac, ps)
            if defocus.ndim == 0
            else _make_wo_env_vmap(defocus, astig, angle, sph, ac, ps)
        )


def _load_image_stack_from_mrc(
    shape: tuple[int, int],
    particle_dataframe_at_index: pd.DataFrame,
    path_to_relion_project: str | pathlib.Path,
) -> Float[Array, "... y_dim x_dim"]:
    # Load particle image stack rlnImageName
    rln_image_names = particle_dataframe_at_index["rlnImageName"].reset_index(drop=True)
    if rln_image_names.convert_dtypes().dtype != "string":
        raise TypeError(
            "The 'rlnImageName' column was expected to be type string. "
            f"Instead, found type {rln_image_names.dtype}."
        )
    # Split the pandas.Series into two: one for the image index
    # and another for the filename
    split = rln_image_names.str.split("@").str
    # ... relion convention starts indexing at 1, not 0
    mrc_index, filenames = split[0], split[1]
    if filenames.isnull().any() or mrc_index.isnull().any():
        raise ValueError(
            "Could not parse filenames from the 'rlnImageName' "
            "column. Check to make sure these are the correct format, "
            "for example '00000@path/to/image-00000.mrcs'."
        )
    mrc_index = mrc_index.astype(int) - 1
    # Allocate memory for stack
    n_images = len(filenames)
    image_stack = np.empty((n_images, *shape), dtype=float)
    # Loop over filenames to fill stack
    unique_filenames = filenames.unique()
    for filename in unique_filenames:
        # Get the MRC indices
        mrc_index_at_filename = mrc_index[filename == filenames]
        particle_index_at_filename = mrc_index_at_filename.index
        path_to_filename = pathlib.Path(path_to_relion_project, filename)
        with mrcfile.mmap(path_to_filename, mode="r", permissive=True) as mrc:
            mrc_data = np.asarray(mrc.data)
            mrc_ndim = mrc_data.ndim
            mrc_shape = mrc_data.shape if mrc_ndim == 2 else mrc_data.shape[1:]

            if shape != mrc_shape:
                raise ValueError(
                    f"The shape of the MRC with filename {filename} "
                    "was found to not have the same shape loaded from "
                    "the 'rlnImageSize'. Check your MRC files and also "
                    "the STAR file optics group formatting."
                )
            image_stack[particle_index_at_filename] = (
                mrc_data if mrc_ndim == 2 else mrc_data[mrc_index_at_filename]
            )

    return jnp.asarray(image_stack)


def _validate_dataset_index(cls, index, n_rows):
    index_error_msg = lambda idx: (
        f"The index at which the `{cls.__name__}` was accessed was out of bounds! "
        f"The number of rows in the dataset is {n_rows}, but you tried to "
        f"access the index {idx}."
    )
    # ... pandas has bad error messages for its indexing
    if isinstance(index, (int, np.integer)):  # type: ignore
        if index > n_rows - 1:
            raise IndexError(index_error_msg(index))
    elif isinstance(index, slice):
        if index.start is not None and index.start > n_rows - 1:
            raise IndexError(index_error_msg(index.start))
    elif isinstance(index, np.ndarray):
        if index.size == 0:
            raise IndexError(
                "Found that the index passed to the dataset "
                "was an empty numpy array. Please pass a "
                "supported index."
            )
    else:
        raise IndexError(
            f"Indexing with the type {type(index)} is not supported by "
            f"`{cls.__name__}`. Indexing by integers is supported, one-dimensional "
            "fancy indexing is supported, and numpy-array indexing is supported. "
            "For example, like `particle = particle_dataset[0]`, "
            "`particle_stack = particle_dataset[0:5]`, "
            "or `particle_stack = dataset[np.array([1, 4, 3, 2])]`."
        )


def _validate_starfile_data(starfile_data: dict[str, pd.DataFrame]):
    if "particles" not in starfile_data.keys():
        raise ValueError("Missing key 'particles' in `starfile.read` output.")
    else:
        required_particle_keys, _ = zip(*RELION_REQUIRED_PARTICLE_ENTRIES)
        if not set(required_particle_keys).issubset(
            set(starfile_data["particles"].keys())
        ):
            raise ValueError(
                "Missing required keys in starfile 'particles' group. "
                f"Required keys are {required_particle_keys}."
            )
    if "optics" not in starfile_data.keys():
        raise ValueError("Missing key 'optics' in `starfile.read` output.")
    else:
        required_optics_keys, _ = zip(*RELION_REQUIRED_OPTICS_ENTRIES)
        if not set(required_optics_keys).issubset(set(starfile_data["optics"].keys())):
            raise ValueError(
                "Missing required keys in starfile 'optics' group. "
                f"Required keys are {required_optics_keys}."
            )


#
# Working with pytrees for I/O
#
def _unpack_particle_stack_dict(
    value: ParticleStackLike,
) -> tuple[Float[NDArrayLike, "... y_dim x_dim"], ParticleParameterLike | None]:
    if "images" in value:
        images = value["images"]
    else:
        raise ValueError(
            "When passing dictionary `foo` as `dataset.append(foo)` or "
            "`dataset[index] = foo`, `foo` must have key `images`."
        )
    if "parameters" in value:
        parameters = value["parameters"]
    else:
        parameters = None

    return images, parameters


#
# STAR file writing. First, functions for writing parameters
#
def _validate_parameters(parameters: ParticleParameterLike, force_keys: bool = False):
    if force_keys:
        if not {"config", "transfer_theory", "pose"}.issubset(parameters):
            raise ValueError(
                "When passing dictionary `foo` as `parameter_file.append(foo)` "
                "`foo` must have keys 'pose', 'transfer_theory', and 'config'."
            )
    if "config" in parameters:
        if not isinstance(parameters["config"], BasicConfig):
            raise TypeError(
                "Found that dict key 'config' was "
                "not type `cryojax.simulator.BasicConfig`. "
                f"Instead, it was type "
                f"{type(parameters['config']).__name__}."
            )
    if "transfer_theory" in parameters:
        if not isinstance(parameters["transfer_theory"], ContrastTransferTheory):
            raise TypeError(
                "Found that dict key 'transfer_theory' was "
                "not type `cryojax.simulator.ContrastTransferTheory`. "
                f"Instead, it was type "
                f"{type(parameters['transfer_theory']).__name__}."
            )
    if "pose" in parameters:
        if not isinstance(parameters["pose"], EulerAnglePose):
            raise TypeError(
                "Found that dict key 'pose' was "
                "not type `cryojax.simulator.EulerAnglePose`. "
                f"Instead, it was type "
                f"{type(parameters['pose']).__name__}."
            )


def _parameters_to_optics_data(
    parameters: ParticleParameterLike, optics_group_index: int
) -> pd.DataFrame:
    if {"config", "transfer_theory"}.issubset(parameters):
        shape = parameters["config"].shape
        if shape[0] == shape[1]:
            dim = shape[0]
        else:
            raise ValueError(
                "When adding optics group to STAR file, found "
                "non-square shape in `config.shape`. Only "
                "square shapes are supported."
            )
        pixel_size = parameters["config"].pixel_size
        voltage_in_kilovolts = parameters["config"].voltage_in_kilovolts
        amplitude_contrast_ratio = parameters["transfer_theory"].amplitude_contrast_ratio
        if isinstance(parameters["transfer_theory"].ctf, AberratedAstigmaticCTF):
            spherical_aberration_in_mm = getattr(
                parameters["transfer_theory"].ctf, "spherical_aberration_in_mm"
            )
        else:
            raise ValueError(
                "When adding optics group or particle to STAR file, "
                "`transfer_theory.ctf` must be type "
                "`AberratedAstigmaticCTF`. Instead, got type "
                f"{type(parameters['transfer_theory'].ctf).__name__}."
            )
        optics_group_dict = {
            "rlnOpticsGroup": optics_group_index,
            "rlnImageSize": dim,
            "rlnImagePixelSize": pixel_size,
            "rlnVoltage": voltage_in_kilovolts,
            "rlnSphericalAberration": spherical_aberration_in_mm,
            "rlnAmplitudeContrast": amplitude_contrast_ratio,
        }
        for k, v in optics_group_dict.items():
            if isinstance(v, Array | np.ndarray):
                arr = np.atleast_1d(np.asarray(v))
                if arr.size > 1:
                    if np.unique(arr).size > 1:
                        raise ValueError(
                            "Tried to fill a RELION optics group entry with an array "
                            "that has multiple unique values. Optics group compatible "
                            "arrays such as `BasicConfig.pixel_size` "
                            "must be either scalars or arrays all with the same value. "
                            f"Error occurred when filling '{k}' with array {v}."
                        )
                optics_group_dict[k] = arr.ravel()[0, None]
            else:
                optics_group_dict[k] = [v]
        optics_data = pd.DataFrame.from_dict(optics_group_dict)

        return optics_data
    else:
        raise ValueError(
            "Tried to add optics group to the STAR file, but "
            "parameter dictionary did not include 'config' "
            "and 'transfer_theory' keys. If you are setting parameters "
            "`parameter_file[index] = dict(pose=pose)` make sure "
            "`parameter_file.updates_optics_group = False`."
        )


def _parameters_to_particle_data(
    parameters: ParticleParameterLike,
    optics_group_index: Optional[int] = None,
) -> pd.DataFrame:
    particles_dict = {}
    if "pose" in parameters:
        # Now, pose parameters
        pose = parameters["pose"]
        if not isinstance(pose, EulerAnglePose):
            raise ValueError(
                "When adding particle to STAR file, "
                "`pose` must be type "
                "`EulerAnglePose`. Instead, got type "
                f"{type(pose).__name__}."
            )
        particles_dict["rlnOriginXAngst"] = -pose.offset_x_in_angstroms
        particles_dict["rlnOriginYAngst"] = -pose.offset_y_in_angstroms
        particles_dict["rlnAngleRot"] = -pose.phi_angle
        particles_dict["rlnAngleTilt"] = -pose.theta_angle
        particles_dict["rlnAnglePsi"] = -pose.psi_angle
        # Now, broadcast parameters to same dimension
        n_particles = pose.offset_x_in_angstroms.size
        for k, v in particles_dict.items():
            if v.shape == ():
                particles_dict[k] = np.full((n_particles,), np.asarray(v))
            elif v.size == n_particles:
                particles_dict[k] = v.ravel()
            else:
                raise ValueError(
                    "Found inconsistent number of particles "
                    "when adding particle to STAR file. "
                    "When running `parameter_file[index] = foo` "
                    "or `parameter_file.append(foo)`, make sure `foo` "
                    "has arrays that are scalars or all have the same "
                    "number of dimensions."
                )
    else:
        raise ValueError(
            "Tried to modify parameters in STAR file, but "
            "parameter dictionary did not include 'pose' "
            "key. If you are setting parameters "
            "`parameter_file[index] = foo` make sure "
            "`foo` is, for example, `foo = dict(pose=EulerAnglePose(...))`."
        )
    # Fill CTF parameters
    if "transfer_theory" in parameters:
        transfer_theory = parameters["transfer_theory"]
        if isinstance(transfer_theory.ctf, AberratedAstigmaticCTF):
            if pose.offset_z_in_angstroms is None:
                defocus_offset = 0.0
            else:
                defocus_offset = pose.offset_z_in_angstroms
            particles_dict["rlnDefocusU"] = (
                transfer_theory.ctf.defocus_in_angstroms
                + defocus_offset
                + transfer_theory.ctf.astigmatism_in_angstroms / 2
            )
            particles_dict["rlnDefocusV"] = (
                transfer_theory.ctf.defocus_in_angstroms
                + defocus_offset
                - transfer_theory.ctf.astigmatism_in_angstroms / 2
            )
            particles_dict["rlnDefocusAngle"] = transfer_theory.ctf.astigmatism_angle
        else:
            raise ValueError(
                "When adding particle to STAR file, "
                "`transfer_theory.ctf` must be type "
                "`AberratedAstigmaticCTF`. Instead, got type "
                f"{type(parameters['transfer_theory'].ctf).__name__}."
            )

        if isinstance(transfer_theory.envelope, FourierGaussian):
            particles_dict["rlnCtfBfactor"] = transfer_theory.envelope.b_factor
            particles_dict["rlnCtfScalefactor"] = transfer_theory.envelope.amplitude
        elif isinstance(transfer_theory.envelope, Constant):
            particles_dict["rlnCtfScalefactor"] = transfer_theory.envelope.value
        elif transfer_theory.envelope is None:
            pass
        else:
            raise ValueError(
                "When adding particle to STAR file, "
                "`transfer_theory.envelope` must be "
                "type `cryojax.image.operators.FourierGaussian` "
                "or `cryojax.image.operators.Constant`, or `None`. Got "
                f"{type(transfer_theory.envelope).__name__}."
            )
        particles_dict["rlnPhaseShift"] = transfer_theory.phase_shift
    # Now, miscellaneous parameters
    if optics_group_index is not None:
        particles_dict["rlnOpticsGroup"] = np.full(
            (n_particles,), optics_group_index, dtype=int
        )
    # Make sure parameters are numpy arrays
    particles_dict = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, Array) else x, particles_dict
    )
    particle_data = pd.DataFrame.from_dict(particles_dict)
    # Finally, see if the particle parameters has metadata and if so,
    # add this
    if "metadata" in parameters:
        if parameters["metadata"] is not None:
            metadata = cast(pd.DataFrame, parameters["metadata"])
            if n_particles != metadata.index.size:
                raise ValueError(
                    "When adding custom metadata to STAR file "
                    "with `parameter_file[index] = foo` or `parameter_file.append(foo)`, "
                    "found the number of particles "
                    "in `foo['metadata']` was inconsistent with the "
                    "number of particles in `foo['pose']`."
                )
            # Add metadata to dataframe
            particle_data = pd.concat(
                [particle_data, metadata], axis="columns", verify_integrity=True
            )
    return particle_data


def _make_optics_group_index(optics_data: pd.DataFrame) -> int:
    optics_group_indices = np.asarray(optics_data["rlnOpticsGroup"], dtype=int)
    last_optics_group_index = (
        0 if optics_group_indices.size == 0 else int(optics_group_indices[-1])
    )
    return last_optics_group_index + 1


def _parse_optics_group_index(particle_data_at_index: pd.DataFrame | pd.Series) -> int:
    # ... read optics data
    optics_group_indices = np.unique(
        np.atleast_1d(np.asarray(particle_data_at_index["rlnOpticsGroup"]))
    )
    if optics_group_indices.size > 1:
        raise NotImplementedError(
            "Tried to read multiple particles at once that belong "
            "to different optics groups, but this is not yet "
            "implemented. In the meantime, try reading one particle "
            "at a time."
        )
    optics_group_index = optics_group_indices[0]

    return int(optics_group_index)


def _get_optics_group_from_index(
    optics_data: pd.DataFrame, optics_group_index: int
) -> pd.Series:
    return optics_data[optics_data["rlnOpticsGroup"] == optics_group_index].iloc[0]


def _get_optics_group_from_particle_data(
    particle_data_at_index: pd.DataFrame | pd.Series, optics_data: pd.DataFrame
) -> pd.Series:
    optics_group_index = _parse_optics_group_index(particle_data_at_index)
    return _get_optics_group_from_index(optics_data, optics_group_index)


#
# Now, functions for writing image files
#
def _dict_to_mrcfile_settings(d: dict[str, Any]) -> MrcfileSettings:
    prefix = d["prefix"] if "prefix" in d else ""
    output_folder = d["output_folder"] if "output_folder" in d else ""
    delimiter = d["delimiter"] if "delimiter" in d else "_"
    n_characters = d["n_characters"] if "n_characters" in d else 6
    overwrite = d["overwrite"] if "overwrite" in d else False
    compression = d["compression"] if "compression" in d else None
    return MrcfileSettings(
        prefix=prefix,
        output_folder=output_folder,
        delimiter=delimiter,
        n_characters=n_characters,
        overwrite=overwrite,
        compression=compression,
    )


def _index_to_array(indices: slice | int | np.ndarray, size: int) -> np.ndarray:
    if isinstance(indices, slice):
        return np.asarray(range(*indices.indices(size)))
    else:
        return np.asarray(indices, dtype=int)


def _make_image_filename(
    index: Int[np.ndarray, " _"],
    particle_data: pd.DataFrame,
    n_particles: int,
    mrcfile_settings: MrcfileSettings,
    path_to_relion_project: pathlib.Path,
) -> tuple[pathlib.Path, list[str]]:
    # Get the file number for this MRC file
    if n_particles == 0:
        file_number = 0
    else:
        last_index = index[0] - 1
        if last_index == -1:
            file_number = 0
        else:
            last_filename = particle_data["rlnImageName"].iloc[last_index].split("@")[1]
            if pd.isna(last_filename):
                raise IOError(
                    "Tried to assign a number to the MRC file while writing "
                    "images, but could not grab the previous file number at "
                    f"index {int(last_index)}. At this index, found that the "
                    "filename was NaN."
                )
            else:
                file_number = _parse_filename_for_number(last_filename) + 1
    # Unpack settings
    prefix = mrcfile_settings["prefix"]
    output_folder = mrcfile_settings["output_folder"]
    delimiter = mrcfile_settings["delimiter"]
    n_characters = mrcfile_settings["n_characters"]
    # Generate filename
    file_number_fmt = _format_number_for_filename(file_number, n_characters=n_characters)
    if prefix == "":
        relative_path_to_filename = str(
            pathlib.Path(output_folder, file_number_fmt + ".mrcs")
        )
    else:
        relative_path_to_filename = str(
            pathlib.Path(output_folder, prefix + delimiter + file_number_fmt + ".mrcs")
        )
    # Finally, generate the 'rln_image_name' column, which includes the particle index
    rln_image_names = [
        _format_number_for_filename(int(i + 1), n_characters)
        + "@"
        + relative_path_to_filename
        for i in range(index.size)
    ]
    # Finally, the path to the filename
    path_to_filename = pathlib.Path(path_to_relion_project, relative_path_to_filename)

    return path_to_filename, rln_image_names


def _parse_filename_for_number(filename: str) -> int:
    match = re.search(r"(\d+)\.[^.]+$", filename)
    try:
        file_number = int(match.group(1))  # type: ignore
    except Exception as err:
        raise IOError(
            f"Could not get the file number from file {filename} "
            "Files must be enumerated with the trailing part of the "
            "filename as the file number, like so: '/path/to/file-0000.txt'. "
            f"When extracting the file number and converting it to an integer, "
            f"found error:\n\t{err}"
        )
    return file_number


def _format_number_for_filename(file_number: int, n_characters: int = 6):
    if file_number == 0:
        return "0" * n_characters
    else:
        n_digits = int(np.log10(file_number)) + 1
        return "0" * (n_characters - n_digits) + str(file_number)
