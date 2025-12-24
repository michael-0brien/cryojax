"""cryoJAX compatibility with [CSPARC](https://relion.readthedocs.io/en/release-5.0/)."""

import abc
import pathlib
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, TypedDict, cast
from typing_extensions import Self, override

import equinox as eqx
import jax
import mrcfile
import numpy as np
import pandas as pd
from jaxtyping import Float, Int

from ...io import read_csparc_data
from ...jax_util import NDArrayLike
from ...ndimage import FourierConstant, FourierGaussian
from ...rotations import SO3
from ...simulator import (
    AstigmaticCTF,
    BasicImageConfig,
    ContrastTransferTheory,
    EulerAnglePose,
)
from .._particle_data import (
    AbstractParticleParameterFile,
    AbstractParticleStackDataset,
)


# CSPARC column entries
CSPARC_INSTRUMENT_ENTRIES = [
    ("blob/shape", "Int64"),
    ("ctf/accel_kv", "Float64"),
    ("blob/psize_A", "Float64"),
]
CSPARC_CTF_ENTRIES = [
    ("ctf/amp_contrast", "Float64"),
    ("ctf/cs_mm", "Float64"),
    ("ctf/df1_A", "Float64"),
    ("ctf/df2_A", "Float64"),
    ("ctf/df_angle_rad", "Float64"),
    ("ctf/phase_shift_rad", "Float64"),
]
CSPARC_POSE_ENTRIES = [
    ("alignments3D/pose", "Float64"),
    ("alignments3D/shift", "Float64"),
    ("alignments_class_0/pose", "Float64"),
    ("alignments_class_0/shift", "Float64"),
]


# Required entries for loading
CSPARC_REQUIRED_PARTICLE_ENTRIES = [
    *CSPARC_CTF_ENTRIES,
]
CSPARC_SUPPORTED_PARTICLE_ENTRIES = [
    *CSPARC_REQUIRED_PARTICLE_ENTRIES,
    *CSPARC_POSE_ENTRIES,
    ("ctf/bfactor", "Float64"),
    ("ctf/scale", "Float64"),
]


class ParticleParameterInfo(TypedDict):
    """Parameters for a particle stack from CSPARC."""

    image_config: BasicImageConfig
    pose: EulerAnglePose
    transfer_theory: ContrastTransferTheory

    metadata: pd.DataFrame | None


class ParticleStackInfo(TypedDict):
    """Particle stack info from CSPARC."""

    parameters: ParticleParameterInfo | None
    images: Float[np.ndarray, "... y_dim x_dim"]


ParticleParameterLike = dict[str, Any] | ParticleParameterInfo
ParticleStackLike = dict[str, Any] | ParticleStackInfo


class MrcfileSettings(TypedDict):
    prefix: str
    output_folder: str | pathlib.Path
    n_characters: int
    delimiter: str
    overwrite: bool
    compression: str | None


class AbstractParticleStarFile(
    AbstractParticleParameterFile[ParticleParameterInfo, ParticleParameterLike]
):
    @property
    @override
    def path_to_output(self) -> pathlib.Path:
        return self.path_to_csparc_metadata

    @path_to_output.setter
    @override
    def path_to_output(self, value: str | pathlib.Path):
        self.path_to_csparc_metadata = value

    @property
    @abc.abstractmethod
    def path_to_csparc_metadata(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_csparc_metadata.setter
    @abc.abstractmethod
    def path_to_csparc_metadata(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def csparc_metadata(self) -> pd.DataFrame:
        raise NotImplementedError

    @csparc_metadata.setter
    @abc.abstractmethod
    def csparc_metadata(self, value: dict[str, pd.DataFrame]):
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
    def broadcasts_image_config(self) -> bool:
        raise NotImplementedError

    @broadcasts_image_config.setter
    @abc.abstractmethod
    def broadcasts_image_config(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def updates_optics_group(self) -> bool:
        raise NotImplementedError

    @updates_optics_group.setter
    @abc.abstractmethod
    def updates_optics_group(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def inverts_rotation(self) -> bool:
        raise NotImplementedError

    @inverts_rotation.setter
    @abc.abstractmethod
    def inverts_rotation(self, value: bool):
        raise NotImplementedError

    def copy(self) -> Self:
        return deepcopy(self)


class CryoSparcParticleParameterFile(AbstractParticleStarFile):
    """A dataset that wraps a CSPARC particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html)
    format.
    """

    def __init__(
        self,
        path_to_metadata: str | pathlib.Path,
        selection_filter: dict[str, Callable] | None = None,
        *,
        loads_metadata: bool = False,
        broadcasts_image_config: bool = True,
        loads_envelope: bool = False,
        updates_optics_group: bool = False,
        inverts_rotation: bool = False,
        pad_options: dict = {},
    ):
        """**Arguments:**

        - `path_to_starlnOriginXAngstrfile`:
            The path to the CSPARC STAR file. If the path does not exist
            and `mode = 'w'`, an empty dataset will be created.
        - `path_to_relion_project`: The path to the CSPARC project directory.
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
        - `broadcasts_image_config`:
            If `True`, image config parameters are broadcasted with leading dimension
            as the number of particles.
        - `loads_envelope`:
            If `True`, read in the parameters of the CTF envelope function, i.e.
            "rlnCtfScalefactor" and "rlnCtfBfactor".
        - `updates_optics_group`:
            If `True`, when re-writing STAR file entries via
            `dataset[idx] = parameters` syntax, creates a new optics group entry.
        - `inverts_rotation`:
            If `True`, invert the pose representation upon return. Depending on
            the image formation process used in `cryojax`, this may be necessary
            for matching to CSPARC convention. For example, set to `True` when
            using the fourier slice extraction and other voxel representations.
        - `pad_options`:
            Padding options for image simulation, passed to the `BasicImageConfig`.
            See `BasicImageConfig` for documentation.
        """
        # Private attributes
        self._pad_options = pad_options

        # The CryoSparc file metadata
        self._path_to_metadata = pathlib.Path(path_to_metadata)
        self._csparc_metadata = _load_csparc_metadata(
            self._path_to_metadata, selection_filter
        )
        # Properties for loading
        self._loads_metadata = loads_metadata
        self._broadcasts_image_config = broadcasts_image_config
        self._loads_envelope = loads_envelope
        # Properties for writing
        self._updates_optics_group = updates_optics_group
        # Shared
        self._inverts_rotation = inverts_rotation

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"]
    ) -> ParticleParameterInfo:
        # Validate index
        n_rows = self.csparc_metadata.shape[0]
        _validate_dataset_index(type(self), index, n_rows)
        # ... read particle data at the requested indice
        csparc_data_at_index = self.csparc_metadata.iloc[index]

        # Load the image stack and STAR file parameters
        image_config, transfer_theory, pose = _make_pytrees_from_csparc_metadata(
            csparc_data_at_index,
            self.broadcasts_image_config,
            self.loads_envelope,
            self._pad_options,
            self._inverts_rotation,
        )
        if self.loads_metadata:
            # ... convert to dataframe for serialization
            if isinstance(csparc_data_at_index, pd.Series):
                csparc_data_at_index = csparc_data_at_index.to_frame().T
            # ... no overlapping keys with loaded pytrees
            redundant_entry_labels, _ = list(zip(*CSPARC_SUPPORTED_PARTICLE_ENTRIES))
            columns = csparc_data_at_index.columns
            remove_columns = [
                column for column in columns if column in redundant_entry_labels
            ]
            metadata = csparc_data_at_index.drop(remove_columns, axis="columns")
        else:
            metadata = None

        return ParticleParameterInfo(
            image_config=image_config,
            pose=pose,
            transfer_theory=transfer_theory,
            metadata=metadata,
        )

    @override
    def __len__(self) -> int:
        return len(self.csparc_metadata)

    @property
    @override
    def path_to_csparc_metadata(self) -> pathlib.Path:
        return self._path_to_medata

    @path_to_csparc_metadata.setter
    @override
    def path_to_csparc_metadata(self, value: str | pathlib.Path):
        self._path_to_medata = pathlib.Path(value)

    @property
    @override
    def csparc_metadata(self) -> pd.DataFrame:
        return self._csparc_metadata

    @csparc_metadata.setter
    @override
    def csparc_metadata(self, value: dict[str, pd.DataFrame]):
        raise NotImplementedError("csparc_metadata cannot be modified")

    @property
    def mode(self) -> Literal["r", "w"]:
        return self._mode  # type: ignore

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
    def broadcasts_image_config(self) -> bool:
        return self._broadcasts_image_config

    @broadcasts_image_config.setter
    @override
    def broadcasts_image_config(self, value: bool):
        self._broadcasts_image_config = value

    @property
    @override
    def updates_optics_group(self) -> bool:
        return self._updates_optics_group

    @updates_optics_group.setter
    @override
    def updates_optics_group(self, value: bool):
        self._updates_optics_group = value

    @property
    def inverts_rotation(self) -> bool:
        return self._inverts_rotation

    @inverts_rotation.setter
    def inverts_rotation(self, value: bool):
        self._inverts_rotation = value

    @override
    def __setitem__(
        self,
        index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"],
        value: ParticleParameterLike,
    ):
        raise NotImplementedError(
            "CryoSparcParticleParameterFile does not have a __setitem__ method"
        )

    @override
    def append(self, value: ParticleParameterLike):
        raise NotImplementedError(
            "append is not supported for CryoSparcParticleParameterFile"
        )

    @override
    def save(
        self,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "saving is not supported for CryoSparcParticleParameterFile"
        )


class CryoSparcParticleStackDataset(
    AbstractParticleStackDataset[ParticleStackInfo, ParticleStackLike]
):
    """A dataset that wraps a CSPARC particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    def __init__(
        self,
        parameter_file: AbstractParticleStarFile,
        path_to_relion_project: str | pathlib.Path,
        *,
        loads_parameters: bool = True,
    ):
        """**Arguments:**

        - `path_to_relion_project`:
            In CSPARC STAR files, only a relative path is added to the
            'rlnImageName' column. This is relative to the path to the
            "project", which is given by this parameter.
        - `parameter_file`:
            The `RelionParticleParameterFile`.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *images*. This is done by removing 'rlnImageName' from
            `parameter_file.csparc_metadata`, if it exists at all.
            does not have a column 'rlnImageName' and image files
            are not yet written.
            - If `mode = 'r'`, images are read from the 'rlnImageName'
            stored in the `parameter_file.csparc_metadata`.
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
        - `loads_parameters`:
            If `True`, load parameters and images. Otherwise, load only images.
        """
        # Set properties. First, core properties of the dataset, starting
        # those images are being used elsewhere
        particle_data = parameter_file.csparc_metadata

        self._parameter_file = parameter_file
        # ... properties common to reading and writing images
        self._path_to_relion_project = pathlib.Path(path_to_relion_project)
        # ... properties for reading images
        self._loads_parameters = loads_parameters
        # Now, initialize for `mode = 'r'` vs `mode = 'w'`
        images_exist = "blob/path" in particle_data.columns
        project_exists = self.path_to_relion_project.exists()
        if not images_exist:
            raise OSError(
                "Could not find column 'blob/path' in the CryoSparc metadata file. "
            )
        if not project_exists:
            raise FileNotFoundError(
                "`CryoSparcParticleStackDataset` opened in "
                "'mode = `r`', but the CSPARC project directory "
                "`path_to_relion_project` does not exist. "
                "To write images in a STAR file in a new CSPARC project, "
                "set `mode = 'w'`."
            )

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> ParticleStackInfo:
        if self.loads_parameters:
            # Load images and parameters. First, read parameters
            # and metadata from the STAR file
            loads_metadata = self.parameter_file.loads_metadata
            self.parameter_file.loads_metadata = True
            # ... read parameters
            parameters = self.parameter_file[index]
            # ... validate the metadata
            csparc_data_at_index = cast(pd.DataFrame, parameters["metadata"])
            _validate_csparc_image_name_exists(csparc_data_at_index, index)
            # ... reset boolean to original value
            self.parameter_file.loads_metadata = loads_metadata
            if not loads_metadata:
                parameters["metadata"] = None
            # ... grab shape
            shape = parameters["image_config"].shape
            # ... load stack of images
            images = _load_image_stack_from_mrc(
                shape, csparc_data_at_index, self.path_to_relion_project
            )
            # ... make sure images and parameters have same leading dim
            if parameters["pose"].offset_x_in_angstroms.ndim == 0:
                images = np.squeeze(images)

            return ParticleStackInfo(parameters=parameters, images=images)
        else:
            # Otherwise, do not read parameters to more efficiently read images. First,
            # validate the dataset index.
            n_rows = self.parameter_file.csparc_metadata.shape[0]
            _validate_dataset_index(type(self), index, n_rows)
            # ... read particle data at the requested indices
            particle_data = self.parameter_file.csparc_metadata
            csparc_data_at_index = particle_data.iloc[index]
            if isinstance(csparc_data_at_index, pd.Series):
                csparc_data_at_index = csparc_data_at_index.to_frame().T
            _validate_csparc_image_name_exists(csparc_data_at_index, index)
            # ... grab shape by reading the optics group
            shape = tuple(int(x) for x in csparc_data_at_index["blob/shape"][0])
            shape = cast(tuple[int, int], shape)
            # ... load stack of images
            images = _load_image_stack_from_mrc(
                shape, csparc_data_at_index, self.path_to_relion_project
            )
            # ... make sure image leading dim matches with index query
            if isinstance(index, int) or (
                isinstance(index, np.ndarray) and index.size == 0
            ):
                images = np.squeeze(images)

            return ParticleStackInfo(parameters=None, images=images)

    @override
    def __len__(self) -> int:
        return len(self.parameter_file)

    @property
    @override
    def parameter_file(self) -> AbstractParticleStarFile:
        return self._parameter_file

    @property
    def path_to_relion_project(self) -> pathlib.Path:
        return self._path_to_relion_project

    @property
    def loads_parameters(self) -> bool:
        return self._loads_parameters

    @loads_parameters.setter
    def loads_parameters(self, value: bool):
        self._loads_parameters = value

    @override
    def __setitem__(
        self, index: int | slice | Int[np.ndarray, ""], value: ParticleStackLike
    ):
        raise NotImplementedError(
            "CryoSparcParticleStackDataset does not have a __setitem__ method"
        )

    @override
    def append(self, value: ParticleStackLike):
        raise NotImplementedError(
            "append is not supported for CryoSparcParticleStackDataset"
        )

    @override
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: ParticleParameterLike | None = None,
    ):
        raise NotImplementedError(
            "writing images is not supported for CryoSparcParticleStackDataset"
        )


def _load_csparc_metadata(
    path_to_csparc_metadata: pathlib.Path,
    selection_filter: dict[str, Callable] | None,
) -> pd.DataFrame:
    if path_to_csparc_metadata.exists():
        csparc_metadata = read_csparc_data(path_to_csparc_metadata)
        _validate_csparc_metadata(csparc_metadata)
        # if selection_filter is not None:
        #    starfile_data = _select_particles(starfile_data, selection_filter)
    else:
        raise FileNotFoundError(
            f"CryoSparc metadata file {str(path_to_csparc_metadata)} does not exist."
        )

    return csparc_metadata


def _select_particles(
    csparc_metadata: dict[str, pd.DataFrame], selection_filter: dict[str, Callable]
) -> dict[str, pd.DataFrame]:
    particle_data = csparc_metadata
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
    csparc_metadata = particle_data[boolean_mask]

    return csparc_metadata


#
# STAR file reading
#
def _make_pytrees_from_csparc_metadata(
    csparc_data,
    broadcasts_image_config,
    loads_envelope,
    pad_options,
    inverts_rotation,
) -> tuple[BasicImageConfig, ContrastTransferTheory, EulerAnglePose]:
    float_dtype = jax.dtypes.canonicalize_dtype(float)
    # Load CTF parameters. First from particle data
    defocus_in_angstroms = (
        np.asarray(csparc_data["ctf/df1_A"], dtype=float_dtype)
        + np.asarray(csparc_data["ctf/df2_A"], dtype=float_dtype)
    ) / 2
    astigmatism_in_angstroms = np.asarray(
        csparc_data["ctf/df1_A"], dtype=float_dtype
    ) - np.asarray(csparc_data["ctf/df2_A"], dtype=float_dtype)
    astigmatism_angle = np.rad2deg(
        np.asarray(csparc_data["ctf/df_angle_rad"], dtype=float_dtype)
    )
    phase_shift = np.rad2deg(
        np.asarray(csparc_data["ctf/phase_shift_rad"], dtype=float_dtype)
    )
    # Then from optics data
    batch_shape = (
        () if defocus_in_angstroms.ndim == 0 else (defocus_in_angstroms.shape[0],)
    )
    spherical_aberration_in_mm = np.asarray(csparc_data["ctf/cs_mm"], dtype=float_dtype)
    amplitude_contrast_ratio = np.asarray(
        csparc_data["ctf/amp_contrast"], dtype=float_dtype
    )

    ctf_params = (
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
        spherical_aberration_in_mm,
        amplitude_contrast_ratio,
        phase_shift,
    )
    # Envelope parameters
    if loads_envelope:
        b_factor, scale_factor = (
            (
                np.asarray(csparc_data["ctf/bfactor"], dtype=float_dtype)
                if "ctf/bfactor" in csparc_data.keys()
                else None
            ),
            (
                np.asarray(csparc_data["ctf/scale"], dtype=float_dtype)
                if "ctf/scale" in csparc_data.keys()
                else None
            ),
        )
    else:
        b_factor, scale_factor = None, None
    # Image config parameters
    pixel_size = np.asarray(csparc_data["blob/psize_A"], dtype=float_dtype)
    voltage_in_kilovolts = np.asarray(csparc_data["ctf/accel_kv"], dtype=float_dtype)
    if not broadcasts_image_config and len(batch_shape) > 0:
        pixel_size = pixel_size[0]
        voltage_in_kilovolts = voltage_in_kilovolts[0]
    # Pose parameters. Values for the pose are optional,
    # so look to see if each key is present
    particle_keys = csparc_data.keys()
    # Read the pose. first, xy offsets

    if "alignments3D/shift" in particle_keys:
        csparc_pose_shift = np.array([s for s in csparc_data["alignments3D/shift"]])
    elif "alignments_class_0/shift" in particle_keys:
        csparc_pose_shift = np.array([s for s in csparc_data["alignments_class_0/shift"]])
    else:
        csparc_pose_shift = np.array([0.0, 0.0])

    if "alignments3D/pose" in particle_keys:
        csparc_pose_angles = np.array(
            [angles for angles in csparc_data["alignments3D/pose"]]
        )
    elif "alignments_class_0/pose" in particle_keys:
        csparc_pose_angles = np.array(
            [angles for angles in csparc_data["alignments_class_0/pose"]]
        )
    else:
        csparc_pose_angles = np.array([0.0, 0.0, 0.0])

    # TODO: support for helices, need a dataset to check how it works!

    # Now transform the angles and shift to the EulerAnglePose convention
    pose_rotation_matrix = _csparc_to_rotation_matrix(csparc_pose_angles)

    if len(batch_shape) > 0:
        pose_shift = csparc_pose_shift * pixel_size[:, None]
    else:
        pose_shift = csparc_pose_shift * pixel_size

    # Now, flip the sign of the translations and transpose rotations.
    maybe_make_full = lambda param, ndim: (
        np.full(batch_shape, param)
        if len(batch_shape) > 0 and param.ndim == ndim
        else param
    )

    def _tranpose_rot_matrix(rot_matrix):
        if rot_matrix.ndim == 2:
            return rot_matrix.T
        elif rot_matrix.ndim == 3:
            return np.transpose(rot_matrix, (0, 2, 1))

    pose_params = (
        -maybe_make_full(pose_shift, 1),
        _tranpose_rot_matrix(maybe_make_full(pose_rotation_matrix, 2)),
    )

    # Now, create cryojax objects. Do this on the CPU
    cpu_device = jax.devices(backend="cpu")[0]
    with jax.default_device(cpu_device):
        # First, create the `BasicImageConfig`
        if len(batch_shape) > 0:
            image_shape = tuple(int(x) for x in csparc_data["blob/shape"][0])
        else:
            image_shape = tuple(int(x) for x in csparc_data["blob/shape"])
        image_config = _make_config(
            image_shape, pixel_size, voltage_in_kilovolts, pad_options
        )

        # ... now the `ContrastTransferTheory`
        envelope = (
            _make_envelope_function(scale_factor, b_factor) if loads_envelope else None
        )
        transfer_theory_params = (*ctf_params, envelope)
        transfer_theory = _make_transfer_theory(*transfer_theory_params)  # type: ignore

        # ... finally the `EulerAnglePose`
        pose = _make_pose(*pose_params)
        if inverts_rotation:
            pose = _invert_rotation(pose)
    # Now, convert arrays to numpy in case the user wishes to do preprocessing
    pytree_dynamic, pytree_static = eqx.partition(
        (image_config, transfer_theory, pose), eqx.is_array
    )
    pytree_dynamic = jax.tree.map(lambda x: np.asarray(x), pytree_dynamic)
    image_config, transfer_theory, pose = eqx.combine(pytree_dynamic, pytree_static)

    return image_config, transfer_theory, pose


def _map_to_lie_algebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.shape[-1] == 3

    R_x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=v.dtype)

    R_y = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=v.dtype)

    R_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=v.dtype)
    R = (
        R_x * v[..., 0, None, None]
        + R_y * v[..., 1, None, None]
        + R_z * v[..., 2, None, None]
    )
    return R


def _csparc_to_rotation_matrix(csparc_angles):
    """
    Convert CryoSparc angles to rotation matrix.
    This code was adapted from cryoDRGN.
    """
    theta = np.linalg.norm(csparc_angles, ord=2, axis=-1, keepdims=True)
    # normalize K
    K = _map_to_lie_algebra(csparc_angles / theta)

    rotation_matrix = (
        np.eye(3, dtype=csparc_angles.dtype)
        + np.sin(theta)[..., None] * K
        + (1.0 - np.cos(theta))[..., None] * (K @ K)
    )
    return rotation_matrix


def _make_config(
    image_shape,
    pixel_size,
    voltage_in_kilovolts,
    pad_options,
):
    return eqx.tree_at(
        lambda x: (x.pixel_size, x.voltage_in_kilovolts),
        BasicImageConfig(image_shape, 1.0, 1.0, pad_options=pad_options),
        (pixel_size, voltage_in_kilovolts),
    )


def _make_pose(shift, rotation_matrix):
    _make_fn = (
        lambda _shift, _rotation_matrix: EulerAnglePose.from_rotation_and_translation(
            SO3.from_matrix(_rotation_matrix), _shift
        )
    )
    if shift.ndim == 2:
        _make_fn = eqx.filter_vmap(_make_fn)
    return _make_fn(shift, rotation_matrix)


def _make_envelope_function(amp, b_factor):
    if b_factor is None and amp is None:
        warnings.warn(
            "`loads_envelope` was set to True, but no envelope parameters were found. "
            "Setting envelope as None. "
            "Make sure your starfile is correctly formatted or set "
            "`loads_envelope=False`."
        )
        return None

    elif b_factor is None and amp is not None:
        return eqx.tree_at(lambda x: x.value, FourierConstant(1.0), amp)
    else:
        if amp is None:
            amp = np.asarray(1.0) if b_factor.ndim == 0 else np.ones_like(b_factor)
        return eqx.tree_at(
            lambda x: (x.amplitude, x.b_factor),
            FourierGaussian(1.0, 1.0),
            (amp, b_factor),
        )


def _make_transfer_theory(defocus, astig, angle, sph, ac, ps, env=None):
    ctf = eqx.tree_at(
        lambda x: (
            x.defocus_in_angstroms,
            x.astigmatism_in_angstroms,
            x.astigmatism_angle,
            x.spherical_aberration_in_mm,
        ),
        AstigmaticCTF(),
        (defocus, astig, angle, sph),
    )
    transfer_theory = ContrastTransferTheory(
        ctf, envelope=env, amplitude_contrast_ratio=0.1, phase_shift=0.0
    )

    return eqx.tree_at(
        lambda x: (x.amplitude_contrast_ratio, x.phase_shift), transfer_theory, (ac, ps)
    )


def _invert_rotation(pose: EulerAnglePose) -> EulerAnglePose:
    negate_angle = lambda angle: ((-angle + 180) % 360) - 180
    return eqx.tree_at(
        lambda x: (x.phi_angle, x.theta_angle, x.psi_angle),
        pose,
        (
            negate_angle(pose.psi_angle),
            negate_angle(pose.theta_angle),
            negate_angle(pose.phi_angle),
        ),
    )


def _load_image_stack_from_mrc(
    shape: tuple[int, int],
    particle_dataframe_at_index: pd.DataFrame,
    path_to_relion_project: str | pathlib.Path,
) -> Float[np.ndarray, "... y_dim x_dim"]:
    # Load particle image stack rlnImageName
    mrc_filenames_and_indices = particle_dataframe_at_index[
        ["blob/path", "blob/idx"]
    ].copy()
    mrc_filenames_and_indices["idx_in_df"] = mrc_filenames_and_indices.index.copy()
    try:
        mrc_filenames_and_indices.loc[:, "blob/path"] = mrc_filenames_and_indices[
            "blob/path"
        ].astype(str)
    except ValueError as err:
        raise TypeError(
            "The 'blob/path' entry in the CryoSparc metadata could not be converted"
            f"to string. Caught error:\n{err}"
        )

    # groupby filename to get indices
    grouped_filenames = mrc_filenames_and_indices.groupby("blob/path").agg(list)

    # Allocate memory for stack
    n_images = len(mrc_filenames_and_indices)
    image_stack = np.empty((n_images, *shape), dtype=float)
    # Loop over filenames to fill stack
    for filename in grouped_filenames.index:
        # Get the MRC indices
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
            idx_in_filename = np.array(
                grouped_filenames.loc[filename, "blob/idx"], dtype=int
            )
            idx_in_df = np.array(grouped_filenames.loc[filename, "idx_in_df"], dtype=int)
            image_stack[idx_in_df] = (
                mrc_data if mrc_ndim == 2 else mrc_data[idx_in_filename]
            )

    return image_stack


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


def _validate_csparc_metadata(csparc_metadata: pd.DataFrame):
    required_particle_keys, _ = zip(*CSPARC_REQUIRED_PARTICLE_ENTRIES)
    if not set(required_particle_keys).issubset(set(csparc_metadata.keys())):
        raise ValueError(
            "Missing required keys in starfile 'particles' group. "
            f"Required keys are {required_particle_keys}."
        )


def _validate_csparc_image_name_exists(particle_data, index):
    if "blob/path" not in particle_data.columns:
        raise OSError(
            "Tried to read CryoSparc metadata file for "
            f"`RelionParticleStackDataset` index = {index}, "
            "but no entry found for 'blob/path'."
        )
