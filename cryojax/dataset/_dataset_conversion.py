import pathlib
from typing import cast

import equinox as eqx

from ..simulator import AxisAnglePose, EulerAnglePose
from . import CryoSparcParticleParameterFile, RelionParticleParameterFile
from ._particle_data.relion import ParticleParameterLike, _format_number_for_filename


def convert_csparc_to_relion(
    cryosparc_parameter_file: CryoSparcParticleParameterFile,
    path_to_starfile: str | pathlib.Path,
    exists_ok: bool = False,
) -> RelionParticleParameterFile:
    """Convert a `CryoSparcParticleParameterFile` to a
    `RelionParticleParameterFile`.

    **Arguments:**

    - `path_to_starfile`:
        The path to the output Relion starfile.

    **Returns:**
    A `RelionParticleParameterFile` containing the converted particle
    parameters from the input `CryoSparcParticleParameterFile`.

    """

    relion_particle_parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exists_ok=exists_ok,
        # inverts_rotation=True,
    )

    # set particle parameters
    parameters = cryosparc_parameter_file[:]
    parameters["pose"] = _convert_axisangle_to_euler(parameters["pose"])
    parameters = cast(ParticleParameterLike, parameters)
    relion_particle_parameter_file.append(parameters)

    particle_filenames = cryosparc_parameter_file.csparc_metadata["blob/path"].astype(str)
    particle_indices = cryosparc_parameter_file.csparc_metadata["blob/idx"]

    rln_image_names = [
        _format_number_for_filename(int(i + 1), n_characters=6)
        + "@"
        + particle_filenames[i]
        for i in particle_indices
    ]

    # set image names
    relion_particle_parameter_file.starfile_data["particles"]["rlnImageName"] = (
        rln_image_names
    )

    return relion_particle_parameter_file


@eqx.filter_vmap
def _convert_axisangle_to_euler(pose: AxisAnglePose) -> EulerAnglePose:
    return EulerAnglePose.from_rotation_and_translation(
        rotation=pose.rotation, offset_in_angstroms=pose.offset_in_angstroms
    )
