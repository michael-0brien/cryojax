import pathlib

import numpy as np
import pandas as pd


def read_csparc_data(
    path_to_csparc_metadata: pathlib.Path,
) -> pd.DataFrame:
    _validate_filename(path_to_csparc_metadata)

    metadata = np.load(path_to_csparc_metadata, allow_pickle=True)
    data_entries = [metadata.dtype.names[i] for i in range(len(metadata.dtype.names))]
    csparc_data = pd.DataFrame(
        {
            entry: [metadata[j][entry] for j in range(len(metadata))]
            for entry in data_entries
        }
    )
    return csparc_data


def _validate_filename(filename: str | pathlib.Path):
    suffixes = pathlib.Path(filename).suffixes
    if not (len(suffixes) == 1 and suffixes[0] == ".cs"):
        raise OSError(
            f"Tried to read a CryoSparc metadata file, "
            "but the filename does not include a '.cs' "
            f"suffix. Got filename '{filename}'."
        )
