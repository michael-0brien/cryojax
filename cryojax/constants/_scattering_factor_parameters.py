"""
Routines for loading atomic structures.
Large amounts of the code are adapted from the ioSPI package
"""

from typing import Literal

import equinox as eqx
import numpy as np
from jaxtyping import Float, Int


class PengScatteringFactorParameters(eqx.Module, strict=True):
    """A convenience wrapper for loading electron scattering factor
    parameters from "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996).

    To access scattering factors $a_i$ and $b_i$ given in
    the citation,

    ```python
    from cryojax.io import read_atoms_from_pdb
    from cryojax.constants import PengScatteringFactorParameters

    # Load positions of atoms and one-hot encoded atom names
    atom_positions, atomic_numbers = read_atoms_from_pdb(...)
    parameters = PengScatteringFactorParameters(atomic_numbers)
    print(parameters.a, parameters.b)  # a_i and b_i
    ```
    """

    a: Float[np.ndarray, " n_atoms 5"]
    b: Float[np.ndarray, " n_atoms 5"]

    def __init__(self, atomic_numbers: Int[np.ndarray, " n_atoms"]):
        """**Arguments:**

        - `atomic_numbers`:
            The atom types as an integer array.
        """
        parameters = _PENG_PARAMETER_TABLE[atomic_numbers]
        self.a = parameters[:, 0, :]
        self.b = parameters[:, 1, :]


def check_atomic_numbers_supported(atomic_numbers: Int[np.ndarray, " _"]):
    """Throw an error if `atomic_numbers` contains values not
    supported by
    [`cryojax.constants.extract_scattering_factor_parameters`][].
    """
    unique_atomic_numbers = np.unique(atomic_numbers).tolist()
    if not set(unique_atomic_numbers).issubset(set(_SUPPORTED_ATOMIC_NUMBERS)):
        raise ValueError(
            "Found unsupported atomic numbers when running "
            "function `cryojax.constants.check_atomic_numbers_supported`. "
            f"Supported atomic numbers are `{_SUPPORTED_ATOMIC_NUMBERS}`, "
            f"but found values {unique_atomic_numbers}."
        )


def extract_scattering_factor_parameters(
    atomic_numbers: Int[np.ndarray, " n_atoms"],
    tabulation: Literal["peng"] = "peng",
) -> PengScatteringFactorParameters:
    """Gets the parameters for the scattering factor for each atom in
    `atom_types`.

    !!! warning

        Only elements found in PDB files, e.g. for proteins, DNA/RNA,
        and small molecules are supported.
        That is, `atomic_numbers` may have the following values:

            - 1: Hydrogen
            - 6: Carbon
            - 7: Nitrogen
            - 8: Oxygen
            - 9: Fluorine
            - 11: Sodium
            - 12: Magnesium
            - 15: Phosphorus
            - 16: Sulfur
            - 17: Chlorine
            - 19: Potassium
            - 20: Calcium
            - 25: Manganese
            - 26: Iron
            - 27: Colbalt
            - 29: Copper
            - 30: Zinc

        If `atomic_numbers` contains values not in this list, scattering
        factor parameters are returned with `numpy.nan` values or an
        index out of bounds error will be thrown. To check if
        `atomic_numbers` is a valid array, use
        [`cryojax.constants.check_atomic_numbers_supported`][].

    **Arguments:**

    - `atomic_numbers`:
        Atomic numbers for each atom type.
    - `tabulation`:
        Which electron scattering factor tabulation to choose.
        Currenly, only "peng" for
        [`cryojax.constants.PengScatteringFactorParameters`][]
        is supported.

    **Returns:**

    The scattering factor parameters corresponding to `'tabulation'`
    queried at `atomic_numbers`.
    """  # noqa: E501
    if tabulation == "peng":
        return PengScatteringFactorParameters(atomic_numbers)
    else:
        raise ValueError(
            "Only `tabulation = 'peng'` is supported. "
            f"Instead, got `tabulation = {tabulation}`."
        )


_SUPPORTED_ATOMIC_NUMBERS = [
    1,
    6,
    7,
    8,
    9,
    11,
    12,
    15,
    16,
    17,
    19,
    20,
    25,
    26,
    27,
    29,
    30,
]

_PENG_PARAMETER_TABLE = np.full((31, 2, 5), np.nan)

_PENG_PARAMETER_TABLE[1] = [
    [0.0349, 0.1201, 0.197, 0.0573, 0.1195],
    [0.5347, 3.5867, 12.3471, 18.9525, 38.6269],
]
_PENG_PARAMETER_TABLE[6] = [
    [0.0893, 0.2563, 0.757, 1.0487, 0.3575],
    [0.2465, 1.71, 6.4094, 18.6113, 50.2523],
]
_PENG_PARAMETER_TABLE[7] = [
    [0.1022, 0.3219, 0.7982, 0.8197, 0.1715],
    [0.2451, 1.7481, 6.1925, 17.3894, 48.1431],
]
_PENG_PARAMETER_TABLE[8] = [
    [0.0974, 0.2921, 0.691, 0.699, 0.2039],
    [0.2067, 1.3815, 4.6943, 12.7105, 32.4726],
]
_PENG_PARAMETER_TABLE[9] = [
    [0.1083, 0.3175, 0.6487, 0.5846, 0.1421],
    [0.2057, 1.3439, 4.2788, 11.3932, 28.7881],
]
_PENG_PARAMETER_TABLE[11] = [
    [0.2142, 0.6853, 0.7692, 1.6589, 1.4482],
    [0.3334, 2.3446, 10.083, 48.3037, 138.27],
]
_PENG_PARAMETER_TABLE[12] = [
    [0.2314, 0.6866, 0.9677, 2.1882, 1.1339],
    [0.3278, 2.272, 10.9241, 39.2898, 101.9748],
]
_PENG_PARAMETER_TABLE[15] = [
    [0.2548, 0.6106, 1.4541, 2.3204, 0.8477],
    [0.2908, 1.874, 8.5176, 24.3434, 63.2996],
]
_PENG_PARAMETER_TABLE[16] = [
    [0.2497, 0.5628, 1.3899, 2.1865, 0.7715],
    [0.2681, 1.6711, 7.0267, 19.5377, 50.3888],
]
_PENG_PARAMETER_TABLE[17] = [
    [0.2443, 0.5397, 1.3919, 2.0197, 0.6621],
    [0.2468, 1.5242, 6.1537, 16.6687, 42.3086],
]
_PENG_PARAMETER_TABLE[19] = [
    [0.4115, 1.4031, 2.2784, 2.6742, 2.2162],
    [0.3703, 3.3874, 13.1029, 68.9592, 194.4329],
]
_PENG_PARAMETER_TABLE[20] = [
    [0.4054, 1.388, 2.1602, 3.7532, 2.2063],
    [0.3499, 3.0991, 11.9608, 53.9353, 142.3892],
]
_PENG_PARAMETER_TABLE[25] = [
    [0.3796, 1.2094, 1.7815, 2.542, 1.5937],
    [0.2699, 2.0455, 7.4726, 31.0604, 91.5622],
]
_PENG_PARAMETER_TABLE[26] = [
    [0.3946, 1.2725, 1.7031, 2.314, 1.4795],
    [0.2717, 2.0443, 7.6007, 29.9714, 86.2265],
]
_PENG_PARAMETER_TABLE[27] = [
    [0.4118, 1.3161, 1.6493, 2.193, 1.283],
    [0.2742, 2.0372, 7.7205, 29.968, 84.9383],
]
_PENG_PARAMETER_TABLE[29] = [
    [0.4314, 1.3208, 1.5236, 1.4671, 0.8562],
    [0.2694, 1.9223, 7.3474, 28.9892, 90.6246],
]
_PENG_PARAMETER_TABLE[30] = [
    [0.4288, 1.2646, 1.4472, 1.8294, 1.0934],
    [0.2593, 1.7998, 6.75, 25.586, 73.5284],
]
