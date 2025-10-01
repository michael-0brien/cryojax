"""
Routines for loading atomic structures.
Large amounts of the code are adapted from the ioSPI package
"""

import importlib.resources as pkg_resources
import os
from typing import Optional

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Float, Int


def get_tabulated_scattering_factor_parameters(
    atom_types: Int[np.ndarray, " n_atoms"],
    scattering_factor_parameter_table: Optional[xr.Dataset] = None,
) -> dict[str, Float[np.ndarray, " n_atoms n_scattering_factors"]]:
    """Gets the parameters for the scattering factor for each atom in
    `atom_types`.

    **Arguments:**

    - `atom_identitites`:
        Array containing the index of the one-hot encoded atom names.
        By default, Hydrogen is "1", Carbon is "6", Nitrogen is "7", etc.
    - `scattering_factor_parameter_table`:
        The table of scattering factors as an `xarray.Dataset`. By default, this
        is the tabulation from "Robust Parameterization of Elastic and
        Absorptive Electron Atomic Scattering Factors" by Peng et al. (1996),
        given by [`read_peng_element_scattering_factor_parameter_table`](https://michael-0brien.github.io/cryojax/api/constants/scattering_factor_parameters/#cryojax.constants.read_peng_element_scattering_factor_parameter_table).

    **Returns:**

    The particular scattering factor parameters stored in
    `scattering_factor_parameter_table` for `atom_types`.
    """  # noqa: E501
    if scattering_factor_parameter_table is None:
        scattering_factor_parameter_table = read_peng_scattering_factor_parameter_table()
    return {
        str(k): np.asarray(v.data[np.asarray(atom_types), ...])
        for k, v in scattering_factor_parameter_table.items()
    }


def read_peng_scattering_factor_parameter_table() -> xr.Dataset:
    r"""Function to load the atomic scattering factor parameter
    table from "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996).

    **Returns:**

    The parameter table for parameters $\{a_i\}_{i = 1}^5$ and $\{b_i\}_{i = 1}^5$
    for each atom, described in the above reference. This is stored as an
    `xarray.Dataset`.
    """
    with pkg_resources.as_file(
        pkg_resources.files("cryojax").joinpath("constants")
    ) as path:
        scattering_factor_parameters = jnp.load(
            os.path.join(path, "peng1996_element_params.npy")
        )
    dims = ("n_elements", "n_gaussians")
    scattering_factor_parameter_table = xr.Dataset(
        data_vars=dict(
            a=xr.DataArray(scattering_factor_parameters[0], dims=dims),
            b=xr.DataArray(scattering_factor_parameters[1], dims=dims),
        ),
        attrs=dict(description="Scattering factor parameters from Peng et al. (1996)"),
    )

    return scattering_factor_parameter_table
