import abc
from typing import Any, Optional
from typing_extensions import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from ...constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from ...internal import NDArrayLike


class AbstractScatteringPotential(
    eqx.Module, strict=eqx.StrictConfig(force_abstract=True)
):
    """Abstract interface for generating a spatial potential energy distribution.

    !!! info
        In, `cryojax`, potentials should be built in units of *inverse length squared*,
        $[L]^{-2}$. This rescaled potential is defined to be

        $$U(\\mathbf{r}) = \\frac{2 m e}{\\hbar^2} V(\\mathbf{r}),$$

        where $V$ is the electrostatic potential energy, $\\mathbf{r}$ is a positional
        coordinate, $m$ is the electron mass, and $e$ is the electron charge.

        For a single atom, this rescaled potential has the advantage that under usual
        scattering approximations (i.e. the first-born approximation), the
        fourier transform of this quantity is closely related to tabulated electron scattering
        factors. In particular, for a single atom with scattering factor $f^{(e)}(\\mathbf{q})$
        and scattering vector $\\mathbf{q}$, its rescaled potential is equal to

        $$U(\\mathbf{r}) = 4 \\pi \\mathcal{F}^{-1}[f^{(e)}(\\boldsymbol{\\xi} / 2)](\\mathbf{r}),$$

        where $\\boldsymbol{\\xi} = 2 \\mathbf{q}$ is the wave vector coordinate and
        $\\mathcal{F}^{-1}$ is the inverse fourier transform operator in the convention

        $$\\mathcal{F}[f](\\boldsymbol{\\xi}) = \\int d^3\\mathbf{r} \\ \\exp(2\\pi i \\boldsymbol{\\xi}\\cdot\\mathbf{r}) f(\\mathbf{r}).$$

        The rescaled potential $U$ gives the following time-independent schrodinger equation
        for the scattering problem,

        $$(\\nabla^2 + k^2) \\psi(\\mathbf{r}) = - U(\\mathbf{r}) \\psi(\\mathbf{r}),$$

        where $k$ is the incident wavenumber of the electron beam.

        **References**:

        - For the definition of the rescaled potential, see
        Chapter 69, Page 2003, Equation 69.6 from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*
        - To work out the correspondence between the rescaled potential and the electron
        scattering factors, see the supplementary information from *Vulović, Miloš, et al.
        "Image formation modeling in cryo-electron microscopy." Journal of structural
        biology 183.1 (2013): 19-32.*
    """  # noqa: E501


class AbstractTabulatedScatteringPotential(AbstractScatteringPotential, strict=True):
    """Abstract class for a tabulated scattering potential."""

    @classmethod
    @abc.abstractmethod
    def from_scattering_factor_parameters(
        cls,
        atom_positions: Float[NDArrayLike, "n_atoms 3"],
        parameters: PyTree[Any],
        extra_b_factors: Optional[Float[NDArrayLike, " n_atoms"]] = None,
    ) -> Self:
        raise NotImplementedError


class PengScatteringFactorParameters(eqx.Module, strict=True):
    """A convenience wrapper for instantiating the
    scattering factor parameters from Peng et al. (1996).

    To access scattering factors $a_i$ and $b_i$ given in
    the citation,

    ```python
    from cryojax.io import read_atoms_from_pdb
    from cryojax.simulator import PengScatteringFactorParameters

    # Load positions of atoms and one-hot encoded atom names
    atom_positions, atom_identities = read_atoms_from_pdb(...)
    parameters = PengScatteringFactorParameters(atom_identities)
    print(parameters.a)  # a_i
    print(parameters.b)  # b_i
    ```
    """

    a: Float[Array, " n_atoms 5"]
    b: Float[Array, " n_atoms 5"]

    def __init__(self, atom_identities: Int[np.ndarray, " n_atoms"]):
        """**Arguments:**

        - `atom_types`:
            The atom types as an integer array.
        """
        scattering_factor_parameter_table = (
            read_peng_element_scattering_factor_parameter_table()
        )
        scattering_factor_parameter_dict = get_tabulated_scattering_factor_parameters(
            atom_identities, scattering_factor_parameter_table
        )
        self.a = jnp.asarray(scattering_factor_parameter_dict["a"], dtype=float)
        self.b = jnp.asarray(scattering_factor_parameter_dict["b"], dtype=float)
