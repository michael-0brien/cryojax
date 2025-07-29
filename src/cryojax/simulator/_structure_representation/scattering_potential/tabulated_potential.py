from typing import Optional
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from ....constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from ....internal import NDArrayLike, error_if_negative
from ..._structure_conversion import AbstractRealVoxelRendering
from ..atomic_structure import AbstractIndependentAtomStructure
from ..common_functions import gaussians_to_real_voxels
from .base_potential import AbstractScatteringPotential


class PengScatteringFactorParameters(eqx.Module, strict=True):
    """Scattering factor parameters from Peng et al. (1996)."""

    a: Float[Array, " n_atoms 5"]
    b: Float[Array, " n_atoms 5"]

    def __init__(self, atomic_numbers: Int[np.ndarray, " n_atoms"]):
        scattering_factor_parameter_table = (
            read_peng_element_scattering_factor_parameter_table()
        )
        scattering_factor_parameter_dict = get_tabulated_scattering_factor_parameters(
            atomic_numbers, scattering_factor_parameter_table
        )
        self.a = jnp.asarray(scattering_factor_parameter_dict["a"], dtype=float)
        self.b = jnp.asarray(scattering_factor_parameter_dict["b"], dtype=float)


class AbstractTabulatedScatteringPotential(AbstractScatteringPotential, strict=True):
    """Abstract interface for the spatial potential energy distribution of a
    scatterer.

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

    scattering_factor_parameters: eqx.AbstractVar[PyTree]
    b_factors: eqx.AbstractVar[Optional[Float[Array, " n_atoms"]]]


class PengTabulatedAtomicPotential(
    AbstractTabulatedScatteringPotential,
    AbstractIndependentAtomStructure,
    AbstractRealVoxelRendering,
    strict=True,
):
    """The scattering potential parameterized as a mixture of five gaussians
    per atom, through work by Lian-Mao Peng.

    **References:**

    - Peng, L-M. "Electron atomic scattering factors and scattering potentials of crystals."
      Micron 30.6 (1999): 625-648.
    - Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic
      scattering factors." Acta Crystallographica Section A: Foundations of Crystallography
      52.2 (1996): 257-276.
    """  # noqa: E501

    atom_positions: Float[Array, "n_atoms 3"]
    scattering_factor_parameters: PengScatteringFactorParameters
    b_factors: Optional[Float[Array, " n_atoms"]]

    def __init__(
        self,
        atom_positions: Float[NDArrayLike, "n_atoms 3"],
        scattering_factor_parameters: PengScatteringFactorParameters,
        b_factors: Optional[Float[NDArrayLike, " n_atoms"]] = None,
    ):
        self.atom_positions = jnp.asarray(atom_positions, dtype=float)
        self.scattering_factor_parameters = scattering_factor_parameters
        if b_factors is None:
            self.b_factors = None
        else:
            self.b_factors = error_if_negative(jnp.asarray(b_factors, dtype=float))

    @override
    def as_real_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        options: dict = {"batch_size": 1, "n_batches": 1},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        """Return a voxel grid of the potential in real space.

        Through the work of Peng et al. (1996), tabulated elastic electron scattering factors
        are defined as

        $$f^{(e)}(\\mathbf{q}) = \\sum\\limits_{i = 1}^5 a_i \\exp(- b_i |\\mathbf{q}|^2),$$

        where $a_i$ is stored as `PengAtomicPotential.scattering_factor_a` and $b_i$ is
        stored as `PengAtomicPotential.scattering_factor_b` for the scattering vector $\\mathbf{q}$.
        Under usual scattering approximations (i.e. the first-born approximation),
        the rescaled electrostatic potential energy $U(\\mathbf{r})$ is then given by
        $4 \\pi \\mathcal{F}^{-1}[f^{(e)}(\\boldsymbol{\\xi} / 2)](\\mathbf{r})$, which is computed
        analytically as

        $$U(\\mathbf{r}) = 4 \\pi \\sum\\limits_{i = 1}^5 \\frac{a_i}{(2\\pi (b_i / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{r} - \\mathbf{r}'|^2}{2 (b_i / 8 \\pi^2)}),$$

        where $\\mathbf{r}'$ is the position of the atom. Including an additional B-factor (denoted by
        $B$ and stored as `PengAtomicPotential.b_factors`) gives the expression for the potential
        $U(\\mathbf{r})$ of a single atom type and its fourier transform pair $\\tilde{U}(\\boldsymbol{\\xi}) \\equiv \\mathcal{F}[U](\\boldsymbol{\\xi})$,

        $$U(\\mathbf{r}) = 4 \\pi \\sum\\limits_{i = 1}^5 \\frac{a_i}{(2\\pi ((b_i + B) / 8 \\pi^2))^{3/2}} \\exp(- \\frac{|\\mathbf{r} - \\mathbf{r}'|^2}{2 ((b_i + B) / 8 \\pi^2)}),$$

        $$\\tilde{U}(\\boldsymbol{\\xi}) = 4 \\pi \\sum\\limits_{i = 1}^5 a_i \\exp(- (b_i + B) |\\boldsymbol{\\xi}|^2 / 4) \\exp(2 \\pi i \\boldsymbol{\\xi}\\cdot\\mathbf{r}'),$$

        where $\\mathbf{q} = \\boldsymbol{\\xi} / 2$ gives the relationship between the wave vector and the
        scattering vector.

        In practice, for a discretization on a grid with voxel size $\\Delta r$ and grid point $\\mathbf{r}_{\\ell}$,
        the potential is evaluated as the average value inside the voxel

        $$U_{\\ell} = 4 \\pi \\frac{1}{\\Delta r^3} \\sum\\limits_{i = 1}^5 a_i \\prod\\limits_{j = 1}^3 \\int_{r^{\\ell}_j-\\Delta r/2}^{r^{\\ell}_j+\\Delta r/2} dr_j \\ \\frac{1}{{\\sqrt{2\\pi ((b_i + B) / 8 \\pi^2)}}} \\exp(- \\frac{(r_j - r'_j)^2}{2 ((b_i + B) / 8 \\pi^2)}),$$

        where $j$ indexes the components of the spatial coordinate vector $\\mathbf{r}$. The above expression is evaluated using the error function as

        $$U_{\\ell} = 4 \\pi \\frac{1}{(2 \\Delta r)^3} \\sum\\limits_{i = 1}^5 a_i \\prod\\limits_{j = 1}^3 \\textrm{erf}(\\frac{r_j^{\\ell} - r'_j + \\Delta r / 2}{\\sqrt{2 ((b_i + B) / 8\\pi^2)}}) - \\textrm{erf}(\\frac{r_j^{\\ell} - r'_j - \\Delta r / 2}{\\sqrt{2 ((b_i + B) / 8\\pi^2)}}).$$

        **Arguments:**

        - `shape`:
            The shape of the resulting voxel grid.
        - `voxel_size`:
            The voxel size of the resulting voxel grid.
        - `options`:
            Advanced options for rendering. This is a dictionary
            with the following keys:
            - "batch_size":
                The number of z-planes to evaluate in parallel with
                `jax.vmap`. By default, `1`.
            - "n_batches":
                The number of iterations used to evaluate the volume,
                where the iteration is taken over groups of atoms.
                This is useful if `batch_size_for_z_planes = 1`
                and GPU memory is exhausted. By default, `1`.

        **Returns:**

        The rescaled potential $U_{\\ell}$ as a voxel grid of shape `shape`
        and voxel size `voxel_size`.
        """  # noqa: E501
        gaussian_amplitudes = self.scattering_factor_parameters.a
        if self.b_factors is None:
            gaussian_widths = self.scattering_factor_parameters.b
        else:
            gaussian_widths = (
                self.scattering_factor_parameters.b + self.b_factors[:, None]
            )
        return gaussians_to_real_voxels(
            shape,
            jnp.asarray(voxel_size, dtype=float),
            self.atom_positions,
            gaussian_amplitudes,
            gaussian_widths,
            **options,
        )
