from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ...constants import (
    read_peng_element_scattering_factor_parameter_table,
)
from ...internal import NDArrayLike
from .base_potential import AbstractIndependentAtomPotential


class AbstractTabulatedPotential(AbstractIndependentAtomPotential):
    b_factors: eqx.AbstractVar[Optional[Float[Array, " n_atoms"]]]


class PengTabulatedPotential(AbstractTabulatedPotential):
    """The scattering potential parameterized as a mixture of five gaussians
    per atom, through work by Lian-Mao Peng.

    **References:**

    - Peng, L-M. "Electron atomic scattering factors and scattering potentials of crystals."
      Micron 30.6 (1999): 625-648.
    - Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic
      scattering factors." Acta Crystallographica Section A: Foundations of Crystallography
      52.2 (1996): 257-276.
    """  # noqa: E501

    scattering_factor_a: Float[Array, " n_atoms 5"]
    scattering_factor_b: Float[Array, " n_atoms 5"]
    b_factors: Optional[Float[Array, " n_atoms"]]

    def __init__(
        self,
        atomic_numbers: Int[NDArrayLike, " n_atoms"],
        b_factors: Optional[Float[NDArrayLike, " n_atoms"]] = None,
    ):
        scattering_factor_parameter_table = (
            read_peng_element_scattering_factor_parameter_table()
        )
        self.scattering_factor_a = jnp.asarray(
            scattering_factor_parameter_table["a"].data[atomic_numbers, ...], dtype=float
        )
        self.scattering_factor_b = jnp.asarray(
            scattering_factor_parameter_table["b"].data[atomic_numbers, ...], dtype=float
        )
        if b_factors is None:
            self.b_factors = None
        else:
            self.b_factors = jnp.asarray(b_factors, dtype=float)
