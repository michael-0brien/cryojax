from typing import TypeVar
from typing_extensions import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from ....constants import PengScatteringFactorParameters
from ....jax_util import NDArrayLike
from ....ndimage.operators import AbstractFourierOperator
from .base_representations import AbstractAtomVolume


T = TypeVar("T")


class PengScatteringFactor(AbstractFourierOperator, strict=True):
    a: Float[Array, "5"]
    b: Float[Array, "5"]
    b_factor = Float[Array, ""]

    def __init__(
        self,
        a: Float[NDArrayLike, "5"],
        b: Float[NDArrayLike, "5"],
        b_factor: float | Float[NDArrayLike, ""] | None = None,
    ):
        self.a = jnp.asarray(a, dtype=float)
        self.b = jnp.asarray(b, dtype=float)
        self.b_factor = None if b_factor is None else jnp.asarray(b_factor, dtype=float)

    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ):
        q_squared = jnp.sum(frequency_grid**2, axis=-1)
        b_factor = 0.0 if self.b_factor is None else self.b_factor
        gaussian_fn = lambda _a, _b: _a * jnp.exp(-0.25 * (_b + b_factor) * q_squared)
        return jnp.sum(jax.vmap(gaussian_fn)(self.a, self.b), axis=0)


class IndependentAtomVolume(AbstractAtomVolume, strict=True):
    atom_positions: PyTree[Float[Array, "_ 3"]]
    scattering_factors: PyTree[AbstractFourierOperator]

    def __init__(
        self,
        atom_positions: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        scattering_factors: PyTree[AbstractFourierOperator, "T"],
    ):
        self.atom_positions = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=float), atom_positions
        )
        self.scattering_factors = scattering_factors

    @classmethod
    def from_tabulated_parameters(
        cls,
        positions_by_element: tuple[Float[NDArrayLike, "_ 3"], ...],
        parameters: PengScatteringFactorParameters,
        *,
        b_factors_by_element: tuple[float | Float[NDArrayLike, ""], ...] | None = None,
    ) -> Self:
        n_elements = len(positions_by_element)
        a, b = parameters.a, parameters.b
        if a.shape[0] != n_elements or b.shape[0] != n_elements:
            raise ValueError(
                "When constructing an `IndependentAtomVolume` via "
                "`from_tabulated_parameters`, found that "
                "`parameters.a.shape[0] != len(positions_by_element)` "
                "or `parameters.b.shape[0] != len(positions_by_element)`. "
                "Make sure that `a` and `b` correspond to the element types "
                "in `positions_by_element.`"
            )
        if b_factors_by_element is not None:
            if len(b_factors_by_element) != n_elements:
                raise ValueError(
                    "When constructing an `IndependentAtomVolume` via "
                    "`from_tabulated_parameters`, found that "
                    "`len(b_factor_by_element) != len(positions_by_element)`. "
                    "Make sure that `b_factor_by_element` is a tuple with "
                    "length matching the number of atom types."
                )
            scattering_factors_by_element = tuple(
                PengScatteringFactor(a_i, b_i, b_factor)
                for a_i, b_i, b_factor in zip(
                    parameters.a, parameters.b, b_factors_by_element
                )
            )
        else:
            scattering_factors_by_element = tuple(
                PengScatteringFactor(a_i, b_i)
                for a_i, b_i in zip(parameters.a, parameters.b)
            )
        return cls(positions_by_element, scattering_factors_by_element)
