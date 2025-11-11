from collections.abc import Sequence
from typing import TypeVar
from typing_extensions import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from ....constants import PengScatteringFactorParameters
from ....jax_util import FloatLike, NDArrayLike
from ....ndimage.operators import AbstractFourierOperator
from ..._pose import AbstractPose
from .base_representations import AbstractAtomVolume


T = TypeVar("T")


class PengScatteringFactor(AbstractFourierOperator, strict=True):
    a: Float[Array, "5"]
    b: Float[Array, "5"]
    b_factor: Float[Array, ""] | None

    def __init__(
        self,
        a: Float[NDArrayLike, "5"],
        b: Float[NDArrayLike, "5"],
        b_factor: FloatLike | None = None,
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
    position_pytree: PyTree[Float[Array, "_ 3"]]
    scattering_factor_pytree: PyTree[AbstractFourierOperator]

    def __init__(
        self,
        position_pytree: PyTree[Float[NDArrayLike, "_ 3"], "T"],
        scattering_factor_pytree: PyTree[AbstractFourierOperator, "T"],
    ):
        self.position_pytree = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=float), position_pytree
        )
        self.scattering_factor_pytree = scattering_factor_pytree

    def __check_init__(self):
        if jax.tree.structure(self.position_pytree) != jax.tree.structure(
            self.scattering_factor_pytree,
            is_leaf=lambda x: isinstance(x, AbstractFourierOperator),
        ):
            raise ValueError(
                "When instantiating an `IndependentAtomVolume`, found "
                "that the pytree structures of `positions_pytree` and "
                "`scattering_factor_pytree` were not equal."
            )

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new potential with rotated `positions`."""
        rotate_fn = lambda pos: pose.rotate_coordinates(pos, inverse=inverse)
        return eqx.tree_at(
            lambda x: x.position_pytree,
            self,
            jax.tree.map(rotate_fn, self.position_pytree),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        offset_in_angstroms = pose.offset_in_angstroms
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        translate_fn = lambda pos: pos + offset_in_angstroms
        return eqx.tree_at(
            lambda x: x.position_pytree,
            self,
            jax.tree.map(translate_fn, self.position_pytree),
        )

    @classmethod
    def from_tabulated_parameters(
        cls,
        positions_by_element: tuple[Float[NDArrayLike, "_ 3"], ...],
        parameters: PengScatteringFactorParameters,
        *,
        b_factor_by_element: FloatLike | tuple[FloatLike, ...] | None = None,
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
        if b_factor_by_element is not None:
            if isinstance(b_factor_by_element, Sequence):
                if len(b_factor_by_element) != n_elements:
                    raise ValueError(
                        "When constructing an `IndependentAtomVolume` via "
                        "`from_tabulated_parameters`, found that "
                        "`len(b_factor_by_element) != len(positions_by_element)`. "
                        "Make sure that `b_factor_by_element` is a tuple with "
                        "length matching the number of atom types."
                    )
            else:
                b_factor_by_element = tuple(
                    b_factor_by_element for _ in range(n_elements)
                )
            scattering_factors_by_element = tuple(
                PengScatteringFactor(a_i, b_i, b_factor)
                for a_i, b_i, b_factor in zip(
                    parameters.a, parameters.b, b_factor_by_element
                )
            )
        else:
            scattering_factors_by_element = tuple(
                PengScatteringFactor(a_i, b_i)
                for a_i, b_i in zip(parameters.a, parameters.b)
            )
        return cls(positions_by_element, scattering_factors_by_element)
