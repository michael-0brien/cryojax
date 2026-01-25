"""
Implementation of operators on images in real-space.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from ...jax_util import FloatLike, NDArrayLike, error_if_not_positive


class AbstractRealOperator(eqx.Module, strict=True):
    """
    The base class for all real operators.

    By convention, operators should be defined to
    have units of inverse area (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the `__call__` method.
    """

    @abstractmethod
    def __call__(  # pyright: ignore
        self,
        coordinate_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError

    def __add__(self, other) -> "AbstractRealOperator":
        if isinstance(other, AbstractRealOperator):
            return _SumRealOperator(self, other)
        return _SumRealOperator(self, RealConstant(other))

    def __radd__(self, other) -> "AbstractRealOperator":
        if isinstance(other, AbstractRealOperator):
            return _SumRealOperator(other, self)
        return _SumRealOperator(RealConstant(other), self)

    def __sub__(self, other) -> "AbstractRealOperator":
        if isinstance(other, AbstractRealOperator):
            return _DiffRealOperator(self, other)
        return _DiffRealOperator(self, RealConstant(other))

    def __rsub__(self, other) -> "AbstractRealOperator":
        if isinstance(other, AbstractRealOperator):
            return _DiffRealOperator(other, self)
        return _DiffRealOperator(RealConstant(other), self)

    def __mul__(self, other) -> "AbstractRealOperator":
        if isinstance(other, AbstractRealOperator):
            return _ProductRealOperator(self, other)
        return _ProductRealOperator(self, RealConstant(other))

    def __rmul__(self, other) -> "AbstractRealOperator":
        if isinstance(other, AbstractRealOperator):
            return _ProductRealOperator(other, self)
        return _ProductRealOperator(RealConstant(other), self)


class _SumRealOperator(AbstractRealOperator, strict=True):
    """A helper to represent the sum of two operators."""

    operator1: AbstractRealOperator
    operator2: AbstractRealOperator

    @override
    def __call__(
        self,
        coordinate_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.operator1(coordinate_grid) * self.operator2(coordinate_grid)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _DiffRealOperator(AbstractRealOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractRealOperator
    operator2: AbstractRealOperator

    @override
    def __call__(
        self,
        coordinate_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.operator1(coordinate_grid) * self.operator2(coordinate_grid)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class _ProductRealOperator(AbstractRealOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractRealOperator
    operator2: AbstractRealOperator

    @override
    def __call__(
        self,
        coordinate_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.operator1(coordinate_grid) * self.operator2(coordinate_grid)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class RealGaussian(AbstractRealOperator, strict=True):
    """This operator is a normalized gaussian in real space

    $$g(r) = \\frac{\\kappa}{2\\pi \\beta} \\exp(- (r - r_0)^2 / (2 \\sigma))$$

    where $r^2 = x^2 + y^2$.
    """

    amplitude: Float[Array, ""]
    variance: Float[Array, ""]
    offset: Float[Array, " _"] | None

    def __init__(
        self,
        amplitude: FloatLike = 1.0,
        variance: FloatLike = 1.0,
        offset: Float[NDArrayLike, " _"] | Sequence[float] | None = None,
    ):
        """**Arguments:**

        - `amplitude`:
            The amplitude of the operator, equal to $\\kappa$
            in the above equation.
        - `variance`:
            The variance of the gaussian, equal to $\\sigma$
            in the above equation.
        - `offset`:
            An offset to the origin, equal to $r_0$
            in the above equation.
        """
        self.amplitude = jnp.asarray(amplitude, dtype=float)
        self.variance = error_if_not_positive(jnp.asarray(variance, dtype=float))
        if offset is not None:
            self.offset = jnp.asarray(offset, dtype=float)
        else:
            self.offset = None

    @override
    def __call__(
        self, coordinate_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        ndim = coordinate_grid.ndim - 1
        offset = jnp.zeros((ndim,), dtype=float) if self.offset is None else self.offset
        r_squared = jnp.sum((coordinate_grid - offset) ** 2, axis=-1)
        scaling = (
            self.amplitude / jnp.sqrt(2 * jnp.pi * self.variance) ** ndim
        ) * jnp.exp(-0.5 * r_squared / self.variance)
        return scaling


class RealConstant(AbstractRealOperator, strict=True):
    """An operator that is a constant."""

    value: Float[Array, "..."]

    def __init__(self, value: float | Float[NDArrayLike, "..."]):
        """**Arguments:**

        - `value`: The value of the constant
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(
        self,
        coordinate_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, ""]:
        del coordinate_grid
        return self.value
