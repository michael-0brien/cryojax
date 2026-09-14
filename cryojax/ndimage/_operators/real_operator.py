"""
Implementation of operators on images in real-space.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from ..._internal import error_if_not_positive, leaf_asarray
from ...jax_util import FloatLike, NDArrayLike


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

    spatial_dims: eqx.AbstractClassVar[list[int]]

    @abstractmethod
    def __call__(  # pyright: ignore
        self,
        coordinates: Float[Array, "..."],
    ) -> Inexact[Array, "..."]:
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

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    @override
    def __call__(self, coordinates: Float[Array, "..."]) -> Inexact[Array, "..."]:
        return self.operator1(coordinates) + self.operator2(coordinates)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _DiffRealOperator(AbstractRealOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractRealOperator
    operator2: AbstractRealOperator

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    @override
    def __call__(self, coordinates: Float[Array, "..."]) -> Inexact[Array, "..."]:
        return self.operator1(coordinates) - self.operator2(coordinates)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class _ProductRealOperator(AbstractRealOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractRealOperator
    operator2: AbstractRealOperator

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    @override
    def __call__(self, coordinates: Float[Array, "..."]) -> Inexact[Array, "..."]:
        return self.operator1(coordinates) * self.operator2(coordinates)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class RealGaussian(AbstractRealOperator, strict=True):
    """This operator is a normalized gaussian in real space

    $$g(r) = \\frac{\\kappa}{2\\pi \\beta} \\exp(- (r - r_0)^2 / (2 \\sigma))$$

    where $r^2 = x^2 + y^2$.
    """

    amplitude: Float[NDArrayLike, "..."]
    variance: Float[NDArrayLike, "..."]
    offset: Float[NDArrayLike, " _"] | None

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(
        self,
        amplitude: FloatLike = 1.0,
        variance: FloatLike = 1.0,
        offset: FloatLike | Float[NDArrayLike, "... _"] | Sequence[float] | None = None,
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
        self.amplitude = leaf_asarray(amplitude, dtype=float)
        self.variance = leaf_asarray(variance, dtype=float)
        if offset is not None:
            offset = leaf_asarray(offset, dtype=float)
            self.offset = offset[None] if offset.ndim == 0 else offset
        else:
            self.offset = None

    @override
    def __call__(self, coordinates: Float[Array, "..."]) -> Float[Array, "..."]:
        coordinates, ndim, flag = _standardize_coordinates(coordinates)
        variance = jnp.asarray(self.variance)
        offset = (
            jnp.zeros((ndim,), dtype=float)
            if self.offset is None
            else jnp.asarray(self.offset)
        )
        r_squared = jnp.sum((coordinates - offset) ** 2, axis=-1)
        scaling = (
            jnp.asarray(self.amplitude)
            / jnp.sqrt(2 * jnp.pi * error_if_not_positive(variance)) ** ndim
        ) * jnp.exp(-0.5 * r_squared / variance)
        return _standardize_output(scaling, flag=flag)


class RealConstant(AbstractRealOperator, strict=True):
    """An operator that is a constant."""

    value: Float[NDArrayLike, "..."]

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(self, value: float | Float[NDArrayLike, "..."]):
        """**Arguments:**

        - `value`: The value of the constant
        """
        self.value = leaf_asarray(value, dtype=float)

    @override
    def __call__(self, coordinates: Float[Array, "..."]) -> Float[Array, ""]:
        del coordinates
        return jnp.asarray(self.value)


def _standardize_coordinates(x: Array):
    flag = False
    if x.ndim == 0:
        flag = True
        x = x[None, None]
    elif x.ndim == 1:
        x = x[:, None]
    ndim = x.shape[-1]
    return x, ndim, flag


def _standardize_output(out: Array, *, flag: bool):
    return out[0] if flag else out
