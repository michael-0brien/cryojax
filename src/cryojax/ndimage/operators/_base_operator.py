"""
Base classes for image operators.
"""

from abc import abstractmethod
from typing import Any, Callable
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


class AbstractImageOperator(eqx.Module, strict=True):
    """Abstract base class for image operators in `cryojax`."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        raise NotImplementedError

    def __add__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return SumImageOperator(self, other)
        return SumImageOperator(self, Constant(other))

    def __radd__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return SumImageOperator(other, self)
        return SumImageOperator(Constant(other), self)

    def __sub__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return DiffImageOperator(self, other)
        return DiffImageOperator(self, Constant(other))

    def __rsub__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return DiffImageOperator(other, self)
        return DiffImageOperator(Constant(other), self)

    def __mul__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return ProductImageOperator(self, other)
        return ProductImageOperator(self, Constant(other))

    def __rmul__(self, other) -> "AbstractImageOperator":
        if isinstance(other, AbstractImageOperator):
            return ProductImageOperator(other, self)
        return ProductImageOperator(Constant(other), self)


class Constant(AbstractImageOperator, strict=True):
    """An operator that is a constant."""

    value: Float[Array, "..."]

    def __init__(self, value: float | Float[Array, "..."]):
        """**Arguments:**

        - `value`: The value of the constant
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Float[Array, ""]:
        return self.value


class CustomOperator(AbstractImageOperator, strict=True):
    """An operator that calls a custom function."""

    fn: Callable[..., Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]]
    kwargs: dict[str, Any]

    def __init__(
        self,
        fn: Callable[
            ..., Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
        ],
        **kwargs: Any,
    ):
        """**Arguments:**

        - `fn`: The `Callable` wrapped into a `AbstractImageOperator`.
        """
        self.fn = fn
        self.kwargs = kwargs

    @override
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.fn(*args, **kwargs, **self.kwargs)


class Empirical(AbstractImageOperator, strict=True):
    """This operator stores and returns an array, rather than
    computing one from a model.
    """

    array: (
        Inexact[Array, "... y_dim x_dim"]
        | Inexact[Array, "... z_dim y_dim x_dim"]
        | Float[Array, "..."]
    )

    def __init__(
        self,
        array: (
            Inexact[Array, "... y_dim x_dim"]
            | Inexact[Array, "... z_dim y_dim x_dim"]
            | Float[Array, "..."]
        ),
    ):
        """**Arguments:**

        - `array`: The array to be returned upon calling `Empirical`.
        """
        self.array = jnp.asarray(array)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Inexact[Array, "y_dim x_dim"]:
        """Return the scaled and offset measurement."""
        return self.array


class SumImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the sum of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.operator1(*args, **kwargs) + self.operator2(*args, **kwargs)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class DiffImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.operator1(*args, **kwargs) - self.operator2(*args, **kwargs)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class ProductImageOperator(AbstractImageOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractImageOperator
    operator2: AbstractImageOperator

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.operator1(*args, **kwargs) * self.operator2(*args, **kwargs)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"
