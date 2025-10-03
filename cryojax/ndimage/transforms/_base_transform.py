"""
Base classes for image transformations.
"""

import abc
from typing_extensions import override

import equinox as eqx
from jaxtyping import Array, Inexact


class AbstractImageTransform(eqx.Module, strict=True):
    """Base class for computing and applying an `Array` to an image."""

    @abc.abstractmethod
    def __call__(
        self, image: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError

    def __mul__(self, other) -> "AbstractImageTransform":
        return ProductImageTransform(transform1=self, transform2=other)

    def __rmul__(self, other) -> "AbstractImageTransform":
        return ProductImageTransform(transform1=other, transform2=self)


class ProductImageTransform(AbstractImageTransform, strict=True):
    """A helper to represent the product of two transforms."""

    transform1: AbstractImageTransform
    transform2: AbstractImageTransform

    def __init__(
        self,
        transform1: AbstractImageTransform,
        transform2: AbstractImageTransform,
    ):
        self.transform1 = transform1
        self.transform2 = transform2

    @override
    def __call__(
        self, image: Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.transform1(self.transform2(image))

    def __repr__(self):
        return f"{repr(self.transform1)} * {repr(self.transform2)}"
