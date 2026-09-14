"""
Base classes for image transformations.
"""

import abc
from typing import ClassVar
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from ..._internal import leaf_asarray
from ...jax_util import FloatLike, NDArrayLike


class AbstractImageTransform(eqx.Module, strict=True):
    """Base class for computing and applying an `Array` to an image."""

    is_real_space: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def __call__(
        self, image: Inexact[Array, "*batch _ _"] | Inexact[Array, "*batch _ _ _"]
    ) -> Inexact[Array, "*batch _ _"] | Inexact[Array, "*batch _ _ _"]:
        raise NotImplementedError

    def __mul__(self, other) -> "AbstractImageTransform":
        if self.is_real_space:
            return _RealProductImageTransform(transform1=self, transform2=other)
        else:
            return _FourierProductImageTransform(transform1=self, transform2=other)

    def __rmul__(self, other) -> "AbstractImageTransform":
        if self.is_real_space:
            return _RealProductImageTransform(transform1=other, transform2=self)
        else:
            return _FourierProductImageTransform(transform1=other, transform2=self)


class _AbstractProductImageTransform(AbstractImageTransform, strict=True):
    """A helper to represent the product of two transforms."""

    transform1: AbstractImageTransform
    transform2: AbstractImageTransform

    def __check_init__(self):
        if self.transform1.is_real_space != self.transform2.is_real_space:
            raise TypeError(
                "Cannot compose two `AbstractImageTransforms` "
                " with unequal attributes `is_real_space`. Type "
                f"{self.transform1.__class__.__name__} had "
                f"`is_real_space = {self.transform1.is_real_space}`, "
                f"but {self.transform2.__class__.__name__} had "
                f"`is_real_space = {self.transform2.is_real_space}`."
            )

    @override
    def __call__(
        self, image: Inexact[Array, "*batch _ _"] | Inexact[Array, "*batch _ _ _"]
    ) -> Inexact[Array, "*batch _ _"] | Inexact[Array, "*batch _ _ _"]:
        return self.transform1(self.transform2(image))

    def __repr__(self):
        return f"{repr(self.transform1)} * {repr(self.transform2)}"


class _RealProductImageTransform(_AbstractProductImageTransform):
    is_real_space: ClassVar[bool] = True


class _FourierProductImageTransform(_AbstractProductImageTransform):
    is_real_space: ClassVar[bool] = False


class ScaleImage(AbstractImageTransform, strict=True):
    scale: Float[NDArrayLike, ""]
    offset: Float[NDArrayLike, ""]

    is_real_space: ClassVar[bool] = True

    def __init__(self, scale: FloatLike = 1.0, offset: FloatLike = 0.0):
        self.scale = leaf_asarray(scale, dtype=float)
        self.offset = leaf_asarray(offset, dtype=float)

    @override
    def __call__(
        self,
        image: (
            Inexact[Array, "*batch y_dim x_dim"]
            | Inexact[Array, "*batch z_dim y_dim x_dim"]
        ),
    ) -> (
        Inexact[Array, "*batch y_dim x_dim"] | Inexact[Array, "*batch z_dim y_dim x_dim"]
    ):
        return jnp.asarray(self.scale) * image + jnp.asarray(self.offset)
