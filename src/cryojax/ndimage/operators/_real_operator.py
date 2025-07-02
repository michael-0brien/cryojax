"""
Implementation of operators on images in real-space.
"""

from abc import abstractmethod
from typing import overload
from typing_extensions import override

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ...internal import error_if_not_positive
from ._base_operator import AbstractImageOperator


class AbstractRealOperator(AbstractImageOperator, strict=True):
    """
    The base class for all real operators.

    By convention, operators should be defined to
    have units of inverse area (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the ``__call__`` method.
    """

    @overload
    @abstractmethod
    def __call__(
        self, coordinate_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    @abstractmethod
    def __call__(  # type: ignore
        self, coordinate_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    @abstractmethod
    def __call__(  # pyright: ignore
        self,
        coordinate_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError


RealOperatorLike = AbstractRealOperator | AbstractImageOperator


class Gaussian2D(AbstractRealOperator, strict=True):
    """This operator represents a simple gaussian in 2D.
    Specifically, this is

    $$g(r) = \\frac{\\kappa}{2\\pi \\beta} \\exp(- (r - r_0)^2 / (2 \\sigma))$$

    where $r^2 = x^2 + y^2$.
    """

    amplitude: Float[Array, ""]
    variance: Float[Array, ""]
    offset: Float[Array, "2"]

    def __init__(
        self,
        amplitude: float | Float[Array, ""] = 1.0,
        variance: float | Float[Array, ""] = 1.0,
        offset: Float[Array | np.ndarray, "2"] | tuple[float, float] = (0.0, 0.0),
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
        self.offset = jnp.asarray(offset, dtype=float)

    @override
    def __call__(
        self, coordinate_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        r_sqr = jnp.sum((coordinate_grid - self.offset) ** 2, axis=-1)
        scaling = (self.amplitude / jnp.sqrt(2 * jnp.pi * self.variance)) * jnp.exp(
            -0.5 * r_sqr / self.variance
        )
        return scaling
