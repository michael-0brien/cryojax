"""Implementation of an AbstractFourierOperator. Put simply, these are
functions commonly applied to images in fourier space.

Opposed to a AbstractFilter, a AbstractFourierOperator is computed at
runtime---not upon initialization. AbstractFourierOperators also do not
have a rule for how they should be applied to images.

These classes are modified from the library ``tinygp``.
"""

from abc import abstractmethod
from typing import overload
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from ...internal import error_if_negative, error_if_not_positive
from ._base_operator import AbstractImageOperator


class AbstractFourierOperator(AbstractImageOperator, strict=True):
    """
    The base class for all fourier-based operators.

    By convention, operators should be defined to
    be dimensionless (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the ``__call__`` method.
    """

    @overload
    @abstractmethod
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Inexact[Array, "y_dim x_dim"]: ...

    @overload
    @abstractmethod
    def __call__(  # type: ignore
        self, frequency_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Inexact[Array, "z_dim y_dim x_dim"]: ...

    @abstractmethod
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError


FourierOperatorLike = AbstractFourierOperator | AbstractImageOperator


class ZeroMode(AbstractFourierOperator, strict=True):
    """This operator returns a constant in the zero mode."""

    value: Float[Array, ""]

    def __init__(self, value: float | Float[Array, ""] = 0.0):
        """**Arguments:**

        - `value`: The value of the zero mode.
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        N1, N2 = frequency_grid.shape[0:-1]
        return jnp.zeros((N1, N2)).at[0, 0].set(self.value)


class FourierExp2D(AbstractFourierOperator, strict=True):
    r"""This operator, in real space, represents a
    function equal to an exponential decay, given by

    .. math::
        g(|r|) = \frac{\kappa}{2 \pi \xi^2} \exp(- |r| / \xi),

    where :math:`|r| = \sqrt{x^2 + y^2}` is a radial coordinate.
    Here, :math:`\xi` has dimensions of length and :math:`g(r)`
    has dimensions of inverse area. The power spectrum from such
    a correlation function (in two-dimensions) is given by its
    Hankel transform pair

    .. math::
        P(|k|) = \frac{\kappa}{2 \pi \xi^3} \frac{1}{(\xi^{-2} + |k|^2)^{3/2}}.

    Here :math:`\kappa` is a scale factor and :math:`\xi` is a length
    scale.
    """

    amplitude: Float[Array, ""]
    length_scale: Float[Array, ""]

    def __init__(
        self,
        amplitude: float | Float[Array, ""] = 1.0,
        length_scale: float | Float[Array, ""] = 1.0,
    ):
        """**Arguments:**

        - `amplitude`: The amplitude of the operator, equal to $\\kappa$
                in the above equation.
        - `length_scale`: The length scale of the operator, equal to $\\xi$
                    in the above equation.
        """
        self.amplitude = jnp.asarray(amplitude, dtype=float)
        self.length_scale = error_if_not_positive(jnp.asarray(length_scale, dtype=float))

    @override
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        scaling = 1.0 / (k_sqr + jnp.divide(1, (self.length_scale) ** 2)) ** 1.5
        scaling *= jnp.divide(self.amplitude, 2 * jnp.pi * self.length_scale**3)
        return scaling


class Lorenzian(AbstractFourierOperator, strict=True):
    r"""This operator is the Lorenzian, given

    .. math::
        P(|k|) = \frac{\kappa}{\xi^2} \frac{1}{(\xi^{-2} + |k|^2)}.

    Here :math:`\kappa` is a scale factor and :math:`\xi` is a length
    scale.
    """

    amplitude: Float[Array, ""]
    length_scale: Float[Array, ""]

    def __init__(
        self,
        amplitude: float | Float[Array, ""] = 1.0,
        length_scale: float | Float[Array, ""] = 1.0,
    ):
        """**Arguments:**

        - `amplitude`: The amplitude of the operator, equal to $\\kappa$
                in the above equation.
        - `length_scale`: The length scale of the operator, equal to $\\xi$
                    in the above equation.
        """
        self.amplitude = jnp.asarray(amplitude, dtype=float)
        self.length_scale = error_if_not_positive(jnp.asarray(length_scale, dtype=float))

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(  # type: ignore
        self, frequency_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        scaling = 1.0 / (k_sqr + jnp.divide(1, self.length_scale**2))
        scaling *= jnp.divide(self.amplitude, self.length_scale**2)
        return scaling


class FourierGaussian(AbstractFourierOperator, strict=True):
    r"""This operator represents a simple gaussian.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta k^2 / 4),

    where :math:`k^2 = k_x^2 + k_y^2` is the length of the
    wave vector. Here, :math:`\beta` has dimensions of length
    squared.
    """

    amplitude: Float[Array, ""]
    b_factor: Float[Array, ""]

    def __init__(
        self,
        amplitude: float | Float[Array, ""] = 1.0,
        b_factor: float | Float[Array, ""] = 1.0,
    ):
        """**Arguments:**

        - `amplitude`:
            The amplitude of the operator, equal to $\\kappa$
            in the above equation.
        - `b_factor`:
            The B-factor of the gaussian, equal to $\\beta$
            in the above equation.
        """
        self.amplitude = jnp.asarray(amplitude, dtype=float)
        self.b_factor = error_if_not_positive(jnp.asarray(b_factor, dtype=float))

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(  # type: ignore
        self, frequency_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        scaling = self.amplitude * jnp.exp(-0.25 * self.b_factor * k_sqr)
        return scaling


class FourierGaussianWithRadialOffset(AbstractFourierOperator, strict=True):
    r"""This operator represents a gaussian with a radial offset.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta (|k| - m)^2 / 4),

    where :math:`k^2 = k_x^2 + k_y^2` is the length of the
    wave vector. Here, :math:`\beta` has dimensions of length
    squared.
    """

    amplitude: Float[Array, ""]
    b_factor: Float[Array, ""]
    offset: Float[Array, ""]

    def __init__(
        self,
        amplitude: float | Float[Array, ""] = 1.0,
        b_factor: float | Float[Array, ""] = 1.0,
        offset: float | Float[Array, ""] = 0.0,
    ):
        """**Arguments:**

        - `amplitude`:
            The amplitude of the operator, equal to $\\kappa$
            in the above equation.
        - `b_factor`:
            The B-factor of the gaussian, equal to $\\beta$
            in the above equation.
        - `offset`:
            The radial offset of the gaussian.
        """
        self.amplitude = jnp.asarray(amplitude, dtype=float)
        self.b_factor = error_if_not_positive(jnp.asarray(b_factor, dtype=float))
        self.offset = error_if_negative(jnp.asarray(offset, dtype=float))

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(  # type: ignore
        self, frequency_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        k = jnp.linalg.norm(frequency_grid, axis=-1)
        scaling = self.amplitude * jnp.exp(-0.25 * self.b_factor * (k - self.offset) ** 2)
        return scaling
