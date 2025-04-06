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
from equinox import field
from jaxtyping import Array, Float, Inexact

from ...internal import error_if_negative, error_if_not_positive
from ._operator import AbstractImageOperator


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
    def __call__(
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

    value: Float[Array, ""] = field(default=0.0, converter=jnp.asarray)

    @override
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        N1, N2 = frequency_grid.shape[0:-1]
        return jnp.zeros((N1, N2)).at[0, 0].set(self.value)


ZeroMode.__init__.__doc__ = """**Arguments:**

- `value`: The value of the zero mode.
"""


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

    amplitude: Float[Array, ""] = field(default=1.0, converter=jnp.asarray)
    length_scale: Float[Array, ""] = field(default=1.0, converter=error_if_not_positive)

    @override
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]:
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        scaling = 1.0 / (k_sqr + jnp.divide(1, (self.length_scale) ** 2)) ** 1.5
        scaling *= jnp.divide(self.amplitude, 2 * jnp.pi * self.length_scale**3)
        return scaling


FourierExp2D.__init__.__doc__ = """**Arguments:**

- `amplitude`: The amplitude of the operator, equal to $\\kappa$
           in the above equation.
- `length_scale`: The length scale of the operator, equal to $\\xi$
              in the above equation.
"""


class Lorenzian(AbstractFourierOperator, strict=True):
    r"""This operator is the Lorenzian, given

    .. math::
        P(|k|) = \frac{\kappa}{\xi^2} \frac{1}{(\xi^{-2} + |k|^2)}.

    Here :math:`\kappa` is a scale factor and :math:`\xi` is a length
    scale.
    """

    amplitude: Float[Array, ""] = field(default=1.0, converter=jnp.asarray)
    length_scale: Float[Array, ""] = field(default=1.0, converter=error_if_not_positive)

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
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


Lorenzian.__init__.__doc__ = """**Arguments:**

- `amplitude`: The amplitude of the operator, equal to $\\kappa$
           in the above equation.
- `length_scale`: The length scale of the operator, equal to $\\xi$
              in the above equation.
"""


class FourierGaussian(AbstractFourierOperator, strict=True):
    r"""This operator represents a simple gaussian.
    Specifically, this is

    .. math::
        P(k) = \kappa \exp(- \beta k^2 / 4),

    where :math:`k^2 = k_x^2 + k_y^2` is the length of the
    wave vector. Here, :math:`\beta` has dimensions of length
    squared.
    """

    amplitude: Float[Array, ""] = field(default=1.0, converter=jnp.asarray)
    b_factor: Float[Array, ""] = field(default=1.0, converter=error_if_negative)

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
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


FourierGaussian.__init__.__doc__ = """**Arguments:**

- `amplitude`: The amplitude of the operator, equal to $\\kappa$
           in the above equation.
- `b_factor`: The variance of the real-space gaussian, equal to $\\beta$
              in the above equation.
"""

class Parkhurst2024_Gaussian(AbstractFourierOperator, strict=True):
    r"""
    This operator represents the sum of two gaussians. 
    Specifically, this is

    .. math:: 
        P(k) = a_1 \exp(-(k-m_1)^2/(2 s_1^2)) + a_2 \exp(-(k-m_2)^2/(2 s_2^2)),

    Where default values given by Parkhurst et al. (2024) are:
    a_1 = 0.199
    s_1 = 0.731
    m_1 = 0
    a_2 = 0.801
    s_2 = 0.081
    m_2 = 1/2.88 (Å^(-1))
    """

    a1: Float[Array, ""] = field(default=0.199, converter=jnp.asarray)
    s1: Float[Array, ""] = field(default=0.731, converter=error_if_negative)
    m1: Float[Array, ""] = field(default=0, converter=error_if_negative)

    a2: Float[Array, ""] = field(default=0.801, converter=jnp.asarray)
    s2: Float[Array, ""] = field(default=0.081, converter=error_if_negative)
    m2: Float[Array, ""] = field(default=1/2.88, converter=error_if_negative)

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "y_dim x_dim 2"]
    ) -> Float[Array, "y_dim x_dim"]: ...

    @overload
    def __call__(
        self, frequency_grid: Float[Array, "z_dim y_dim x_dim 3"]
    ) -> Float[Array, "z_dim y_dim x_dim"]: ...

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:

        # SCALE a1, a2, s1, s2 based on pixel size in 
        
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        scaling = self.a1 * jnp.exp(- (k_sqr - self.m1)**2 / (2*self.s1**2)) + self.a2 * jnp.exp(- (k_sqr - self.m2)**2 / (2*self.s2**2))
        return scaling


Parkhurst2024_Gaussian.__init__.__doc__ = """**Arguments:**
- `a1`: The amplitude of the first gaussian
- `s1`: The variance of the first gaussian
- `m1`: The center of the first gaussian
- `a2`: The amplitude of the second gaussian
- `s2`: The variance of the second gaussian
- `m2`: The center of the second gaussian
"""