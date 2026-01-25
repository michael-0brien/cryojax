"""Implementation of an `AbstractFourierOperator`. Put simply, these are
functions commonly applied to images in fourier space.

Opposed to a `AbstractFilter`, a `AbstractFourierOperator` is computed at
runtime---not upon initialization. `AbstractFourierOperators` also do not
have a rule for how they should be applied to images and can be composed
with other operators.

These classes are modified from the library `tinygp`.
"""

import functools
import operator
from abc import abstractmethod
from collections.abc import Callable
from typing import Any
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from ...jax_util import FloatLike, error_if_negative, error_if_not_positive


class AbstractFourierOperator(eqx.Module, strict=True):
    """
    The base class for all fourier-based operators.

    By convention, operators should be defined to
    be dimensionless (up to a scale factor).

    To create a subclass,

        1) Include the necessary parameters in
           the class definition.
        2) Overrwrite the `__call__` method.
    """

    @abstractmethod
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        raise NotImplementedError

    def __add__(self, other) -> "AbstractFourierOperator":
        if isinstance(other, AbstractFourierOperator):
            return _SumFourierOperator(self, other)
        return _SumFourierOperator(self, FourierConstant(other))

    def __radd__(self, other) -> "AbstractFourierOperator":
        if isinstance(other, AbstractFourierOperator):
            return _SumFourierOperator(other, self)
        return _SumFourierOperator(FourierConstant(other), self)

    def __sub__(self, other) -> "AbstractFourierOperator":
        if isinstance(other, AbstractFourierOperator):
            return _DiffFourierOperator(self, other)
        return _DiffFourierOperator(self, FourierConstant(other))

    def __rsub__(self, other) -> "AbstractFourierOperator":
        if isinstance(other, AbstractFourierOperator):
            return _DiffFourierOperator(other, self)
        return _DiffFourierOperator(FourierConstant(other), self)

    def __mul__(self, other) -> "AbstractFourierOperator":
        if isinstance(other, AbstractFourierOperator):
            return _ProductFourierOperator(self, other)
        return _ProductFourierOperator(self, FourierConstant(other))

    def __rmul__(self, other) -> "AbstractFourierOperator":
        if isinstance(other, AbstractFourierOperator):
            return _ProductFourierOperator(other, self)
        return _ProductFourierOperator(FourierConstant(other), self)


class _SumFourierOperator(AbstractFourierOperator, strict=True):
    """A helper to represent the sum of two operators."""

    operator1: AbstractFourierOperator
    operator2: AbstractFourierOperator

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.operator1(frequency_grid) + self.operator2(frequency_grid)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _DiffFourierOperator(AbstractFourierOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractFourierOperator
    operator2: AbstractFourierOperator

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.operator1(frequency_grid) - self.operator2(frequency_grid)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class _ProductFourierOperator(AbstractFourierOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractFourierOperator
    operator2: AbstractFourierOperator

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.operator1(frequency_grid) * self.operator2(frequency_grid)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class CustomFourierOperator(AbstractFourierOperator, strict=True):
    """An operator that calls a custom function."""

    fn: Callable[..., Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]]
    args: Any
    kwargs: Any

    def __init__(
        self,
        fn: Callable[
            ..., Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]
        ],
        *args: Any,
        **kwargs: Any,
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Inexact[Array, "y_dim x_dim"] | Inexact[Array, "z_dim y_dim x_dim"]:
        return self.fn(frequency_grid, *self.args, **self.kwargs)


CustomFourierOperator.__init__.__doc__ = """**Arguments:**

- `fn`:
    The `Callable` wrapped into a `AbstractFourierOperator`.
    Has signature `out = fn(frequency_grid, *args, **kwargs)`
- `args`:
    Passed to `fn`.
- `kwargs`:
    Passed to `fn`.
"""


class FourierDC(AbstractFourierOperator, strict=True):
    """This operator returns a constant in the DC component."""

    value: Float[Array, ""]

    def __init__(self, value: FloatLike = 0.0):
        """**Arguments:**

        - `value`: The value of the zero mode.
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        return jnp.zeros(frequency_grid.shape[0:-1]).at[0, 0].set(self.value)


class FourierConstant(AbstractFourierOperator, strict=True):
    """An operator that is a constant."""

    value: Float[Array, "..."]

    def __init__(self, value: float | Float[Array, "..."]):
        """**Arguments:**

        - `value`: The value of the constant
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "..."]:
        del frequency_grid
        return self.value


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
        amplitude: FloatLike = 1.0,
        length_scale: FloatLike = 1.0,
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

    def __init__(self, amplitude: FloatLike = 1.0, b_factor: FloatLike = 1.0):
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

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        k_sqr = jnp.sum(frequency_grid**2, axis=-1)
        gaussian = self.amplitude * jnp.exp(-0.25 * self.b_factor * k_sqr)

        return gaussian


class PeakedFourierGaussian(AbstractFourierOperator, strict=True):
    r"""This operator represents a gaussian with a peak
    at a given frequency shell.
    """

    amplitude: Float[Array, ""]
    b_factor: Float[Array, ""]
    radial_peak: Float[Array, ""]

    def __init__(
        self,
        amplitude: FloatLike = 1.0,
        b_factor: FloatLike = 1.0,
        radial_peak: FloatLike = 0.0,
    ):
        """**Arguments:**

        - `amplitude`:
            The amplitude of the operator, equal to $\\kappa$
            in the above equation.
        - `b_factor`:
            The B-factor of the gaussian, equal to $\\beta$
            in the above equation.
        - `radial_peak`:
            The frequency shell of the gaussian peak.
        """
        self.amplitude = jnp.asarray(amplitude, dtype=float)
        self.b_factor = error_if_not_positive(jnp.asarray(b_factor, dtype=float))
        self.radial_peak = error_if_negative(jnp.asarray(radial_peak, dtype=float))

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        k = jnp.linalg.norm(frequency_grid, axis=-1)
        gaussian = self.amplitude * jnp.exp(
            -0.25 * self.b_factor * (k - self.radial_peak) ** 2
        )
        return gaussian


class FourierSinc(AbstractFourierOperator, strict=True):
    r"""The separable sinc function is the Fourier transform
    of the box function and is commonly used for anti-aliasing
    applications. In 2D, this is

    $$f_{2D}(\vec{q}) = \sinc(q_x w) \sinc(q_y w),$$

    and in 3D this is

    $$f_{3D}(\vec{q}) = \sinc(q_x w) \sinc(q_y w) \sinc(q_z w)},$$

    where $\sinc(x) = \frac{\sin(\pi x)}{\pi x}$,
    $\vec{q} = (q_x, q_y)$ or $\vec{q} = (q_x, q_y, q_z)$ are spatial
    frequency coordinates for 2D and 3D respectively,
    and $w$ is width of the real-space box function.
    """

    box_width: Float[Array, ""]

    def __init__(self, box_width: FloatLike = 1.0):
        """**Arguments:**

        - `box_width`:
            If the inverse fourier transform of this class
            is the rectangular function, its interval is
            `- box_width / 2` to `+ box_width / 2`.
        """
        self.box_width = jnp.asarray(box_width, dtype=float)

    @override
    def __call__(
        self,
        frequency_grid: (
            Float[Array, "y_dim x_dim 2"] | Float[Array, "z_dim y_dim x_dim 3"]
        ),
    ) -> Float[Array, "y_dim x_dim"] | Float[Array, "z_dim y_dim x_dim"]:
        ndim = frequency_grid.ndim - 1
        return functools.reduce(
            operator.mul,
            [jnp.sinc(frequency_grid[..., i] * self.box_width) for i in range(ndim)],
        )
