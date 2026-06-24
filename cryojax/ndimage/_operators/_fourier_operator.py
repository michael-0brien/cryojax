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
from typing import Any, ClassVar
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ..._internal import error_if_negative, error_if_not_positive
from ...jax_util import FloatLike, NDArrayLike


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

    spatial_dims: eqx.AbstractClassVar[list[int]]

    @abstractmethod
    def __call__(
        self,
        frequencies: Float[Array, "..."],
    ) -> Inexact[Array, "..."]:
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

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    @override
    def __call__(
        self,
        frequencies: Float[Array, "..."],
    ) -> Inexact[Array, "..."]:
        return self.operator1(frequencies) + self.operator2(frequencies)

    def __repr__(self):
        return f"{repr(self.operator1)} + {repr(self.operator2)}"


class _DiffFourierOperator(AbstractFourierOperator, strict=True):
    """A helper to represent the difference of two operators."""

    operator1: AbstractFourierOperator
    operator2: AbstractFourierOperator

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Inexact[Array, "..."]:
        return self.operator1(frequencies) - self.operator2(frequencies)

    def __repr__(self):
        return f"{repr(self.operator1)} - {repr(self.operator2)}"


class _ProductFourierOperator(AbstractFourierOperator, strict=True):
    """A helper to represent the product of two operators."""

    operator1: AbstractFourierOperator
    operator2: AbstractFourierOperator

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Inexact[Array, "..."]:
        return self.operator1(frequencies) * self.operator2(frequencies)

    def __repr__(self):
        return f"{repr(self.operator1)} * {repr(self.operator2)}"


class CustomFourierOperator(AbstractFourierOperator, strict=True):
    """An operator that calls a custom function."""

    fn: Callable[..., Inexact[Array, "..."]]
    args: Any
    kwargs: Any

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(
        self, fn: Callable[..., Inexact[Array, "..."]], *args: Any, **kwargs: Any
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Inexact[Array, "..."]:
        return self.fn(frequencies, *self.args, **self.kwargs)


CustomFourierOperator.__init__.__doc__ = """**Arguments:**

- `fn`:
    The `Callable` wrapped into a `AbstractFourierOperator`.
    Has signature `out = fn(frequencies, *args, **kwargs)`
- `args`:
    Passed to `fn`.
- `kwargs`:
    Passed to `fn`.
"""


class FourierDC(AbstractFourierOperator, strict=True):
    """This operator returns a constant in the DC component."""

    value: Float[Array, ""]

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(self, value: FloatLike = 0.0):
        """**Arguments:**

        - `value`: The value of the zero mode.
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Float[Array, "..."]:
        frequencies, ndim, flag = _standardize_frequencies(frequencies)
        if ndim == 1 and not flag:
            return jnp.zeros(frequencies.shape).at[0].set(self.value)
        elif ndim == 2:
            return jnp.zeros(frequencies.shape[0:-1]).at[0, 0].set(self.value)
        elif ndim == 3:
            return jnp.zeros(frequencies.shape[0:-1]).at[0, 0, 0].set(self.value)
        else:
            raise ValueError(
                "Could not parse the spatial dimension of `frequencies` passed to "
                f"`FourierDC`. Got `frequencies` with shape {frequencies.shape}."
            )


class FourierConstant(AbstractFourierOperator, strict=True):
    """An operator that is a constant."""

    value: Float[Array, "..."]

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(self, value: FloatLike):
        """**Arguments:**

        - `value`: The value of the constant
        """
        self.value = jnp.asarray(value)

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Float[Array, "..."]:
        del frequencies
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

    spatial_dims: ClassVar[list[int]] = [2]

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
        self.length_scale = jnp.asarray(length_scale, dtype=float)

    @override
    def __call__(self, frequencies: Float[Array, "... 2"]) -> Float[Array, "..."]:
        frequencies, ndim, _ = _standardize_frequencies(frequencies)
        if ndim != 2:
            raise ValueError(
                "`cryojax.ndimage.FourierExp2D` only supports "
                "`spatial_dim = 2`, but passed a frequency "
                "array with `spatial_dim = 1`."
            )
        k_sqr = jnp.sum(frequencies**2, axis=-1)
        scaling = (
            1.0
            / (k_sqr + jnp.divide(1, (error_if_not_positive(self.length_scale)) ** 2))
            ** 1.5
        )
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

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

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
        self.b_factor = jnp.asarray(b_factor, dtype=float)

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Float[Array, "..."]:
        frequencies, _, flag = _standardize_frequencies(frequencies)
        q_squared = jnp.sum(frequencies**2, axis=-1)
        gaussian = self.amplitude * jnp.exp(
            -0.25 * error_if_not_positive(self.b_factor) * q_squared
        )
        return _standardize_output(gaussian, flag=flag)


class PeakedFourierGaussian(AbstractFourierOperator, strict=True):
    r"""This operator represents a gaussian with a peak
    at a given frequency shell.
    """

    amplitude: Float[Array, ""]
    b_factor: Float[Array, ""]
    radial_peak: Float[Array, ""]

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

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
        self.b_factor = jnp.asarray(b_factor, dtype=float)
        self.radial_peak = jnp.asarray(radial_peak, dtype=float)

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Float[Array, "..."]:
        frequencies, _, flag = _standardize_frequencies(frequencies)
        k = jnp.linalg.norm(frequencies, axis=-1)
        gaussian = self.amplitude * jnp.exp(
            -0.25
            * error_if_not_positive(self.b_factor)
            * (k - error_if_negative(self.radial_peak)) ** 2
        )
        return _standardize_output(gaussian, flag=flag)


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

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(self, box_width: FloatLike = 1.0):
        """**Arguments:**

        - `box_width`:
            If the inverse fourier transform of this class
            is the rectangular function, its interval is
            `- box_width / 2` to `+ box_width / 2`.
        """
        self.box_width = jnp.asarray(box_width, dtype=float)

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Float[Array, "..."]:
        frequencies, ndim, flag = _standardize_frequencies(frequencies)
        return _standardize_output(
            functools.reduce(
                operator.mul,
                [jnp.sinc(frequencies[..., i] * self.box_width) for i in range(ndim)],
            ),
            flag=flag,
        )


class FourierPhaseShifts(AbstractFourierOperator):
    """Apply a phase shift the Fourier domain."""

    shift: Float[Array, " _"]

    spatial_dims: ClassVar[list[int]] = [1, 2, 3]

    def __init__(
        self,
        shift: FloatLike | Float[NDArrayLike, "2"] | Float[NDArrayLike, "3"],
    ):
        """**Arguments:**

        - `shift`:
            The shift to apply in the Fourier domain. The units of this should
            be the inverse of the units of the `frequencies` passed at runtime.
        """
        self.shift = jnp.asarray(jnp.atleast_1d(shift), dtype=float)

    @override
    def __call__(self, frequencies: Float[Array, "..."]) -> Complex[Array, "..."]:
        frequencies, ndim, flag = _standardize_frequencies(frequencies)
        if ndim != self.shift.size:
            raise ValueError(
                "The `frequencies` passed to `FourierPhaseShift` had "
                "dimensionality that does not seem to match `FourierPhaseShift.shift`. "
                f"Got that the dimensionality of the grid was `{ndim}`, but the "
                f"shift was an array of size {self.shift.size}"
            )
        return _standardize_output(
            jnp.exp(-1.0j * (2 * jnp.pi * jnp.matmul(frequencies, self.shift))), flag=flag
        )


def _standardize_frequencies(q: Array):
    flag = False
    if q.ndim == 0:
        flag = True
        q = q[None, None]
    elif q.ndim == 1:
        q = q[:, None]
    ndim = q.shape[-1]
    return q, ndim, flag


def _standardize_output(out: Array, *, flag: bool):
    return out[0] if flag else out
