"""
Abstraction of rotations represented by matrix lie groups.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar
from typing_extensions import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractClassVar
from jaxtyping import Array, Float, PRNGKeyArray

from ..jax_util import FloatLike, NDArrayLike
from ._rotation import AbstractRotation


class AbstractMatrixLieGroup(AbstractRotation, strict=True):
    """Base class for a rotation that is represented by
    a matrix lie group.

    The class is almost exactly derived from the `jaxlie.MatrixLieGroup`
    object.

    `jaxlie` was written for [Yi, Brent, et al. 2021](https://ieeexplore.ieee.org/abstract/document/9636300).
    """

    parameter_dimension: AbstractClassVar[int]
    tangent_dimension: AbstractClassVar[int]
    matrix_dimension: AbstractClassVar[int]

    def __check_init__(self):
        if self.parameter_dimension != self.parameters.size:
            cls = self.__class__.__name__
            raise AttributeError(
                "Found incorrect array size in "
                f"in class {cls}. Expected "
                f"an array with size {self.parameter_dimension} as input, "
                f"but got an array of size {self.parameters.size}."
            )

    @property
    @abstractmethod
    def parameters(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def exp(cls, tangent: Array) -> Self:
        """Computes the exponential map of an element of the
        lie algebra.
        """
        raise NotImplementedError

    @abstractmethod
    def log(self) -> Array:
        """Computes the logarithmic map of the lie group element."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: Array) -> Self:
        """Computes the group element from a rotation matrix."""
        raise NotImplementedError

    @abstractmethod
    def as_matrix(self) -> Array:
        """Represent the group element as a rotation matrix."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self) -> Self:
        """Projects onto a group element."""
        raise NotImplementedError


class SO3(AbstractMatrixLieGroup, strict=True):
    """A rotation in 3D space, represented by the
    SO3 matrix lie group.

    The class is almost exactly derived from `jaxlie.SO3`.

    `jaxlie` was written for [Yi, Brent, et al. 2021](https://ieeexplore.ieee.org/abstract/document/9636300).
    """

    space_dimension: ClassVar[int] = 3
    parameter_dimension: ClassVar[int] = 4
    tangent_dimension: ClassVar[int] = 3
    matrix_dimension: ClassVar[int] = 3

    wxyz: Float[Array, "4"]

    def __init__(self, wxyz: Float[NDArrayLike, "4"] | Sequence[float]):
        """**Arguments:**

        - `wxyz`:
            A quaternion represented as $(q_w, q_x, q_y, q_z)$.
        """
        self.wxyz = jnp.asarray(wxyz, dtype=float)

    @property
    @override
    def parameters(self) -> Float[Array, "4"]:
        return self.wxyz

    @override
    def apply(self, target: Float[Array, "3"]) -> Float[Array, "3"]:
        # Compute using quaternion multiplys.
        padded_target = jnp.concatenate([jnp.zeros(1), target])
        return (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[1:]

    @override
    def compose(self, other: Self) -> Self:
        w0, x0, y0, z0 = self.wxyz
        w1, x1, y1, z1 = other.wxyz
        cls = type(self)
        return cls(
            wxyz=jnp.array(
                [
                    -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                    x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                    -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                    x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
                ]
            )
        )

    @override
    def inverse(self) -> Self:
        # Negate complex terms.
        return eqx.tree_at(lambda R: R.wxyz, self, self.wxyz * jnp.array([1, -1, -1, -1]))

    @classmethod
    def from_x_radians(cls, angle: FloatLike) -> Self:
        """Generates a x-axis rotation."""
        return cls.exp(jnp.asarray([angle, 0.0, 0.0]))

    @classmethod
    def from_y_radians(cls, angle: FloatLike) -> Self:
        """Generates a x-axis rotation."""
        return cls.exp(jnp.asarray([0.0, angle, 0.0]))

    @classmethod
    def from_z_radians(cls, angle: FloatLike) -> Self:
        """Generates a x-axis rotation."""
        return cls.exp(jnp.asarray([0.0, 0.0, angle]))

    @override
    @classmethod
    def identity(cls) -> Self:
        return cls(jnp.asarray([1.0, 0.0, 0.0, 0.0]))

    @override
    @classmethod
    def from_matrix(cls, matrix: Float[Array, "3 3"]) -> Self:
        # Modified from:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

        def case0(m):
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = jnp.array(
                [
                    m[2, 1] - m[1, 2],
                    t,
                    m[1, 0] + m[0, 1],
                    m[0, 2] + m[2, 0],
                ]
            )
            return t, q

        def case1(m):
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = jnp.array(
                [
                    m[0, 2] - m[2, 0],
                    m[1, 0] + m[0, 1],
                    t,
                    m[2, 1] + m[1, 2],
                ]
            )
            return t, q

        def case2(m):
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = jnp.array(
                [
                    m[1, 0] - m[0, 1],
                    m[0, 2] + m[2, 0],
                    m[2, 1] + m[1, 2],
                    t,
                ]
            )
            return t, q

        def case3(m):
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = jnp.array(
                [
                    t,
                    m[2, 1] - m[1, 2],
                    m[0, 2] - m[2, 0],
                    m[1, 0] - m[0, 1],
                ]
            )
            return t, q

        # Compute four cases, then pick the most precise one.
        # Probably worth revisiting this!
        case0_t, case0_q = case0(matrix)
        case1_t, case1_q = case1(matrix)
        case2_t, case2_q = case2(matrix)
        case3_t, case3_q = case3(matrix)

        cond0 = matrix[2, 2] < 0
        cond1 = matrix[0, 0] > matrix[1, 1]
        cond2 = matrix[0, 0] < -matrix[1, 1]

        t = jnp.where(
            cond0,
            jnp.where(cond1, case0_t, case1_t),
            jnp.where(cond2, case2_t, case3_t),
        )
        q = jnp.where(
            cond0,
            jnp.where(cond1, case0_q, case1_q),
            jnp.where(cond2, case2_q, case3_q),
        )

        return cls(wxyz=q * 0.5 / jnp.sqrt(t))

    @override
    def as_matrix(self) -> Float[Array, "3 3"]:
        norm = self.wxyz @ self.wxyz
        q = self.wxyz * jnp.sqrt(2.0 / norm)
        q = jnp.outer(q, q)
        return jnp.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )

    @classmethod
    def exp(cls, tangent: Float[Array, "3"]) -> Self:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L583
        theta_squared = tangent @ tangent
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < _get_epsilon(tangent.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        safe_theta = jnp.sqrt(
            jnp.where(
                use_taylor,
                1.0,  # Any constant value should do here.
                theta_squared,
            )
        )
        safe_half_theta = 0.5 * safe_theta

        real_factor = jnp.where(
            use_taylor,
            1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0,
            jnp.cos(safe_half_theta),
        )

        imaginary_factor = jnp.where(
            use_taylor,
            0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0,
            jnp.sin(safe_half_theta) / safe_theta,
        )

        return cls(
            wxyz=jnp.concatenate(
                [
                    real_factor[None],
                    imaginary_factor * tangent,
                ]
            )
        )

    @override
    def log(self) -> Float[Array, "3"]:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L247

        w = self.wxyz[..., 0]
        norm_sq = self.wxyz[..., 1:] @ self.wxyz[..., 1:]
        use_taylor = norm_sq < _get_epsilon(norm_sq.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        norm_safe = jnp.sqrt(
            jnp.where(
                use_taylor,
                1.0,  # Any non-zero value should do here.
                norm_sq,
            )
        )
        w_safe = jnp.where(use_taylor, w, 1.0)
        atan_n_over_w = jnp.arctan2(
            jnp.where(w < 0, -norm_safe, norm_safe),
            jnp.abs(w),
        )
        atan_factor = jnp.where(
            use_taylor,
            2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3,
            jnp.where(
                jnp.abs(w) < _get_epsilon(w.dtype),
                jnp.where(w > 0, 1.0, -1.0) * jnp.pi / norm_safe,
                2.0 * atan_n_over_w / norm_safe,
            ),
        )

        return atan_factor * self.wxyz[1:]

    @override
    def normalize(self) -> Self:
        return eqx.tree_at(lambda R: R.wxyz, self, self.wxyz / jnp.linalg.norm(self.wxyz))

    @classmethod
    def sample_uniform(cls, key: PRNGKeyArray) -> Self:
        # Uniformly sample over S^3.
        # > Reference: http://planning.cs.uiuc.edu/node198.html
        u1, u2, u3 = jax.random.uniform(
            key=key,
            shape=(3,),
            minval=jnp.zeros(3),
            maxval=jnp.array([1.0, 2.0 * jnp.pi, 2.0 * jnp.pi]),
        )
        a = jnp.sqrt(1.0 - u1)
        b = jnp.sqrt(u1)

        return cls(
            wxyz=jnp.array(
                [
                    a * jnp.sin(u2),
                    a * jnp.cos(u2),
                    b * jnp.sin(u3),
                    b * jnp.cos(u3),
                ]
            )
        )


class SO2(AbstractMatrixLieGroup, strict=True):
    """A rotation in 2D space, represented by the
    SO2 matrix lie group.

    The class is almost exactly derived from `jaxlie.SO2`.

    `jaxlie` was written for [Yi, Brent, et al. 2021](https://ieeexplore.ieee.org/abstract/document/9636300).
    """

    space_dimension: ClassVar[int] = 2
    parameter_dimension: ClassVar[int] = 2
    tangent_dimension: ClassVar[int] = 1
    matrix_dimension: ClassVar[int] = 2

    unit_complex: Float[Array, "2"]

    def __init__(
        self, unit_complex: Float[NDArrayLike, "2"] | Sequence[float | NDArrayLike]
    ):
        r"""**Arguments:**

        - `unit_complex`:
            A complex number represented as a 2-vector
            $(cos \theta, sin \theta)$.
        """
        self.unit_complex = jnp.asarray(unit_complex, dtype=float)

    @property
    @override
    def parameters(self) -> Float[Array, "4"]:
        return self.unit_complex

    @classmethod
    def from_radians(cls, theta: FloatLike) -> Self:
        """Construct a rotation object from a scalar angle."""
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return cls(unit_complex=jnp.stack([cos, sin], axis=-1))

    def as_radians(self) -> Float[Array, ""]:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()
        return radians

    @override
    @classmethod
    def identity(cls) -> Self:
        return cls(jnp.asarray([1.0, 0.0]))

    @override
    @classmethod
    def from_matrix(cls, matrix: Float[NDArrayLike, "2 2"]) -> Self:
        return cls(unit_complex=jnp.asarray(matrix[:, 0]))

    @override
    def as_matrix(self) -> Float[Array, "2 2"]:
        cos_sin = self.unit_complex
        out = jnp.stack(
            [
                # [cos, -sin],
                cos_sin * jnp.array([1, -1]),
                # [sin, cos],
                cos_sin[::-1],
            ],
            axis=-1,
        )
        return out

    @override
    def apply(self, target: Float[Array, "2"]) -> Float[Array, "2"]:
        return self.as_matrix() @ target

    @override
    def compose(self, other: Self) -> Self:
        cls = type(self)
        return cls(unit_complex=self.as_matrix() @ other.unit_complex)

    @classmethod
    @override
    def exp(cls, tangent: FloatLike) -> Self:
        cos = jnp.cos(tangent)
        sin = jnp.sin(tangent)
        return cls(unit_complex=jnp.concatenate([cos, sin], axis=-1))

    @override
    def log(self) -> Float[Array, ""]:
        return jnp.arctan2(self.unit_complex[1], self.unit_complex[0])

    @override
    def inverse(self) -> Self:
        cls = type(self)
        return cls(unit_complex=self.unit_complex * jnp.array([1, -1]))

    @override
    def normalize(self) -> Self:
        cls = type(self)
        return cls(
            unit_complex=self.unit_complex / jnp.linalg.norm(self.unit_complex, axis=-1)
        )

    @override
    @classmethod
    def sample_uniform(cls, key: PRNGKeyArray) -> Self:
        return cls.from_radians(
            jax.random.uniform(key=key, shape=(), minval=0.0, maxval=2.0 * jnp.pi)
        )


def _get_epsilon(dtype: jnp.dtype) -> float:
    """Helper for grabbing type-specific precision constants."""
    return {
        jnp.dtype("float32"): 1e-5,
        jnp.dtype("float64"): 1e-10,
    }[dtype]
