"""
Representations of rigid body rotations and translations of 3D coordinate systems.
"""

from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Self
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar, Module
from jaxtyping import Array, Complex, Float

from .._internal import leaf_asarray, leaf_stack
from ..jax_util import FloatLike, NDArrayLike
from ..ndimage import (
    FourierPhaseShifts,
    enforce_rfftn_self_conjugates,
    make_1d_frequency_grid,
)
from ..rotations import SO3, convert_quaternion_to_euler_angles


class AbstractPose(Module, strict=True):
    """Base class for the image pose. Subclasses will choose a
    particular convention for parameterizing the rotation by
    overwriting the `AbstractPose.rotation` property.
    """

    offset_in_angstroms: AbstractVar[
        Float[NDArrayLike, "... 2"] | Float[NDArrayLike, "... 3"]
    ]

    def rotate_coordinates(
        self, target: (Float[Array, "... 3"]), /, inverse: bool = False
    ) -> Float[Array, "... 3"]:
        """Rotate a 3D coordinate system.

        **Arguments:**

        - `target`:
            The 3D coordinate system to rotate. This can be any array whose trailing
            dimension has length 3, such as a list of coordinates
            of shape `(N, 3)` or a grid of coordinates `(N1, N2, N3, 3)`.
        - `inverse`:
            If `True`, compute the inverse rotation (i.e. rotation by the matrix $R^T$,
            where $R$ is the rotation matrix).

        **Returns:**

        The rotated version of `target`.
        """
        rotation = self.rotation.inverse() if inverse else self.rotation
        # `rotation.apply` broadcasts over arbitrary leading dimensions, so it
        # handles both a list of coordinates `(N, 3)` and a grid
        # `(N1, N2, N3, 3)` directly, without explicit `jax.vmap`.
        return rotation.apply(target)

    def translate_image(
        self,
        fourier_image: Complex[Array, "{shape[0]} {shape[1]}//2+1"],
        translation_operator: Complex[Array, "{shape[0]} {shape[1]}//2+1"],
        shape: tuple[int, int],
    ) -> Complex[Array, "{shape[0]} {shape[1]}//2+1"]:
        """Apply translational phase shifts to a fourier-space image.

        **Arguments:**

        - `fourier_image`:
            The image in fourier-space, which is the output of a call
            to `cryojax.image.rfftn`.
        - `phase_shifts`:
            The phase shifts for translation, which are computed from
            `AbstractPose.compute_translation_operator`.
        - `shape`:
            The shape of `fourier_image` in real-space.

        **Return:**

        The translated `fourier_image`, taking care to avoid image
        artifacts when applying the phase shifts.
        """
        fourier_image = enforce_rfftn_self_conjugates(
            fourier_image, shape, includes_dc=False, mode="zero"
        )
        return fourier_image * translation_operator

    def compute_translation_operator(
        self,
        shape: tuple[int, int],
        pixel_size: Float[NDArrayLike, ""],
    ) -> Complex[Array, "y_dim x_dim//2+1"]:
        """Compute the phase shifts from the in-plane translation.

        **Arguments:**

        - `shape`:
            The real-space image shape $(N_y, N_x)$.
        - `pixel_size`:
            The pixel size in Angstroms.

        **Returns:**

        From the vector $(t_x, t_y)$ (given by `self.offset_in_angstroms`), returns the
        grid of in-plane phase shifts $\\exp{(- 2 \\pi i (t_x q_x + t_y q_y))}$.
        """
        offset_in_angstroms = jnp.asarray(self.offset_in_angstroms)
        tx, ty = offset_in_angstroms[0], offset_in_angstroms[1]
        q_x, q_y = (
            make_1d_frequency_grid(shape[1], pixel_size, outputs_rfftfreqs=True),
            make_1d_frequency_grid(shape[0], pixel_size, outputs_rfftfreqs=False),
        )
        phase_x, phase_y = (FourierPhaseShifts(tx)(q_x), FourierPhaseShifts(ty)(q_y))
        return phase_y[:, None] * phase_x[None, :]

    @cached_property
    def offset_x_in_angstroms(self) -> Float[Array, "..."]:
        """The in-plane translation in the x direction."""
        # a[..., i] indexing is for convenience outside of `jax.vmap`
        # regions. Be careful! Will silently cause failures
        # if batch dimensions are not the leading dimensions.
        return jnp.asarray(self.offset_in_angstroms)[..., 0]

    @cached_property
    def offset_y_in_angstroms(self) -> Float[Array, "..."]:
        """The in-plane translation in the y direction."""
        # a[..., i] indexing is for convenience outside of `jax.vmap`
        # regions. Be careful! Will silently cause failures
        # if batch dimensions are not the leading dimensions.
        return jnp.asarray(self.offset_in_angstroms)[..., 1]

    @cached_property
    def offset_z_in_angstroms(self) -> Float[Array, "..."] | None:
        """The out-of-plane translation in the z direction."""
        # a[..., i] is for convenience outside of `jax.vmap`
        # regions. Be careful! Will silently cause failures
        # if batch dimensions are not the leading dimensions.
        return (
            None
            if self.offset_in_angstroms.shape[-1] == 2
            else jnp.asarray(self.offset_in_angstroms)[..., 2]
        )

    @cached_property
    @abstractmethod
    def rotation(self) -> SO3:
        """Generate an `SO3` object from a particular angular
        parameterization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        """Construct an `AbstractPose` from an `SO3` object."""
        raise NotImplementedError

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        offset_in_angstroms: Float[NDArrayLike, "... 2"] | Float[NDArrayLike, "... 3"],
    ) -> Self:
        """Construct an `AbstractPose` from an `SO3` object and a
        translation vector.
        """
        if offset_in_angstroms.shape[-1] not in [2, 3]:
            raise ValueError(
                "Array `offset_in_angstroms` given to constructor "
                f"`{cls.__name__}.from_rotation_and_translation` supports "
                "trailing dimension `2` and `3`. Got shape "
                f"`{offset_in_angstroms.shape}`"
            )
        return eqx.tree_at(
            lambda p: p.offset_in_angstroms,
            cls.from_rotation(rotation),
            offset_in_angstroms,
        )

    @classmethod
    def from_translation(
        cls,
        offset_in_angstroms: Float[NDArrayLike, "2"] | Float[NDArrayLike, "3"],
    ) -> Self:
        """Construct an `AbstractPose` from a
        translation vector.
        """
        return eqx.tree_at(
            lambda p: p.offset_in_angstroms,
            cls.from_rotation(SO3(wxyz=jnp.asarray((1, 0, 0, 0), dtype=float))),
            leaf_asarray(offset_in_angstroms, dtype=float),
        )

    @abstractmethod
    def to_inverse_rotation(self) -> Self:
        """Convert an `AbstractPose` to the inverse of its rotation
        representation.
        """
        raise NotImplementedError


class EulerAnglePose(AbstractPose, strict=True):
    r"""An `AbstractPose` represented by Euler angles.
    Angles are given in degrees, and the sequence of rotations is a
    zyz *extrinsic* rotation, with `phi_angle` as the first euler angle,
    `theta_angle` as the second, and `psi_angle` is the third.

    !!! info "Converting to RELION and FREALIGN convention"

        RELION/FREALIGN convention is that the euler angles represent
        a zyz *intrinsic* rotation that "undoes" the rotation in the image. cryoJAX
        defines its convention to be a zyz *extrinsic* rotation that generates the
        pose in the image. In order to convert to the RELION/FREALIGN convention,
        simply **negate each euler angle**.
    """

    offset_in_angstroms: Float[NDArrayLike, "... 2"] | Float[NDArrayLike, "... 3"]

    phi_angle: Float[NDArrayLike, "..."]
    theta_angle: Float[NDArrayLike, "..."]
    psi_angle: Float[NDArrayLike, "..."]

    def __init__(
        self,
        offset_x_in_angstroms: FloatLike = 0.0,
        offset_y_in_angstroms: FloatLike = 0.0,
        phi_angle: FloatLike = 0.0,
        theta_angle: FloatLike = 0.0,
        psi_angle: FloatLike = 0.0,
        *,
        offset_z_in_angstroms: FloatLike | None = None,
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `phi_angle`: Angle to rotate about first rotation axis, which is the z axis.
        - `theta_angle`: Angle to rotate about second rotation axis, which is the y axis.
        - `psi_angle`: Angle to rotate about third rotation axis, which is the z axis.
        - `offset_z_in_angstroms`: Out-of-plane translation in z direction.
        """
        if offset_z_in_angstroms is None:
            self.offset_in_angstroms = leaf_stack(
                (offset_x_in_angstroms, offset_y_in_angstroms), dtype=float
            )
        else:
            self.offset_in_angstroms = leaf_stack(
                (offset_x_in_angstroms, offset_y_in_angstroms, offset_z_in_angstroms),
                dtype=float,
            )
        self.phi_angle = leaf_asarray(phi_angle, dtype=float)
        self.theta_angle = leaf_asarray(theta_angle, dtype=float)
        self.psi_angle = leaf_asarray(psi_angle, dtype=float)

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate a `SO3` object from a set of Euler angles."""
        phi, theta, psi = (
            jnp.asarray(self.phi_angle),
            jnp.asarray(self.theta_angle),
            jnp.asarray(self.psi_angle),
        )
        # Convert to radians.
        phi = jnp.deg2rad(phi)
        theta = jnp.deg2rad(theta)
        psi = jnp.deg2rad(psi)
        # Get sequence of rotations.
        R1, R2, R3 = (
            SO3.from_z_radians(phi),
            SO3.from_y_radians(theta),
            SO3.from_z_radians(psi),
        )
        return R3 @ R2 @ R1

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        phi_angle, theta_angle, psi_angle = convert_quaternion_to_euler_angles(
            rotation.wxyz, convention="zyz", extrinsic=True
        )
        return cls(phi_angle=phi_angle, theta_angle=theta_angle, psi_angle=psi_angle)

    @override
    def to_inverse_rotation(self) -> Self:
        return eqx.tree_at(
            lambda x: (x.phi_angle, x.theta_angle, x.psi_angle),
            self,
            (
                _negate_angle(self.psi_angle),
                _negate_angle(self.theta_angle),
                _negate_angle(self.phi_angle),
            ),
        )


class QuaternionPose(AbstractPose, strict=True):
    """An `AbstractPose` represented by unit quaternions."""

    offset_in_angstroms: Float[NDArrayLike, "... 2"] | Float[NDArrayLike, "... 3"]

    wxyz: Float[NDArrayLike, "... 4"]

    def __init__(
        self,
        offset_x_in_angstroms: FloatLike = 0.0,
        offset_y_in_angstroms: FloatLike = 0.0,
        wxyz: Sequence[float] | Float[NDArrayLike, "... 4"] = (1.0, 0.0, 0.0, 0.0),
        *,
        offset_z_in_angstroms: FloatLike | None = None,
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `wxyz`:
            The quaternion, represented as a vector $\\mathbf{q} = (q_w, q_x, q_y, q_z)$.
        - `offset_z_in_angstroms`: Out-of-plane translation in z direction.
        """
        if offset_z_in_angstroms is None:
            self.offset_in_angstroms = leaf_stack(
                (offset_x_in_angstroms, offset_y_in_angstroms), dtype=float
            )
        else:
            self.offset_in_angstroms = leaf_stack(
                (offset_x_in_angstroms, offset_y_in_angstroms, offset_z_in_angstroms),
                dtype=float,
            )
        wxyz = leaf_asarray(wxyz, dtype=float)
        if wxyz.shape[-1] != 4:
            raise ValueError(
                "Expected `wxyz` to have a trailing dimension of "
                f"length 4, but found `wxyz.shape = {wxyz.shape}`."
            )
        self.wxyz = wxyz

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from the unit quaternion
        $\\mathbf{q} / |\\mathbf{q}|$.
        """
        # Generate SO3 object from unit quaternion
        R = SO3(wxyz=jnp.asarray(self.wxyz)).normalize()
        return R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        return cls(wxyz=rotation.wxyz)

    @override
    def to_inverse_rotation(self) -> Self:
        inverse_rotation = self.rotation.inverse()
        return eqx.tree_at(lambda _pose: _pose.wxyz, self, inverse_rotation.wxyz)


class AxisAnglePose(AbstractPose, strict=True):
    """An `AbstractPose` parameterized in the axis-angle representation.

    The axis-angle representation parameterizes elements of the so3 algebra,
    which are skew-symmetric matrices, with the euler vector
    $\\boldsymbol{\\omega} = (\\omega_x, \\omega_y, \\omega_z)$.
    The magnitude of this vector is the angle, and the unit vector is the axis.

    In a `SO3` object, the euler vector is mapped to SO3 group elements using
    the matrix exponential.
    """

    offset_in_angstroms: Float[NDArrayLike, "... 2"] | Float[NDArrayLike, "... 3"]

    euler_vector: Float[NDArrayLike, "... 3"]

    def __init__(
        self,
        offset_x_in_angstroms: FloatLike = 0.0,
        offset_y_in_angstroms: FloatLike = 0.0,
        euler_vector: Sequence[float] | Float[NDArrayLike, "... 3"] = (0.0, 0.0, 0.0),
        *,
        offset_z_in_angstroms: FloatLike | None = None,
    ):
        """**Arguments:**

        - `offset_x_in_angstroms`: In-plane translation in x direction.
        - `offset_y_in_angstroms`: In-plane translation in y direction.
        - `euler_vector`:
            The axis-angle parameterization, represented with the euler
            vector $\\boldsymbol{\\omega}$.
        - `offset_z_in_angstroms`: Out-of-plane translation in z direction.
        """
        if offset_z_in_angstroms is None:
            self.offset_in_angstroms = leaf_stack(
                (offset_x_in_angstroms, offset_y_in_angstroms), dtype=float
            )
        else:
            self.offset_in_angstroms = leaf_stack(
                (offset_x_in_angstroms, offset_y_in_angstroms, offset_z_in_angstroms),
                dtype=float,
            )
        euler_vector = leaf_asarray(euler_vector, dtype=float)
        if euler_vector.shape[-1] != 3:
            raise ValueError(
                "Expected `euler_vector` to have a trailing dimension of "
                f"length 3, but found `euler_vector.shape = {euler_vector.shape}`."
            )
        self.euler_vector = euler_vector

    @cached_property
    @override
    def rotation(self) -> SO3:
        """Generate rotation from an euler vector using the exponential map."""
        # Convert degrees to radians
        euler_vector = jnp.deg2rad(jnp.asarray(self.euler_vector))
        # Project the tangent vector onto the manifold with
        # the exponential map
        R = SO3.exp(euler_vector)
        return R

    @override
    @classmethod
    def from_rotation(cls, rotation: SO3) -> Self:
        # Compute the euler vector from the logarithmic map
        euler_vector = jnp.rad2deg(rotation.log())
        return cls(euler_vector=euler_vector)

    @override
    def to_inverse_rotation(self) -> Self:
        return self.from_rotation_and_translation(
            self.rotation.inverse(), self.offset_in_angstroms
        )


def _negate_angle(angle):
    return ((-angle + 180) % 360) - 180
