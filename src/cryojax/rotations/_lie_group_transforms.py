"""
Implementation of transformations in local tangent spaces of lie groups.
This is a re-implementation of `jaxlie.manifold`.

This was necessary because `cryojax` has its own pattern for `jax`
transformations, using `equinox`. The `jaxlie.manifold.grad` and
`jaxlie.manifold.value_and_grad` are not compatible with the `cryojax`
wrappers for gradient transformations.
"""

from typing import Generic, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jaxtyping import Array, Float, PyTree

from ..jax_util import AbstractPyTreeTransform
from ..rotations import SE3, SO3, AbstractMatrixLieGroup
from ..simulator import QuaternionPose


T = TypeVar("T")


def _apply_update_with_lie_transform(u, p):
    if u is None:
        return p
    elif isinstance(u, AbstractLieGroupTransform):
        lie_group = type(u.group_element)
        local_tangent = u.local_tangent
        return eqx.tree_at(
            lambda p: p.group_element,
            p,
            p.group_element @ lie_group.exp(local_tangent),
        )
    else:
        return p + u


def _is_none_or_lie_transform(x):
    return x is None or isinstance(x, AbstractLieGroupTransform)


def apply_updates_with_lie_transform(model: PyTree, updates: PyTree) -> PyTree:
    """Modifed `eqx.apply_updates` to apply updates to a model
    with `AbstractLieGroupTransform`s.

    This assumes that `updates` are a prefix of `model`.
    """
    return jtu.tree_map(
        _apply_update_with_lie_transform,
        updates,
        model,
        is_leaf=_is_none_or_lie_transform,
    )


class AbstractLieGroupTransform(AbstractPyTreeTransform[T], Generic[T], strict=True):
    """An abstract base class for lie group transforms."""

    local_tangent: AbstractVar[Array]
    group_element: AbstractVar[AbstractMatrixLieGroup]


class SO3Transform(AbstractLieGroupTransform[Float[Array, "4"]], strict=True):
    """This class transforms a quaternion to the local
    tangent space of the corresponding SO3 element.

    This class is based on the implementation in `jaxlie.manifold`.

    **Attributes:**

    - `local_tangent`: The local tangent vector.
    - `group_element`: The element of SO3.
    """

    local_tangent: Float[Array, "3"]
    group_element: SO3

    def __init__(self, wxyz: Float[Array, "4"]):
        """**Arguments:**

        - `wxyz`: A quaternion that parameterizes the SO3
                  group element.
        """
        self.local_tangent = jnp.zeros(3, dtype=float)
        self.group_element = SO3(wxyz).normalize()

    @property
    @override
    def value(self) -> Float[Array, "4"]:
        """An implementation of the `jaxlie.manifold.rplus`."""
        return (
            jax.lax.stop_gradient(self.group_element) @ SO3.exp(self.local_tangent)
        ).wxyz


class SE3Transform(AbstractLieGroupTransform[QuaternionPose], strict=True):
    """This class transforms a `QuaternionPose` to the local
    tangent space of the corresponding SE3 element.

    This class is based on the implementation in `jaxlie.manifold`.

    **Attributes:**

    - `local_tangent`: The local tangent vector.
    - `group_element`: The element of SE3.
    """

    local_tangent: Float[Array, "6"]
    group_element: SE3

    def __init__(self, quaternion_pose: QuaternionPose):
        """**Arguments:**

        - `quaternion_pose`: A quaternion pose representation that parameterizes
                             the SE3 group element.
        """
        self.local_tangent = jnp.zeros(6, dtype=float)
        self.group_element = SE3(
            rotation=quaternion_pose.rotation,
            xyz=quaternion_pose.offset_in_angstroms,
        )

    @property
    @override
    def value(self) -> QuaternionPose:
        """An implementation of the `jaxlie.manifold.rplus`."""
        group_element = jax.lax.stop_gradient(self.group_element) @ SE3.exp(
            self.local_tangent
        )
        return QuaternionPose.from_rotation_and_translation(
            group_element.rotation, group_element.xyz
        )
