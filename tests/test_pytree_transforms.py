import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    from cryojax.jax_util import (
        CustomTransform,
        StopGradientTransform,
        resolve_transforms,
    )


class Exp(eqx.Module):
    a: Array = eqx.field(converter=jnp.asarray)

    def __call__(self, x):
        return jnp.exp(-self.a * x)


def test_resolve_transform():
    pytree = Exp(a=1.0)
    pytree_with_transform = eqx.tree_at(
        lambda fn: fn.a,
        pytree,
        replace_fn=lambda a: CustomTransform(jnp.exp, jnp.log(a)),
    )
    assert eqx.tree_equal(pytree, resolve_transforms(pytree_with_transform))


def test_nested_resolve_transform():
    pytree = Exp(a=1.0)
    pytree_with_transform = eqx.tree_at(
        lambda fn: fn.a,
        pytree,
        replace_fn=lambda a: CustomTransform(lambda b: 2 * b, a / 2),
    )
    pytree_with_nested_transform = eqx.tree_at(
        lambda fn: fn.a.args[0],
        pytree_with_transform,
        replace_fn=lambda a_scaled: CustomTransform(jnp.exp, jnp.log(a_scaled)),
    )
    assert eqx.tree_equal(
        pytree,
        resolve_transforms(pytree_with_transform),
        resolve_transforms(pytree_with_nested_transform),
    )


def test_stop_gradient():
    @jax.value_and_grad
    def objective_fn(pytree):
        exp, x = resolve_transforms(pytree)
        return exp(x)

    x = jnp.asarray(np.random.random())
    exp = Exp(a=1.0)
    exp_with_stop_gradient = eqx.tree_at(
        lambda fn: fn.a, exp, replace_fn=StopGradientTransform
    )
    _, grads = objective_fn((exp_with_stop_gradient, x))
    grads = resolve_transforms(grads)
    assert grads[0].a == 0.0
