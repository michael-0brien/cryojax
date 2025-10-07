"""
Utilities for creating equinox filter_specs.
"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree


def make_filter_spec(
    pytree: PyTree,
    where: Callable[[PyTree], Any | Sequence[Any]],
    *,
    inverse: bool = False,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree[bool]:
    """A lightweight wrapper around `equinox` for creating a "filter specification".

    A filter specification, or `filter_spec`, is a pytree whose
    leaves are either `True` or `False`. These are commonly used with
    `equinox` [filtering](https://docs.kidger.site/equinox/all-of-equinox/#2-filtering).

    In `cryojax`, it is a common pattern to need to finely specify which
    leaves we would like to take JAX transformations with respect to. This is done with a
    pointer to individual leaves, which is referred to as a `where` function. See
    [`here`](https://docs.kidger.site/equinox/examples/frozen_layer/#freezing-parameters)
    in the `equinox` documentation for an example.

    **Returns:**

    The filter specification. This is a pytree of the same structure as `pytree` with
    `True` where the `where` function points to, and `False` where it does not
    (or the opposite, if `inverse = True`).
    """
    if not inverse:
        false_pytree = jtu.tree_map(lambda _: False, pytree, is_leaf=is_leaf)
        return eqx.tree_at(
            where, false_pytree, replace_fn=lambda _: True, is_leaf=is_leaf
        )
    else:
        true_pytree = jtu.tree_map(lambda _: True, pytree)
        return eqx.tree_at(
            where, true_pytree, replace_fn=lambda _: False, is_leaf=is_leaf
        )
