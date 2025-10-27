from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, PyTree, Shaped

from ._pytree_transforms import NonArrayStaticTransform


X = TypeVar("X")
Y = TypeVar("Y")
Carry = TypeVar("Carry")


def filter_bmap(
    f: Callable[
        [PyTree[Shaped[Array, "_ ..."], "X"]], PyTree[Shaped[Array, "_ ..."], "Y"]
    ],
    xs: PyTree[Shaped[Array, "_ ..."], "X"],
    *,
    batch_size: int = 1,
) -> PyTree[Shaped[Array, "_ ..."], "Y"]:
    """Like `jax.lax.map(..., batch_size=...)`, but accepts `x`
    with the same rank as `xs`. `xs` is filtered in the usual
    `equinox.filter_*` way.

    **Arguments:**

    - `f`:
        As `jax.lax.map` with format `f(x)`, except
        vmapped over the first axis of the arrays of `x`.
    - `xs`:
        As `jax.lax.map`.
    - `batch_size`:
        Compute a loop of vmaps over `xs` in chunks of `batch_size`.

    **Returns:**

    As `jax.lax.map`.
    """

    def f_scan(carry, x):
        return carry, f(x)

    _, ys = filter_bscan(f_scan, None, xs, batch_size=batch_size)

    return ys


def filter_bscan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    unroll: int | bool = 1,
    *,
    batch_size: int = 1,
) -> tuple[Carry, Y]:
    """Like `jax.lax.map(..., batch_size=...)`, except adding
    a `batch_size` to `jax.lax.scan`. Additionally, unlike
    `jax.lax.map`, `f(carry, x)` accepts `x` with the same
    rank as `xs` (e.g. perhaps it is vmapped over `x`).
    `xs` and `carry` are filtered in the usual `equinox.filter_*` way.

    **Arguments:**

    - `f`:
        As `jax.lax.scan` with format `f(carry, x)`.
    - `init`:
        As `jax.lax.scan`.
    - `xs`:
        As `jax.lax.scan`.
    - `length`:
        As `jax.lax.scan`.
    - `unroll`:
        As `jax.lax.scan`.
    - `batch_size`:
        Compute a loop of vmaps over `xs` in chunks of `batch_size`.

    **Returns:**

    As `jax.lax.scan`.
    """
    tree_leaves = jax.tree.leaves(eqx.filter(xs, eqx.is_array))
    if len(tree_leaves) == 0:
        raise ValueError(
            "Called `cryojax.jax_util.filter_bscan` with `xs` "
            "containing no JAX/numpy arrays. Unlike regular `jax.lax.scan` "
            "`xs` is not optional."
        )
    if any(leaf.shape == () for leaf in tree_leaves):
        raise ValueError(
            "Called `cryojax.jax_util.filter_bscan` with `xs` "
            "containing JAX/numpy array scalars (i.e. `shape = ()`). "
            "All JAX/numpy arrays in `xs` must have "
            "a leading dimension that are equal to one another."
        )
    batch_dim = tree_leaves[0].shape[0]
    if not all(leaf.shape[0] == batch_dim for leaf in tree_leaves):
        raise ValueError(
            "Called `cryojax.jax_util.filter_bscan` with `xs` "
            "containing JAX/numpy arrays with different leading "
            "dimensions. All JAX/numpy arrays in `xs` must have "
            "the same leading dimension."
        )
    n_batches = batch_dim // batch_size
    # Filter
    xs_dynamic, xs_static = eqx.partition(xs, eqx.is_array)
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)

    def f_scan(_carry_dynamic, _xs_dynamic):
        _carry, _xs = (
            eqx.combine(_carry_dynamic, init_static),
            eqx.combine(_xs_dynamic, xs_static),
        )
        _carry, _ys = f(_carry, _xs)
        _carry_dynamic = eqx.filter(_carry, eqx.is_array)
        _ys_wrapped = NonArrayStaticTransform(_ys)
        return _carry_dynamic, _ys_wrapped

    # Scan over batches
    xs_reshaped = jax.tree.map(
        lambda x: x[: batch_dim - batch_dim % batch_size, ...].reshape(
            (n_batches, batch_size, *x.shape[1:])
        ),
        xs_dynamic,
    )
    carry_dynamic, ys_wrapped = jax.lax.scan(
        f_scan, init_dynamic, xs_reshaped, length=length, unroll=unroll
    )
    ys_wrapped = jax.tree.map(
        lambda y: y.reshape(n_batches * batch_size, *y.shape[2:]), ys_wrapped
    )
    if batch_dim % batch_size != 0:
        xs_remainder = jax.tree.map(
            lambda x: x[batch_dim - batch_dim % batch_size :, ...], xs_dynamic
        )
        carry_dynamic, ys_remainder = f_scan(carry_dynamic, xs_remainder)
        ys_wrapped = jax.tree.map(
            lambda x, y: jax.lax.concatenate([x, y], dimension=0),
            ys_wrapped,
            ys_remainder,
        )

    return eqx.combine(carry_dynamic, init_static), ys_wrapped.value
