from typing import Callable, TypeVar

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
    batch_dim = jax.tree.leaves(xs)[0].shape[0]
    n_batches = batch_dim // batch_size
    # Filter
    xs_transform = NonArrayStaticTransform(xs)
    init_transform = NonArrayStaticTransform(init)

    def f_scan(_carry_transform, _xs_transform):
        _carry, _ys = f(_carry_transform.value, _xs_transform.value)
        return NonArrayStaticTransform(_carry), NonArrayStaticTransform(_ys)

    # Scan over batches
    xs_transform = jax.tree.map(
        lambda x: x[: batch_dim - batch_dim % batch_size, ...].reshape(
            (n_batches, batch_size, *x.shape[1:])
        ),
        xs_transform,
    )
    carry_transform, ys_transform = jax.lax.scan(
        f_scan, init_transform, xs_transform, length=length, unroll=unroll
    )
    ys_transform = jax.tree.map(
        lambda y: y.reshape(n_batches * batch_size, *y.shape[2:]), ys_transform
    )
    if batch_dim % batch_size != 0:
        xs_remainder = jax.tree.map(
            lambda x: x[batch_dim - batch_dim % batch_size :, ...], xs_transform
        )
        carry_transform, ys_remainder = f_scan(carry_transform, xs_remainder)
        ys_transform = jax.tree.map(
            lambda x, y: jax.lax.concatenate([x, y], dimension=0),
            ys_transform,
            ys_remainder,
        )

    return carry_transform.value, ys_transform.value
