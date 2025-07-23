from typing import Callable, TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, PyTree, Shaped


X = TypeVar("X")
Y = TypeVar("Y")
Carry = TypeVar("Carry")


def batched_map(
    f: Callable[
        [PyTree[Shaped[Array, "_ ..."], "X"]], PyTree[Shaped[Array, "_ ..."], "Y"]
    ],
    xs: PyTree[Shaped[Array, "_ ..."], "X"],
    *,
    batch_size: int = 1,
) -> PyTree[Shaped[Array, "_ ..."], "Y"]:
    """Like `jax.lax.map(..., batch_size=...)`, except
    `f(x)` is already vmapped by the user. In particular,
    it must be vmapped over the first axis of the arrays of `x`.

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

    @eqx.filter_jit
    def f_scan(carry, x):
        return carry, f(x)

    _, ys = batched_scan(f_scan, None, xs, batch_size=batch_size)

    return ys


def batched_scan(
    f: Callable[
        [Carry, PyTree[Shaped[Array, "_ ..."], "X"]],
        tuple[Carry, PyTree[Shaped[Array, "_ ..."], "Y"]],
    ],
    init: Carry,
    xs: PyTree[Shaped[Array, "_ ..."], "X"],
    length: int | None = None,
    unroll: int | bool = 1,
    *,
    batch_size: int = 1,
) -> tuple[Carry, PyTree[Shaped[Array, "_ ..."], "Y"]]:
    """Like `jax.lax.map(..., batch_size=...)`, except adding
    a `batch_size` to `jax.lax.scan`. Additionally, unlike
    `jax.lax.map`, it is assumed that `f(carry, x)` is already
    vmapped over the first axis of the arrays of `x`.

    **Arguments:**

    - `f`:
        As `jax.lax.scan` with format `f(carry, x)`, except
        vmapped over the first axis of the arrays of `x`.
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
    # Scan over batches
    scan_xs = jax.tree.map(
        lambda x: x[: batch_dim - batch_dim % batch_size, ...].reshape(
            (n_batches, batch_size, *x.shape[1:])
        ),
        xs,
    )
    carry, scan_ys = jax.lax.scan(f, init, scan_xs, length=length, unroll=unroll)
    ys = jax.tree.map(lambda y: y.reshape(n_batches * batch_size, *y.shape[2:]), scan_ys)
    if batch_dim % batch_size != 0:
        remainder_xs = jax.tree.map(
            lambda x: x[batch_dim - batch_dim % batch_size :, ...], xs
        )
        carry, remainder_ys = f(carry, remainder_xs)
        ys = jax.tree.map(
            lambda x, y: jax.lax.concatenate([x, y], dimension=0),
            ys,
            remainder_ys,
        )

    return carry, ys
