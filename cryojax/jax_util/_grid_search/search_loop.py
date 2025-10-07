"""The main search loop for the grid search."""

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .._tqdm_decorator import fori_loop_tqdm_decorator
from .custom_types import PyTreeGrid, PyTreeGridPoint
from .pytree_manipulation import (
    tree_grid_shape,
    tree_grid_take,
    tree_grid_unravel_index,
)
from .search_method import AbstractGridSearchMethod


@eqx.filter_jit
def run_grid_search(
    fn: Callable[[PyTreeGridPoint, Any], Array],
    method: AbstractGridSearchMethod,
    tree_grid: PyTreeGrid,
    args: Any,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    progress_bar: bool = False,
    total_progress_bar_updates: int = 20,
) -> PyTree[Any]:
    """Run a grid search to minimize the function `fn`.

    !!! question "What is a `tree_grid`?"

        For the grid search, we represent the grid as an arbitrary
        pytree whose leaves are JAX arrays with a leading dimension.
        For a particular leaf, its leading dimension indexes a set
        grid points. The entire grid is then the cartesian product
        of the grid points of all of its leaves.

    !!! warning

        A `tree_grid` can only have leaves that are JAX arrays of
        grid points and `None`. It is difficult to precisely check this
        condition even with a run-time type checker, so breaking it may
        result in unhelpful errors.

    To learn more, see the `tree_grid` manipulation routines [`tree_grid_shape`][] and
    [`tree_grid_take`][].

    **Arguments:**

    - `fn`: The function we would like to minimize with grid search. This
            should be evaluated at arguments `fn(y, args)`, where `y` is a
            particular grid point of `tree_grid`. The value returned by `fn`
            must be compatible with the respective `method`.
    - `method`: An interface that specifies what we would like to do with
                each evaluation of `fn`.
    - `tree_grid`: The grid as a pytree. Importantly, its leaves can only be JAX
                   arrays with leading dimensions and `None`.
    - `args`: Arguments passed to `fn`, as `fn(y, args)`.
    - `is_leaf`: As [`jax.tree_util.tree_flatten`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_flatten.html).
                 This specifies what is to be treated as a leaf in `tree_grid`.
    - `progress_bar`: Add a [`tqdm`](https://github.com/tqdm/tqdm) progress bar to the
                      search loop.
    - `total_progress_bar_updates`:
        An interval for the number of iterations at which toupdate the
        tqdm progress bar. By default, this is `20`. Ignored if `progress_bar = False`.

    **Returns:**

    Any pytree, as specified by the method `AbstractGridSearchMethod.postprocess`.
    """
    # Evaluate the shape and dtype of the output of `fn` using
    # eqx.filter_closure_convert.
    test_tree_grid_point = tree_grid_take(
        tree_grid,
        tree_grid_unravel_index(0, tree_grid, is_leaf=is_leaf),
    )
    fn = eqx.filter_closure_convert(fn, test_tree_grid_point, args)
    f_struct = jtu.tree_map(
        lambda x: x.value,
        jtu.tree_map(eqxi.Static, fn.out_struct),  # type: ignore
        is_leaf=lambda x: isinstance(x, eqxi.Static),
    )
    # Get the initial state of the search method
    init_state = method.init(tree_grid, f_struct, is_leaf=is_leaf)
    dynamic_init_state, static_state = eqx.partition(init_state, eqx.is_array)
    # Finally, build the loop
    init_carry = (dynamic_init_state, tree_grid)

    def brute_force_body_fun(iteration_index, carry):
        dynamic_state, tree_grid = carry
        state = eqx.combine(static_state, dynamic_state)
        tree_grid_point = tree_grid_take(
            tree_grid,
            tree_grid_unravel_index(iteration_index, tree_grid, is_leaf=is_leaf),
        )
        new_state = method.update(fn, tree_grid_point, args, state, iteration_index)
        new_dynamic_state, new_static_state = eqx.partition(new_state, eqx.is_array)
        assert eqx.tree_equal(static_state, new_static_state) is True
        return new_dynamic_state, tree_grid

    def batched_body_fun(iteration_index, carry):
        dynamic_state, tree_grid = carry
        state = eqx.combine(static_state, dynamic_state)
        raveled_grid_index_batch = jnp.linspace(
            iteration_index * method.batch_size,
            (iteration_index + 1) * method.batch_size - 1,
            method.batch_size,  # type: ignore
            dtype=jnp.int32,
        )
        tree_grid_points = tree_grid_take(
            tree_grid,
            tree_grid_unravel_index(raveled_grid_index_batch, tree_grid, is_leaf=is_leaf),
        )
        new_state = method.batch_update(
            fn, tree_grid_points, args, state, raveled_grid_index_batch
        )
        new_dynamic_state, new_static_state = eqx.partition(new_state, eqx.is_array)
        assert eqx.tree_equal(static_state, new_static_state) is True
        return new_dynamic_state, tree_grid

    # Get the number of iterations of the loop (the size of the grid)
    grid_size = math.prod(tree_grid_shape(tree_grid, is_leaf=is_leaf))
    if method.batch_size is None:
        n_iterations = grid_size
        body_fun = brute_force_body_fun
    else:
        n_iterations = grid_size // method.batch_size
        body_fun = batched_body_fun
    # Run and unpack results
    if progress_bar:
        print_every = n_iterations // total_progress_bar_updates
        if print_every == 0:
            raise ValueError(
                "The number of progress bar updates is greater than the "
                "number of iterations. Try decreasing `total_progress_bar_updates` "
                "or setting it to `None`."
            )
        body_fun = fori_loop_tqdm_decorator(n_iterations, print_every)(body_fun)
    final_carry = jax.lax.fori_loop(0, n_iterations, body_fun, init_carry)
    if method.batch_size is not None and grid_size % method.batch_size != 0:
        final_carry = jax.lax.fori_loop(
            n_iterations,
            n_iterations + (grid_size % method.batch_size),
            batched_body_fun,
            final_carry,
        )
    dynamic_final_state, _ = final_carry
    final_state = eqx.combine(static_state, dynamic_final_state)
    # Return the solution
    solution = method.postprocess(tree_grid, final_state, f_struct, is_leaf=is_leaf)
    return solution
