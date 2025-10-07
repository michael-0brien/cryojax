"""An interface for a grid search method."""

import math
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int, PyTree

from .custom_types import PyTreeGrid, PyTreeGridPoint, SearchSolution, SearchState
from .pytree_manipulation import (
    tree_grid_shape,
    tree_grid_take,
    tree_grid_unravel_index,
)


class AbstractGridSearchMethod(
    eqx.Module, Generic[SearchState, SearchSolution], strict=True
):
    """An abstract interface that determines the behavior of the grid
    search.
    """

    batch_size: eqx.AbstractVar[int | None]

    @abstractmethod
    def init(
        self,
        tree_grid: PyTreeGrid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> SearchState:
        """Initialize the state of the search method.

        **Arguments:**

        - `tree_grid`: As [`run_grid_search`][].
        - `f_struct`: A container that stores the `shape` and `dtype`
                      returned by `fn`.
        - `is_leaf`: As [`run_grid_search`][].

        **Returns:**

        Any pytree that represents the state of the grid search.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point: PyTreeGridPoint,
        args: Any,
        state: SearchState,
        raveled_grid_index: Int[Array, ""],
    ) -> SearchState:
        """Update the state of the grid search.

        **Arguments:**

        - `fn`: As [`run_grid_search`][].
        - `tree_grid_point`: The grid point at which to evaluate `fn`. Specifically,
                             `fn` is evaluated as `fn(tree_grid_point, args)`.
        - `args`: As [`run_grid_search`][].
        - `state`: The current state of the search.
        - `raveled_grid_index`: The current index of the grid. This is
                                used to index `tree_grid` to extract the
                                `tree_grid_point`.

        **Returns:**

        The updated state of the grid search.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_update(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point_batch: PyTreeGridPoint,
        args: Any,
        state: SearchState,
        raveled_grid_index_batch: Int[Array, " _"],
    ) -> SearchState:
        """Update the state of the grid search with a batch of grid points as
        input.

        **Arguments:**

        - `fn`: As [`run_grid_search`][].
        - `tree_grid_point_batch`: The grid points at which to evaluate `fn` in
                                   parallel.
        - `args`: As [`run_grid_search`][].
        - `state`: The current state of the search.
        - `raveled_grid_index_batch`: The current batch of indices on which to evaluate
                                      the grid.

        **Returns:**

        The updated state of the grid search.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: SearchState,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> SearchSolution:
        """Post-process the final state of the grid search into a
        solution.

        **Arguments:**

        - `tree_grid`: As [`run_grid_search`][].
        - `final_state`: The final state of the grid search.
        - `f_struct`: A container that stores the `shape` and `dtype`
                      returned by `fn`.
        - `is_leaf`: As [`run_grid_search`][].

        **Returns:**

        Any pytree that represents the solution of the grid search.
        """
        raise NotImplementedError


class MinimumState(eqx.Module, strict=True):
    current_minimum_eval: Array
    current_best_raveled_index: Array
    current_eval: Array | None = None


class MinimumSolution(eqx.Module, strict=True):
    value: PyTreeGridPoint | None
    stats: dict[str, Any]
    state: MinimumState


class MinimumSearchMethod(
    AbstractGridSearchMethod[MinimumState, MinimumSolution], strict=True
):
    """Simply find the minimum value returned by `fn` over all grid points.

    The minimization is done *elementwise* for the output returned by `fn(y, args)`.
    This allows for more clever grid searches than a brute-force approach--for example,
    `fn` can explore its own region of parameter space in parallel.
    """

    stores_solution_value: bool
    stores_current_eval: bool
    batch_size: int | None

    def __init__(
        self,
        *,
        stores_solution_value: bool = True,
        stores_current_eval: bool = False,
        batch_size: int | None = None,
    ):
        """**Arguments:**

        - `stores_solution_value`: If `True`, the grid search solution will contain the
                                best grid point found. If `False`, only the flattened
                                index corresponding to these grid points are returned
                                and [`tree_grid_take`][] must be used to extract the
                                actual grid points. Setting this to `False` may be
                                necessary if the grid contains large arrays.
        - `stores_current_eval`: If `True`, carry over the last function evaluation in
                                the `MinimumState`. This is useful when wrapping this
                                class into new `AbstractGridSearchMethod`s.
        - `batch_size`: The stride of grid points over which to evaluate in parallel.
        """
        self.stores_solution_value = stores_solution_value
        self.stores_current_eval = stores_current_eval
        self.batch_size = batch_size

    def init(
        self,
        tree_grid: PyTreeGrid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> MinimumState:
        # Initialize the state, just keeping track of the best function values
        # and their respective grid index
        return MinimumState(
            current_minimum_eval=jnp.full(f_struct.shape, jnp.inf, dtype=float),
            current_best_raveled_index=jnp.full(f_struct.shape, 0, dtype=int),
            current_eval=(
                (
                    jnp.full(f_struct.shape, 0.0, dtype=float)
                    if self.batch_size is None
                    else jnp.full((self.batch_size, *f_struct.shape), 0.0, dtype=float)
                )
                if self.stores_current_eval
                else None
            ),
        )

    def update(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point: PyTreeGridPoint,
        args: Any,
        state: MinimumState,
        raveled_grid_index: Int[Array, ""],
    ) -> MinimumState:
        # Evaluate the function
        value = fn(tree_grid_point, args)
        # Unpack the current state
        last_minimum_value = state.current_minimum_eval
        last_best_raveled_index = state.current_best_raveled_index
        # Update the minimum and best grid index, elementwise
        is_less_than_last_minimum = value < last_minimum_value
        current_minimum_eval = jnp.where(
            is_less_than_last_minimum, value, last_minimum_value
        )
        current_best_raveled_index = jnp.where(
            is_less_than_last_minimum, raveled_grid_index, last_best_raveled_index
        )
        return MinimumState(
            current_minimum_eval,
            current_best_raveled_index,
            current_eval=value if self.stores_current_eval else None,
        )

    def batch_update(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point_batch: PyTreeGridPoint,
        args: Any,
        state: MinimumState,
        raveled_grid_index_batch: Int[Array, " _"],
    ) -> MinimumState:
        # Evaluate the batch of grid points and extract the best one
        value_batch = jax.vmap(fn, in_axes=[0, None])(tree_grid_point_batch, args)
        best_batch_index = jnp.argmin(value_batch, axis=0)
        raveled_grid_index = jnp.take(raveled_grid_index_batch, best_batch_index)
        value = jnp.amin(value_batch, axis=0)
        # Unpack the current state
        last_minimum_value = state.current_minimum_eval
        last_best_raveled_index = state.current_best_raveled_index
        # Update the minimum and best grid index, elementwise
        is_less_than_last_minimum = value < last_minimum_value
        current_minimum_eval = jnp.where(
            is_less_than_last_minimum, value, last_minimum_value
        )
        current_best_raveled_index = jnp.where(
            is_less_than_last_minimum, raveled_grid_index, last_best_raveled_index
        )
        return MinimumState(
            current_minimum_eval,
            current_best_raveled_index,
            current_eval=value_batch if self.stores_current_eval else None,
        )

    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: MinimumState,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> MinimumSolution:
        # Make sure that shapes did not get modified during loop
        if final_state.current_best_raveled_index.shape != f_struct.shape:
            raise ValueError(
                "The shape of the search state solution does "
                "not match the shape of the output of `fn`. Got "
                f"output shape {f_struct.shape} for `fn`, but got "
                f"shape {final_state.current_best_raveled_index.shape} for the "
                "solution."
            )
        if self.stores_solution_value:
            # Extract the solution of the search, i.e. the grid point(s) corresponding
            # to the raveled grid index
            if f_struct.shape == ():
                raveled_index = final_state.current_best_raveled_index
            else:
                raveled_index = final_state.current_best_raveled_index.ravel()
            # ... get the pytree representation of the index
            tree_grid_index = tree_grid_unravel_index(
                raveled_index, tree_grid, is_leaf=is_leaf
            )
            # ... index the full grid, reshaping the solution's leaves to be the same
            # shape as returned by `fn`
            _reshape_fn = lambda x: (
                x.reshape((*f_struct.shape, *x.shape[1:]))
                if x.ndim > 1
                else x.reshape(f_struct.shape)
            )
            value = jtu.tree_map(_reshape_fn, tree_grid_take(tree_grid, tree_grid_index))
        else:
            value = None
        # ... build and return the solution
        return MinimumSolution(
            value,
            {"grid_size": math.prod(tree_grid_shape(tree_grid, is_leaf=is_leaf))},
            final_state,
        )
