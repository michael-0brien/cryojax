from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, PyTree

from ._filter_specs import make_filter_spec


Args = TypeVar("Args")


def make_linear_operator(
    fn: Callable[[Args], Array],
    args: Args,
    where_vector: Callable[[Args], Any],
    *,
    tags: object | Iterable[object] = (),
) -> tuple[lx.FunctionLinearOperator, Args]:
    """Instantiate a [`lineax.FunctionLinearOperator`](https://docs.kidger.site/lineax/api/operators/#lineax.FunctionLinearOperator)
    from a function that takes an arbitrary pytree as input.

    This is useful for converting from the cryoJAX abstraction for image simulation
    to a [`lineax`](https://docs.kidger.site/lineax/) matrix-vector multiplication
    abstraction. It is easy to get backprojection operators using `lineax`, which
    calls [`jax.linear_transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.linear_transpose.html)
    under the hood.

    !!! example "Backprojection with `lineax`"

        ```python
        import cryojax.simulator as cxs
        import cryojax.jax_util as jxu
        import lineax as lx

        # Instantiate a linear operator
        volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(...)
        image_model = cxs.make_image_model(volume, ...)
        where_vector = lambda x: x.volume.fourier_voxel_grid
        operator, vector = jxu.make_linear_operator(
            fn=lambda x: x.simulate(),
            args=image_model,
            where_vector=where_vector,
        )
        # Simulate an image
        image = operator.mv(vector)
        # Compute backprojection
        adjoint = lx.conj(operator.T)
        backprojection = where_vector(adjoint.mv(image))
        ```

    !!! warning

        This function promises that `fn` can be expressed as a
        linear operator with respect to the input arguments at `where_vector`.
        CryoJAX does not explicitly check if this is the case, so JAX will
        throw errors downstream.

    **Arguments:**

    - `fn`:
        A function with signature `out = fn(args)`
    - `args`:
        Input arguments to `fn`
    - `where_vector`:
        A pointer to where the arguments for the volume
        input space are in `args`.
    - `tags`:
        See `lineax.FunctionLinearOperator` for documentation.

    **Returns:**

    A tuple with first element `lineax.FunctionLinearOperator` and second element
    a pytree with the same structure as `pytree`, partitioned to only include the
    arguments at `where_vector`.
    """  # noqa: E501
    # Extract arguments for the volume at `where_vector`
    filter_spec = make_filter_spec(args, where_vector)
    vector_args, other_args = eqx.partition(args, filter_spec)
    vector, static_args = eqx.partition(vector_args, eqx.is_array)
    other_args = eqx.combine(other_args, static_args)
    # Instantiate the `lineax.FunctionLinearOperator`
    fn_wrapper = _FnWrapper(fn, other_args)
    input_structure = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vector
    )
    linear_operator = lx.FunctionLinearOperator(
        fn=fn_wrapper, input_structure=input_structure, tags=tags
    )
    return linear_operator, vector


class _FnWrapper(eqx.Module):
    fn: Callable[[PyTree], Array]
    args: PyTree

    def __call__(self, vector: PyTree) -> Array:
        args = eqx.combine(vector, self.args)
        return self.fn(args)
