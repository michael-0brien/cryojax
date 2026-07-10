"""The contents here are not public API, but are used internally throughout
cryoJAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .jax_util import NDArrayLike, maybe_error_if


#
# Helpers for converting `__init__` inputs to array leaves without
# prematurely moving host-resident data to a device.
#
# Low-level cryoJAX modules store array leaves as `NDArrayLike` (a JAX or
# NumPy array): JAX arrays (including tracers) are kept on-device, while NumPy
# arrays and Python scalars/sequences are kept on the host. This keeps
# host-resident data (and scalar defaults) on the host at "load time" while
# still supporting instantiation inside `jit`/`vmap`: any value that must be
# traced arrives as a `jax.Array` (a tracer), which is caught by the on-device
# branch, whereas a plain Python scalar is always a compile-time constant and
# is safely kept on the host. Both JAX and NumPy arrays are traced by
# equinox's filtered transformations, so either backend behaves as a parameter
# (not a static). Stored leaves are cast to JAX arrays at compute time in the
# relevant functions.
#
def leaf_asarray(x, dtype=float) -> NDArrayLike:
    """Convert an `__init__` input to an array leaf, preserving its backend.

    - If `x` is a JAX array (including a tracer), keep it on-device / traced
      (casting `dtype` if needed).
    - If `x` is a NumPy array, keep it on the host (casting `dtype` if needed).
    - Otherwise (Python scalar, sequence, etc.), convert to a NumPy array on
      the host.

    Routing Python scalars to the host does not break instantiation inside
    `jit`/`vmap`: values that must be traced arrive as `jax.Array` tracers and
    are handled by the first branch, while a plain Python scalar is a
    compile-time constant regardless of context.

    The `dtype` is canonicalized as in `jax.numpy.asarray` so the NumPy and
    JAX branches agree (e.g. `float` maps to `float32` unless `jax_enable_x64`
    is set). If `dtype` is `None`, the dtype of `x` is preserved.
    """
    target = None if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    if isinstance(x, jax.Array):
        return x if (target is None or x.dtype == target) else x.astype(target)
    if isinstance(x, np.ndarray):
        return x if target is None else x.astype(target, copy=False)
    return np.asarray(x, dtype=target)


def leaf_stack(components, *, axis: int = -1, dtype=float) -> NDArrayLike:
    """Stack `__init__` inputs into a single array leaf, preserving backend.

    Each component is converted with [`leaf_asarray`][]. If any resulting
    component is a JAX array (or tracer), the stack is performed with JAX
    (moving any NumPy components on-device); otherwise it is performed with
    NumPy, keeping the result on the host. Stacking uses `axis = -1` by default
    so leading batch dimensions of the components are preserved.
    """
    converted = [leaf_asarray(c, dtype) for c in components]
    if any(isinstance(c, jax.Array) for c in converted):
        return jnp.stack([jnp.asarray(c, dtype=dtype) for c in converted], axis=axis)
    return np.stack(converted, axis=axis)


#
# Helpers for performing internal error checks
#
_make_msg = lambda _s: (
    "While inspecting runtime errors with `CRYOJAX_ENABLE_CHECKS=true`, "
    + _s
    + (
        " Inspect the traceback to determine where this occurred, or "
        "set `EQX_ON_ERROR=breakpoint` and use a debugger."
    )
)


def error_if_negative(x: Array) -> Array:
    return maybe_error_if(
        x,
        lambda _x: _x < 0,
        _make_msg("a non-negative quantity was found to be negative."),
    )


def error_if_not_positive(x: Array) -> Array:
    return maybe_error_if(
        x,
        lambda _x: _x <= 0,
        _make_msg("a positive quantity was found to be negative or zero."),
    )


def error_if_zero(x: Array) -> Array:
    return maybe_error_if(
        x,
        lambda _x: jnp.isclose(_x, 0.0),
        _make_msg("a non-zero quantity was found to be zero."),
    )


def error_if_not_fractional(x: Array) -> Array:
    return maybe_error_if(
        x,
        lambda _x: ~jnp.logical_and(_x >= 0.0, _x <= 1.0),
        _make_msg("a fractional quantity was found to not be between 0 and 1."),
    )
