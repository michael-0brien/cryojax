"""The contents here are not public API, but are used internally throughout
cryoJAX.
"""

import jax.numpy as jnp
from jaxtyping import Array

from .jax_util import maybe_error_if


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
