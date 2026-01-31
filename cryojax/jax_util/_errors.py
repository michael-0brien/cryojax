from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
from jaxtyping import ArrayLike, Bool, PyTree

from .._config import CRYOJAX_ENABLE_CHECKS


T = TypeVar("T", bound="PyTree")


def maybe_error_if(
    x: T, pred_fn: Callable[[T], Bool[ArrayLike, "..."]], msg: str
) -> PyTree:
    """Applies [`equinox.error_if`](https://docs.kidger.site/equinox/api/errors/#equinox.error_if)
    depending on the value of the environmental variable `CRYOJAX_ENABLE_CHECKS`.

    - If `CRYOJAX_ENABLE_CHECKS=true`:
        This function is equivalent to `equinox.error_if`, with the replacement of
        the input `pred` with `pred_fn`, where `pred = pred_fn(x)`. This way, `pred` is only evaluated
        if checks are enabled.
    - If `CRYOJAX_ENABLE_CHECKS=false`:
        This function is the identity, i.e. `lambda x: x`.

    By default, `CRYOJAX_ENABLE_CHECKS=false` because checks may cause slowdowns, particularly
    on GPU.

    This function is used to achieve a similar idea as
    [`JAX_ENABLE_CHECKS`](https://docs.jax.dev/en/latest/config_options.html#jax_enable_checks)
    in `cryojax` and is exposed as public API for development downstream.
    """  # noqa: E501
    if CRYOJAX_ENABLE_CHECKS:
        return eqx.error_if(x, pred_fn(x), msg)
    else:
        return x
