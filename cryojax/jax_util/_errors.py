import equinox as eqx
from jaxtyping import ArrayLike, Bool, PyTree

from .._config import CRYOJAX_ENABLE_CHECKS


def maybe_error_if(x: PyTree, pred: Bool[ArrayLike, "..."], msg: str) -> PyTree:
    """Applies [`equinox.error_if`](https://docs.kidger.site/equinox/api/errors/#equinox.error_if)
    depending on the value of the environmental variable `CRYOJAX_ENABLE_CHECKS`.

    - If `CRYOJAX_ENABLE_CHECKS=true`:
        This function is equivalent to `equinox.error_if`.
    - If `CRYOJAX_ENABLE_CHECKS=false`:
        This function is the identity, i.e. `lambda x: x`.

    By default, `CRYOJAX_ENABLE_CHECKS=false` because checks may cause slowdowns, particularly
    on GPU.

    This function is used to achieve a similar idea as
    ['JAX_ENABLE_CHECKS'](https://docs.jax.dev/en/latest/config_options.html#jax_enable_checks)
    in `cryojax` and is exposed as public API for development downstream.
    """  # noqa: E501
    # `enable_checks` keyword is included for unit testing; it is not a public
    # argument.
    if CRYOJAX_ENABLE_CHECKS:
        return eqx.error_if(x, pred, msg)
    else:
        return x
