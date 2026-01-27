"""Environmental variables for configuring cryoJAX behavior."""

import os


def _get_bool_var(varname: str, default: bool) -> bool:
    """Read an environmental variable and interpret it as a boolean.

    This function is modified from JAX:
    https://github.com/jax-ml/jax/blob/main/jax/_src/config.py.
    """
    val = os.getenv(varname, str(default))
    val = val.lower()
    if val in ("true", "1"):
        return True
    elif val in ("false", "0"):
        return False
    else:
        raise ValueError(
            f"Unrecognized value for `{varname}`. To enable `{varname}`, "
            "set to `true` or `1`. To disable, set to `false` or `0`."
        )


# Not public API
CRYOJAX_ENABLE_CHECKS: bool = _get_bool_var("CRYOJAX_ENABLE_CHECKS", False)
