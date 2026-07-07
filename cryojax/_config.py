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


def _get_optional_int_var(varname: str) -> int | None:
    """Read an environmental variable and interpret it as an `int`, or
    `None` if unset."""
    val = os.getenv(varname)
    return None if val is None else int(val)


def _get_str_var(varname: str, default: str, choices: tuple[str, ...]) -> str:
    """Read an environmental variable and interpret it as one of `choices`."""
    val = os.getenv(varname, default)
    if val not in choices:
        raise ValueError(
            f"Unrecognized value for `{varname}`. Got `{varname} = {val}`, "
            f"but must be one of {choices}."
        )
    return val


# Not public API
CRYOJAX_ENABLE_CHECKS: bool = _get_bool_var("CRYOJAX_ENABLE_CHECKS", False)
# Not public API
CRYOJAX_ENABLE_PALLAS: bool = _get_bool_var("CRYOJAX_ENABLE_PALLAS", False)
# Not public API
CRYOJAX_PALLAS_BLOCK_SIZE: int | None = _get_optional_int_var("CRYOJAX_PALLAS_BLOCK_SIZE")
# Not public API
CRYOJAX_FINUFFT_BACKEND: str = _get_str_var(
    "CRYOJAX_FINUFFT_BACKEND", "nufftax", ("nufftax", "jax-finufft")
)
