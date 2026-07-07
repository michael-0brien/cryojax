from typing import Any as _Any

from . import (
    atom_util as atom_util,
    constants as constants,
    experimental as experimental,
    io as io,
    jax_util as jax_util,
    ndimage as ndimage,
    rotations as rotations,
    simulator as simulator,
)
from .cryojax_version import __version__ as __version__


def __getattr__(name: str) -> _Any:
    if name == "coordinates":
        raise ImportError(
            "Submodule `cryojax.coordinates` was removed in cryoJAX 0.6.0. "
            "Use `cryojax.ndimage` instead."
        )
    if name == "dataset":
        raise ImportError(
            "Submodule `cryojax.dataset` was removed in cryoJAX 0.6.0. "
            "This functionality has moved to the library `cryospax` "
            "(https://github.com/michael-0brien/cryospax)."
        )
    raise AttributeError(f"module 'cryojax' has no attribute '{name}'")
