import warnings as _warnings
from typing import Any as _Any

from ._dataset import AbstractDataset as AbstractDataset
from ._particle_data import (
    AbstractParticleDataset as AbstractParticleDataset,
    AbstractParticleParameterFile as AbstractParticleParameterFile,
    AbstractParticleStarFile as AbstractParticleStarFile,
    RelionParticleDataset as RelionParticleDataset,
    RelionParticleParameterFile as RelionParticleParameterFile,
    simulate_particle_stack as simulate_particle_stack,
)


_warnings.warn(
    "Submodule `cryojax.dataset` is deprecated and "
    "has been moved to the library `cryospax` "
    "(https://github.com/michael-0brien/cryospax). "
    "`cryojax.dataset` will be removed in cryoJAX 0.6.0.",
    category=FutureWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> _Any:
    # Future deprecations
    if name == "RelionParticleStackDataset":
        _warnings.warn(
            "The 'RelionParticleStackDataset' alias is deprecated and will be removed in "
            "`cryospax`. Use 'RelionParticleDataset' instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return RelionParticleDataset

    if name == "ParticleParameterInfo":
        _warnings.warn(
            "The 'ParticleParameterInfo' TypedDict is deprecated and will be removed in "
            "`cryospax`.",
            category=FutureWarning,
            stacklevel=2,
        )
        from ._particle_data.relion import (
            _ParticleParameterInfo as ParticleParameterInfo,
        )

        return ParticleParameterInfo

    if name == "ParticleStackInfo":
        _warnings.warn(
            "The 'ParticleStackInfo' TypedDict is deprecated and will be removed in "
            "`cryospax`.",
            category=FutureWarning,
            stacklevel=2,
        )
        from ._particle_data.relion import (
            _ParticleStackInfo as ParticleStackInfo,
        )

        return ParticleStackInfo
