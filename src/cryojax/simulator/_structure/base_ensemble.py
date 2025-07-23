"""
Abstractions of ensembles of biological specimen.
"""

from typing import Any, Optional

import equinox as eqx

from .base_structure import AbstractStructure


class AbstractStructuralEnsemble(AbstractStructure, strict=True):
    """A map from a pose and conformational variable to an
    `AbstractPotentialRepresentation`.
    """

    conformation: eqx.AbstractVar[Optional[Any]]
