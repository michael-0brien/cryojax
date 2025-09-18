import abc
from typing import Generic, TypeVar
from typing_extensions import Self

import equinox as eqx
from jaxtyping import PRNGKeyArray

from ..base_parametrisation import AbstractVolumeParametrisation


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class AbstractConformationalEnsemble(
    AbstractVolumeParametrisation, Generic[T1, T2], strict=True
):
    """Abstract interface for a volume with conformational
    heterogeneity.
    """

    conformational_space: eqx.AbstractVar[T1]
    """A variable for the ensemble's conformational space."""

    conformation: eqx.AbstractVar[T2]
    """A variable for the ensemble's conformation."""

    @classmethod
    @abc.abstractmethod
    def sample_conformation(cls, rng_key: PRNGKeyArray, conformational_space: T1) -> Self:
        raise NotImplementedError
