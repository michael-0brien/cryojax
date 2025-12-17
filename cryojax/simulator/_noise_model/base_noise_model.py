"""
Base class for a cryojax distribution.
"""

from abc import abstractmethod

import equinox as eqx
import jax
from jaxtyping import Array, Float, Inexact, PRNGKeyArray

from ...ndimage import AbstractFilter, AbstractMask
from .._detector import AbstractDetector
from .._solvent_2d import AbstractRandomSolvent2D


class AbstractNoiseModel(eqx.Module, strict=True):
    """An image formation model equipped with a noise model."""

    @abstractmethod
    def sample(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> Inexact[Array, "y_dim x_dim"]:
        """Sample from the distribution.

        **Arguments:**

        - `rng_key` :
            The RNG key used to sample the noise model.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_signal(
        self,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> Inexact[Array, "y_dim x_dim"]:
        """Compute the signal of the image formation model."""
        raise NotImplementedError


class AbstractLikelihoodNoiseModel(AbstractNoiseModel, strict=True):
    """A noise model equipped with a likelihood."""

    @abstractmethod
    def log_likelihood(
        self,
        observed: Inexact[Array, "y_dim x_dim"],
        *,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the log likelihood.

        **Arguments:**

        - `observed` : The observed data.
        """
        raise NotImplementedError


class AbstractEmpiricalNoiseModel(AbstractNoiseModel, strict=True):
    """A noise model that tries to empirically model the noise in
    the image. This class is not compatible with detector or
    solvent models.
    """

    def __check_init__(self):
        if _has_detector(self):
            cls_name = self.__class__.__name__
            raise ValueError(
                "Found an `AbstractDetector` class when instantiating "
                f"{cls_name}, but cryoJAX `AbstractEmpiricalNoiseModel`s are "
                "not compatible with detector classes."
            )
        if _has_solvent(self):
            cls_name = self.__class__.__name__
            raise ValueError(
                "Found an `AbstractRandomSolvent2D` class when instantiating "
                f"{cls_name}, but cryoJAX `AbstractEmpiricalNoiseModel`s are "
                "not compatible with solvent classes."
            )


def _has_detector(pytree) -> bool:
    is_leaf = lambda x: isinstance(x, AbstractDetector)
    boolean_pytree = jax.tree.map(is_leaf, pytree, is_leaf=is_leaf)
    return jax.tree.reduce(lambda x, y: x or y, boolean_pytree)


def _has_solvent(pytree) -> bool:
    is_leaf = lambda x: isinstance(x, AbstractRandomSolvent2D)
    boolean_pytree = jax.tree.map(is_leaf, pytree, is_leaf=is_leaf)
    return jax.tree.reduce(lambda x, y: x or y, boolean_pytree)
