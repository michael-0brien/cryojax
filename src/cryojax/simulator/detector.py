"""
Abstraction of electron detectors in a cryo-EM image.
"""

__all__ = [
    "Detector",
    "NullDetector",
    "GaussianDetector",
]

from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .noise import GaussianNoise
from .kernel import KernelType, Constant
from ..core import field, Module
from ..typing import ComplexImage, ImageCoords


class Detector(Module):
    """
    Base class for an electron detector.
    """

    @abstractmethod
    def sample(self, key: PRNGKeyArray, freqs: ImageCoords) -> ComplexImage:
        """Sample a realization from the detector noise model."""
        raise NotImplementedError


class NullDetector(Detector):
    """
    A 'null' detector.
    """

    @override
    def sample(self, key: PRNGKeyArray, freqs: ImageCoords) -> ComplexImage:
        return jnp.zeros(jnp.asarray(freqs).shape[0:-1], dtype=complex)


class GaussianDetector(GaussianNoise, Detector):
    """
    A detector with a gaussian noise model. By default,
    this is a white noise model.

    Attributes
    ----------
    variance :
        A kernel that computes the variance
        of the detector noise. By default,
        ``Constant()``.
    """

    variance: KernelType = field(default_factory=Constant)  # type: ignore

    @override
    def sample(self, key: PRNGKeyArray, freqs: ImageCoords) -> ComplexImage:
        return super().sample(key, freqs)
