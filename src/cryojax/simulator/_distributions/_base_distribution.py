"""
Base class for a cryojax distribution.
"""

from abc import abstractmethod

from equinox import Module
from jaxtyping import Array, Float, Inexact, PRNGKeyArray


class AbstractDistribution(Module, strict=True):
    """An image formation model equipped with a probabilistic model."""

    @abstractmethod
    def log_likelihood(self, observed: Inexact[Array, "y_dim x_dim"]) -> Float[Array, ""]:
        """Evaluate the log likelihood.

        **Arguments:**

        - `observed` : The observed data in real or fourier space.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, rng_key: PRNGKeyArray, *, outputs_real_space: bool = True
    ) -> Inexact[Array, "y_dim x_dim"]:
        """Sample from the distribution.

        **Arguments:**

        - `rng_key` : The RNG key or key(s). See `AbstractPipeline.sample` for
                  more documentation.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_signal(
        self, *, outputs_real_space: bool = True
    ) -> Inexact[Array, "y_dim x_dim"]:
        """Render the image formation model."""
        raise NotImplementedError
