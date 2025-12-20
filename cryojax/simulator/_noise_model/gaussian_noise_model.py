"""
Image formation models simulated from gaussian noise distributions.
"""

from abc import abstractmethod
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...jax_util import FloatLike, error_if_not_positive
from ...ndimage import (
    AbstractFilter,
    AbstractFourierOperator,
    AbstractMask,
    FourierConstant,
    rfftn,
)
from .._image_model import AbstractImageModel
from .base_noise_model import AbstractEmpiricalNoiseModel, AbstractLikelihoodNoiseModel


RealImageArray = Float[
    Array,
    "{self.image_model.image_config.y_dim} {self.image_model.image_config.x_dim}",  # noqa: E501
]
FourierImageArray = Complex[
    Array,
    "{self.image_model.image_config.y_dim} {self.image_model.image_config.x_dim//2+1}",  # noqa: E501
]
ImageArray = RealImageArray | FourierImageArray


class AbstractGaussianNoiseModel(
    AbstractEmpiricalNoiseModel, AbstractLikelihoodNoiseModel, strict=True
):
    r"""An `AbstractNoiseModel` where images are formed via additive
    gaussian noise.

    Subclasses may compute the likelihood in real or fourier space and
    make different assumptions about the variance / covariance.
    """

    image_model: AbstractVar[AbstractImageModel]
    signal_scale_factor: AbstractVar[Float[Array, ""]]
    signal_offset: AbstractVar[Float[Array, ""]]

    @override
    def sample(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> ImageArray:
        """Sample a noisy image from the gaussian noise model.

        !!! info
            A random number generator key will *not* be passed to
            `AbstractImageModel.simulate`, therefore any
            stochastic elements to the `AbstractImageModel`
            will not be used.

        **Arguments:**

        - `outputs_real_space`:
            If `True`, return the signal in real space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        return self.compute_signal(
            outputs_real_space=outputs_real_space,
            mask=mask,
            filter=filter,
        ) + self.compute_noise(
            rng_key,
            outputs_real_space=outputs_real_space,
            mask=mask,
            filter=filter,
        )

    @override
    def compute_signal(
        self,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> ImageArray:
        """Render the signal from the image formation model.

        **Arguments:**

        - `outputs_real_space`:
            If `True`, return the signal in real space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        simulated_image = self.image_model.simulate(
            outputs_real_space=True, mask=None, filter=filter
        )
        simulated_image = self.signal_scale_factor * simulated_image + self.signal_offset
        if mask is not None:
            simulated_image = mask(simulated_image)
        return simulated_image if outputs_real_space else rfftn(simulated_image)

    @abstractmethod
    def compute_noise(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> ImageArray:
        """Draw a realization from the gaussian noise model and return either in
        real or fourier space.
        """
        raise NotImplementedError


class GaussianWhiteNoiseModel(AbstractGaussianNoiseModel, strict=True):
    r"""A gaussian noise model, where each pixel is independently drawn from
    a zero-mean gaussian of fixed variance (white noise).

    This computes the likelihood in real space, where the variance is a
    constant value across all pixels.
    """

    image_model: AbstractImageModel
    variance: Float[Array, ""]
    signal_scale_factor: Float[Array, ""]
    signal_offset: Float[Array, ""]

    def __init__(
        self,
        image_model: AbstractImageModel,
        variance: FloatLike = 1.0,
        signal_scale_factor: FloatLike = 1.0,
        signal_offset: FloatLike = 0.0,
    ):
        """**Arguments:**

        - `image_model`:
            The image formation model.
        - `variance`:
            The variance of each pixel.
        - `signal_scale_factor`:
            A scale factor for the underlying signal simulated
            from `image_model`.
        - `signal_offset`:
            An offset for the underlying signal simulated from `image_model`.
        """  # noqa: E501
        self.image_model = image_model
        self.variance = error_if_not_positive(jnp.asarray(variance, dtype=float))
        self.signal_scale_factor = error_if_not_positive(
            jnp.asarray(signal_scale_factor, dtype=float)
        )
        self.signal_offset = jnp.asarray(signal_offset, dtype=float)

    @override
    def compute_noise(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> ImageArray:
        """Sample a realization of the noise from the distribution.

        **Arguments:**

        - `outputs_real_space`:
            If `True`, return the noise in real space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        n_pixels = self.image_model.image_config.padded_n_pixels
        frequency_grid = self.image_model.image_config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(n_pixels * self.variance)
        noise = self.image_model.postprocess(
            std
            * jr.normal(rng_key, shape=frequency_grid.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            outputs_real_space=outputs_real_space,
            mask=mask,
            filter=filter,
            options=dict(do_normalize=False, do_transform=False),
        )

        return noise

    @override
    def log_likelihood(
        self,
        observed: RealImageArray,
        *,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the log-likelihood of the gaussian noise model.

        !!! info

            When computing the likelihood, the observed image is assumed to have already
            been preprocessed with `filter` and `mask`. In other words,
            if `filter` or `mask` are `None`, these
            will *not* be applied to `observed`. The user must do this
            manually if desired.

        **Arguments:**

        - `observed` : The observed data in real space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        variance = self.variance
        # Create simulated data
        simulated = self.compute_signal(outputs_real_space=True, mask=mask, filter=filter)
        # Compute residuals
        residuals = simulated - observed
        # Compute standard normal random variables
        squared_standard_normal_per_pixel = jnp.abs(residuals) ** 2 / (2 * variance)
        # Compute the log-likelihood for each pixel.
        log_likelihood_per_pixel = -1.0 * (
            squared_standard_normal_per_pixel + jnp.log(2 * jnp.pi * variance) / 2
        )
        # Compute log-likelihood, summing over pixels
        log_likelihood = jnp.sum(log_likelihood_per_pixel)

        return log_likelihood


class GaussianColoredNoiseModel(AbstractGaussianNoiseModel, strict=True):
    r"""A gaussian noise model, where pixels are correlated, but each
    frequency is independent (colored noise).

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.
    """

    image_model: AbstractImageModel
    variance_fn: AbstractFourierOperator
    signal_scale_factor: Float[Array, ""]
    signal_offset: Float[Array, ""]

    def __init__(
        self,
        image_model: AbstractImageModel,
        variance_fn: AbstractFourierOperator | None = None,
        signal_scale_factor: FloatLike = 1.0,
        signal_offset: FloatLike = 0.0,
    ):
        """**Arguments:**

        - `image_model`:
            The image formation model.
        - `variance_fn`:
            The variance of each fourier mode. By default,
            `cryojax.ndimage.FourierConstant(1.0)`.
        - `signal_scale_factor`:
            A scale factor for the underlying signal simulated from `image_model`.
        - `signal_offset`:
            An offset for the underlying signal simulated from `image_model`.
        """  # noqa: E501
        self.image_model = image_model
        self.variance_fn = variance_fn or FourierConstant(1.0)
        self.signal_scale_factor = error_if_not_positive(
            jnp.asarray(signal_scale_factor, dtype=float)
        )
        self.signal_offset = jnp.asarray(signal_offset, dtype=float)

    def compute_noise(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> ImageArray:
        """Sample a realization of the noise from the distribution.

        **Arguments:**

        - `outputs_real_space`:
            If `True`, return the noise in real space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        n_pixels = self.image_model.image_config.padded_n_pixels
        frequency_grid = self.image_model.image_config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(n_pixels * self.variance_fn(frequency_grid))
        noise = self.image_model.postprocess(
            std
            * jr.normal(rng_key, shape=frequency_grid.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            outputs_real_space=outputs_real_space,
            mask=mask,
            filter=filter,
            options=dict(do_normalize=False, do_transform=False),
        )

        return noise

    @override
    def log_likelihood(
        self,
        observed: FourierImageArray,
        *,
        mask: AbstractMask | None = None,
        filter: AbstractFilter | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the log-likelihood of the gaussian noise model.

        !!! info

            When computing the likelihood, the observed image is assumed to have already
            been preprocessed with `filter` and `mask`. In other words,
            if `filter` or `mask` are `None`, these
            will *not* be applied to `observed`. The user must do this
            manually if desired.

        **Arguments:**

        - `observed` : The observed data in fourier space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        n_pixels = self.image_config.n_pixels
        frequency_grid = self.image_config.frequency_grid_in_angstroms
        # Compute the variance and scale up to be independent of the number of pixels
        variance = n_pixels * self.variance_fn(frequency_grid)
        # Create simulated data
        simulated = self.compute_signal(
            outputs_real_space=False,
            mask=mask,
            filter=filter,
        )
        # Compute residuals
        residuals = simulated - observed
        # Compute standard normal random variables
        squared_standard_normal_per_mode = jnp.abs(residuals) ** 2 / (2 * variance)
        # Compute the log-likelihood for each fourier mode.
        log_likelihood_per_mode = (
            squared_standard_normal_per_mode + jnp.log(2 * jnp.pi * variance) / 2
        )
        # Compute log-likelihood, throwing away the zero mode. Need to take care
        # to compute the loss function in fourier space for a real-valued function.
        log_likelihood = (
            -1.0
            * (
                jnp.sum(log_likelihood_per_mode[1:, 0])
                + 2 * jnp.sum(log_likelihood_per_mode[:, 1:])
            )
            / n_pixels
        )

        return log_likelihood
