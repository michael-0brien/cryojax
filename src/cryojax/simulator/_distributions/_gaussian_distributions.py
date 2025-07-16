"""
Image formation models simulated from gaussian noise distributions.
"""

from abc import abstractmethod
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar, field
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...internal import NDArrayLike, error_if_not_positive
from ...ndimage import rfftn
from ...ndimage.operators import Constant, FourierOperatorLike
from ...ndimage.transforms import AbstractBooleanMask, FilterLike, MaskLike
from .._image_model import AbstractImageModel
from ._base_distribution import AbstractDistribution


RealImageArray = Float[
    Array,
    "{self.image_model.config.y_dim} {self.image_model.config.x_dim}",  # noqa: E501
]
FourierImageArray = Complex[
    Array,
    "{self.image_model.config.y_dim} {self.image_model.config.x_dim//2+1}",  # noqa: E501
]
ImageArray = RealImageArray | FourierImageArray


class AbstractGaussianDistribution(AbstractDistribution, strict=True):
    r"""An `AbstractDistribution` where images are formed via additive
    gaussian noise.

    Subclasses may compute the likelihood in real or fourier space and
    make different assumptions about the variance / covariance.
    """

    image_model: AbstractVar[AbstractImageModel]
    signal_scale_factor: AbstractVar[Float[Array, ""]]
    signal_offset: AbstractVar[Float[Array, ""]]

    normalizes_signal: AbstractVar[bool]
    applies_mask: AbstractVar[bool]

    @override
    def sample(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> ImageArray:
        """Sample a noisy image from the gaussian noise model.

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
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> ImageArray:
        """Render the signal from the image formation model.

        !!! info

            If a `mask` is passed, images will be
            normalized with the mean and standard deviation computed
            within the region where the mask is equal to 1.

            In particular, the following code is used

            ```python
                import jax.numpy as jnp

                signal = ...
                mask_array = ...
                is_signal = mask_array == 1.0
                mean, std = (
                    jnp.mean(signal, where=is_signal),
                    jnp.std(signal, where=is_signal),
                )
                normalized_signal = (signal - mean) / std
            ```

            If `applies_mask = False`, the mask will not be applied to the signal
            but it will still be used for normalization.

        **Arguments:**

        - `outputs_real_space`:
            If `True`, return the signal in real space.
        - `mask`:
            A mask to apply to the final image and/or
            use for normalization.
        - `filter`:
            A filter to apply to the final image.
        """
        simulated_image = self.image_model.render(
            outputs_real_space=True, mask=None, filter=filter
        )
        if mask is None:
            if self.normalizes_signal:
                mean, std = jnp.mean(simulated_image), jnp.std(simulated_image)
                simulated_image = (simulated_image - mean) / std
            simulated_image = (
                self.signal_scale_factor * simulated_image + self.signal_offset
            )
        else:
            if self.normalizes_signal:
                if isinstance(mask, AbstractBooleanMask):
                    is_signal = mask.is_not_masked
                else:
                    is_signal = mask.array == 1.0
                mean, std = (
                    jnp.mean(simulated_image, where=is_signal),
                    jnp.std(simulated_image, where=is_signal),
                )
                simulated_image = (simulated_image - mean) / std
            simulated_image = (
                self.signal_scale_factor * simulated_image + self.signal_offset
            )
            if self.applies_mask:
                simulated_image = mask(simulated_image)
        return simulated_image if outputs_real_space else rfftn(simulated_image)

    @abstractmethod
    def compute_noise(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
    ) -> ImageArray:
        """Draw a realization from the gaussian noise model and return either in
        real or fourier space.
        """
        raise NotImplementedError


class IndependentGaussianPixels(AbstractGaussianDistribution, strict=True):
    r"""A gaussian noise model, where each pixel is independently drawn from
    a zero-mean gaussian of fixed variance (white noise).

    This computes the likelihood in real space, where the variance is a
    constant value across all pixels.
    """

    image_model: AbstractImageModel
    variance: Float[Array, ""]
    signal_scale_factor: Float[Array, ""]
    signal_offset: Float[Array, ""]

    normalizes_signal: bool = field(static=True)
    applies_mask: bool = field(static=True)

    def __init__(
        self,
        image_model: AbstractImageModel,
        variance: float | Float[NDArrayLike, ""] = 1.0,
        signal_scale_factor: float | Float[NDArrayLike, ""] = 1.0,
        signal_offset: float | Float[NDArrayLike, ""] = 0.0,
        normalizes_signal: bool = False,
        applies_mask: bool = False,
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
        - `normalizes_signal`:
            Whether or not the signal is normalized before applying the `signal_scale_factor`
            and `signal_offset`.
            If an `AbstractMask` is given to `image_model.mask`, the signal is normalized
            within the region where the mask is equal to `1`.
        - `applies_mask`:
            If `True`, apply masks passed at run-time. Otherwise, just
            use masks for normalization.
        """  # noqa: E501
        self.image_model = image_model
        self.variance = error_if_not_positive(jnp.asarray(variance, dtype=float))
        self.signal_scale_factor = error_if_not_positive(
            jnp.asarray(signal_scale_factor, dtype=float)
        )
        self.signal_offset = jnp.asarray(jnp.asarray(signal_offset, dtype=float))
        self.normalizes_signal = normalizes_signal
        self.applies_mask = applies_mask

    @override
    def compute_noise(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
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
        pipeline = self.image_model
        n_pixels = pipeline.config.padded_n_pixels
        freqs = pipeline.config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(n_pixels * self.variance)
        noise = pipeline.postprocess(
            std
            * jr.normal(rng_key, shape=freqs.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            outputs_real_space=outputs_real_space,
            mask=(mask if self.applies_mask else None),
            filter=filter,
        )

        return noise

    @override
    def log_likelihood(
        self,
        observed: Float[
            Array,
            "{self.image_model.config.y_dim} " "{self.image_model.config.x_dim}",
        ],
        *,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
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
        simulated = self.compute_signal(
            outputs_real_space=True,
            mask=mask,
            filter=filter,
        )
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


class IndependentGaussianFourierModes(AbstractGaussianDistribution, strict=True):
    r"""A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.
    """

    image_model: AbstractImageModel
    variance_function: FourierOperatorLike
    signal_scale_factor: Float[Array, ""]
    signal_offset: Float[Array, ""]

    normalizes_signal: bool = field(static=True)
    applies_mask: bool = field(static=True)

    def __init__(
        self,
        image_model: AbstractImageModel,
        variance_function: Optional[FourierOperatorLike] = None,
        signal_scale_factor: float | Float[NDArrayLike, ""] = 1.0,
        signal_offset: float | Float[NDArrayLike, ""] = 0.0,
        normalizes_signal: bool = False,
        applies_mask: bool = False,
    ):
        """**Arguments:**

        - `image_model`:
            The image formation model.
        - `variance_function`:
            The variance of each fourier mode. By default,
            `cryojax.image.operators.Constant(1.0)`.
        - `signal_scale_factor`:
            A scale factor for the underlying signal simulated from `image_model`.
        - `signal_offset`:
            An offset for the underlying signal simulated from `image_model`.
        - `normalizes_signal`:
            Whether or not the signal is normalized before applying the `signal_scale_factor`
            and `signal_offset`.
            If an `AbstractMask` is given to `image_model.mask`, the signal is normalized
            within the region where the mask is equal to `1`.
        - `applies_mask`:
            If `True`, apply masks passed at run-time. Otherwise, just
            use masks for normalization.
        """  # noqa: E501
        self.image_model = image_model
        self.variance_function = variance_function or Constant(1.0)
        self.signal_scale_factor = error_if_not_positive(
            jnp.asarray(signal_scale_factor, dtype=float)
        )
        self.signal_offset = jnp.asarray(signal_offset, dtype=float)
        self.normalizes_signal = normalizes_signal
        self.applies_mask = applies_mask

    def compute_noise(
        self,
        rng_key: PRNGKeyArray,
        *,
        outputs_real_space: bool = True,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
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
        pipeline = self.image_model
        n_pixels = pipeline.config.padded_n_pixels
        freqs = pipeline.config.padded_frequency_grid_in_angstroms
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        std = jnp.sqrt(n_pixels * self.variance_function(freqs))
        noise = pipeline.postprocess(
            std
            * jr.normal(rng_key, shape=freqs.shape[0:-1])
            .at[0, 0]
            .set(0.0)
            .astype(complex),
            outputs_real_space=outputs_real_space,
            mask=(mask if self.applies_mask else None),
            filter=filter,
        )

        return noise

    @override
    def log_likelihood(
        self,
        observed: Complex[
            Array,
            "{self.image_model.config.y_dim} " "{self.image_model.config.x_dim//2+1}",
        ],
        *,
        mask: Optional[MaskLike] = None,
        filter: Optional[FilterLike] = None,
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
        pipeline = self.image_model
        n_pixels = pipeline.config.n_pixels
        freqs = pipeline.config.frequency_grid_in_angstroms
        # Compute the variance and scale up to be independent of the number of pixels
        variance = n_pixels * self.variance_function(freqs)
        # Create simulated data
        simulated = self.compute_signal(
            outputs_real_space=False,
            mask=(mask if self.applies_mask else None),
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
