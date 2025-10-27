import warnings as _warnings
from typing import Any as _Any

from . import operators as operators, transforms as transforms
from ._downsample import (
    block_reduce_downsample as block_reduce_downsample,
    fourier_crop_downsample as fourier_crop_downsample,
    fourier_crop_downsample_to_shape as fourier_crop_downsample_to_shape,
)
from ._edges import (
    crop_to_shape as crop_to_shape,
    crop_to_shape_with_center as crop_to_shape_with_center,
    pad_to_shape as pad_to_shape,
    resize_with_crop_or_pad as resize_with_crop_or_pad,
)
from ._fft import (
    fftn as fftn,
    ifftn as ifftn,
    irfftn as irfftn,
    rfftn as rfftn,
)
from ._fourier_statistics import (
    compute_binned_powerspectrum as compute_binned_powerspectrum,
    compute_fourier_ring_correlation as compute_fourier_ring_correlation,
    compute_fourier_shell_correlation as compute_fourier_shell_correlation,
)
from ._fourier_utils import (
    convert_fftn_to_rfftn as convert_fftn_to_rfftn,
    enforce_self_conjugate_rfftn_components as enforce_self_conjugate_rfftn_components,
)
from ._map_coordinates import (
    compute_spline_coefficients as compute_spline_coefficients,
    map_coordinates as map_coordinates,
    map_coordinates_spline as map_coordinates_spline,
)
from ._normalize import normalize_image as normalize_image, rescale_image as rescale_image
from ._radial_average import (
    compute_binned_radial_average as compute_binned_radial_average,
    interpolate_radial_average_on_grid as interpolate_radial_average_on_grid,
)
from ._rescale_pixel_size import (
    rescale_pixel_size as rescale_pixel_size,
)


def __getattr__(name: str) -> _Any:
    # Future deprecations
    if name == "downsample_with_fourier_cropping":
        _warnings.warn(
            "'downsample_with_fourier_cropping' is deprecated"
            "has been renamed to 'fourier_crop_downsample'. "
            "The old name will be deprecated in cryoJAX 0.6.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return fourier_crop_downsample
    if name == "downsample_to_shape_with_fourier_cropping":
        _warnings.warn(
            "'downsample_to_shape_with_fourier_cropping' is deprecated"
            "has been renamed to 'fourier_crop_downsample_to_shape'. "
            "The old name will be deprecated in cryoJAX 0.6.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return fourier_crop_downsample_to_shape

    raise AttributeError(f"cannot import name '{name}' from 'cryojax.ndimage'.")
