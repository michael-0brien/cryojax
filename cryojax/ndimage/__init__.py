import warnings as _warnings
from typing import Any as _Any

from ._coordinates import (
    cartesian_to_polar as cartesian_to_polar,
    make_1d_coordinate_grid as make_1d_coordinate_grid,
    make_1d_frequency_grid as make_1d_frequency_grid,
    make_coordinate_grid as make_coordinate_grid,
    make_frequency_grid as make_frequency_grid,
    make_frequency_slice as make_frequency_slice,
    make_radial_coordinate_grid as make_radial_coordinate_grid,
    make_radial_frequency_grid as make_radial_frequency_grid,
)
from ._downsample import (
    block_reduce_downsample as block_reduce_downsample,
    fourier_crop_downsample as fourier_crop_downsample,
    fourier_crop_to_shape as fourier_crop_to_shape,
)
from ._edges import (
    crop_to_shape as crop_to_shape,
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
from ._normalize import (
    background_subtract_image as background_subtract_image,
    compute_edge_value as compute_edge_value,
    rescale_image as rescale_image,
    standardize_image as standardize_image,
)
from ._operators import (
    AbstractFourierOperator as AbstractFourierOperator,
    AbstractRealOperator as AbstractRealOperator,
    CustomFourierOperator as CustomFourierOperator,
    FourierConstant as FourierConstant,
    FourierDC as FourierDC,
    FourierExp2D as FourierExp2D,
    FourierGaussian as FourierGaussian,
    FourierSinc as FourierSinc,
    PeakedFourierGaussian as PeakedFourierGaussian,
    RealConstant as RealConstant,
    RealGaussian as RealGaussian,
)
from ._radial_average import (
    compute_binned_radial_average as compute_binned_radial_average,
    radial_average_to_grid as radial_average_to_grid,
)
from ._rescale_pixel_size import (
    rescale_pixel_size as rescale_pixel_size,
)
from ._transforms import (
    AbstractFilter as AbstractFilter,
    AbstractImageTransform as AbstractImageTransform,
    AbstractMask as AbstractMask,
    CircularCosineMask as CircularCosineMask,
    CustomFilter as CustomFilter,
    CustomMask as CustomMask,
    Cylindrical2DCosineMask as Cylindrical2DCosineMask,
    HighpassFilter as HighpassFilter,
    ImageScaling as ImageScaling,
    LowpassFilter as LowpassFilter,
    Rectangular2DCosineMask as Rectangular2DCosineMask,
    Rectangular3DCosineMask as Rectangular3DCosineMask,
    SincCorrectionMask as SincCorrectionMask,
    SphericalCosineMask as SphericalCosineMask,
    SquareCosineMask as SquareCosineMask,
    WhiteningFilter as WhiteningFilter,
)


def __getattr__(name: str) -> _Any:
    # Future deprecations
    if name == "downsample_with_fourier_cropping":
        _warnings.warn(
            "'downsample_with_fourier_cropping' is deprecated"
            "has been renamed to 'fourier_crop_downsample'. "
            "The old name will be deprecated in cryoJAX 0.6.0.",
            category=FutureWarning,
            stacklevel=2,
        )
        return fourier_crop_downsample
    if name == "downsample_to_shape_with_fourier_cropping":
        _warnings.warn(
            "'downsample_to_shape_with_fourier_cropping' is deprecated"
            "has been renamed to 'fourier_crop_to_shape'. "
            "The old name will be deprecated in cryoJAX 0.6.0.",
            category=FutureWarning,
            stacklevel=2,
        )
        return fourier_crop_to_shape
    if name == "normalize_image":
        _warnings.warn(
            "'normalize_image' is deprecated and "
            "has been renamed to 'standardize_image'. "
            "The old name will be deprecated in cryoJAX 0.6.0.",
            category=FutureWarning,
            stacklevel=2,
        )
        return standardize_image
    if name == "operators":
        _warnings.warn(
            "Submodule `cryojax.ndimage.operators` is deprecated and "
            "has been moved to the `cryojax.ndimage` namespace. "
            "`cryojax.ndimage.operators` will be removed in cryoJAX 0.6.0.",
            category=FutureWarning,
            stacklevel=2,
        )
        from . import _operators as operators

        return operators
    if name == "transforms":
        _warnings.warn(
            "Submodule `cryojax.ndimage.transforms` is deprecated and "
            "has been moved to the `cryojax.ndimage` namespace. "
            "`cryojax.ndimage.transforms` will be removed in cryoJAX 0.6.0.",
            category=FutureWarning,
            stacklevel=2,
        )
        from . import _transforms as transforms

        return transforms

    raise AttributeError(f"cannot import name '{name}' from 'cryojax.ndimage'.")
