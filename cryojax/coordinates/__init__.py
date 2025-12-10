import warnings as _warnings


_warnings.warn(
    "Submodule `cryojax.coordinates` is deprecated and "
    "has been moved to the `cryojax.ndimage` namespace. "
    "`cryojax.coordinates` will be removed in cryoJAX 0.6.0.",
    category=FutureWarning,
    stacklevel=2,
)

from ..ndimage._coordinates import (
    cartesian_to_polar as cartesian_to_polar,
    make_1d_coordinate_grid as make_1d_coordinate_grid,
    make_1d_frequency_grid as make_1d_frequency_grid,
    make_coordinate_grid as make_coordinate_grid,
    make_frequency_grid as make_frequency_grid,
    make_frequency_slice as make_frequency_slice,
    make_radial_coordinate_grid as make_radial_coordinate_grid,
    make_radial_frequency_grid as make_radial_frequency_grid,
)
