"""This is a module of utilities for working with JAX/Equinox. This includes utilities
for Equinox filtered transformations and Equinox recommendations for creating custom
per-leaf behavior for pytrees.
"""

from ._batched_loop import filter_bmap as filter_bmap, filter_bscan as filter_bscan
from ._errors import (
    error_if_negative as error_if_negative,
    error_if_not_fractional as error_if_not_fractional,
    error_if_not_positive as error_if_not_positive,
    error_if_zero as error_if_zero,
)
from ._filter_specs import make_filter_spec as make_filter_spec
from ._grid_search import (
    AbstractGridSearchMethod as AbstractGridSearchMethod,
    MinimumSearchMethod as MinimumSearchMethod,
    MinimumSolution as MinimumSolution,
    MinimumState as MinimumState,
    run_grid_search as run_grid_search,
    tree_grid_shape as tree_grid_shape,
    tree_grid_take as tree_grid_take,
    tree_grid_unravel_index as tree_grid_unravel_index,
)
from ._pytree_transforms import (
    AbstractPyTreeTransform as AbstractPyTreeTransform,
    CustomTransform as CustomTransform,
    NonArrayStaticTransform as NonArrayStaticTransform,
    StopGradientTransform as StopGradientTransform,
    resolve_transforms as resolve_transforms,
)
from ._typing import (
    BoolLike as BoolLike,
    ComplexLike as ComplexLike,
    FloatLike as FloatLike,
    InexactLike as InexactLike,
    IntLike as IntLike,
    NDArrayLike as NDArrayLike,
)
