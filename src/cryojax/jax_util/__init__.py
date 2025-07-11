"""This is a module of utilities for working with JAX/Equinox. This includes utilities
for Equinox filtered transformations and Equinox recommendations for creating custom
per-leaf behavior for pytrees.
"""

from ._batched_loop import batched_map as batched_map, batched_scan as batched_scan
from ._filter_specs import get_filter_spec as get_filter_spec
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
    StopGradientTransform as StopGradientTransform,
    resolve_transforms as resolve_transforms,
)
