"""This is a module of utilities for working with JAX/Equinox. This includes utilities
for Equinox filtered transformations and Equinox recommendations for creating custom
per-leaf behavior for pytrees.
"""

from ._batched_loop import filter_bmap as filter_bmap, filter_bscan as filter_bscan
from ._errors import maybe_error_if as maybe_error_if
from ._filter_specs import make_filter_spec as make_filter_spec
from ._grid_search import (
    AbstractGridSearchMethod as AbstractGridSearchMethod,
    MinimumSearchMethod as MinimumSearchMethod,
    run_grid_search as run_grid_search,
    tree_grid_shape as tree_grid_shape,
    tree_grid_take as tree_grid_take,
    tree_grid_unravel_index as tree_grid_unravel_index,
)
from ._linear_operator import make_linear_operator as make_linear_operator
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
