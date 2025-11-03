from typing import TypeAlias

import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Inexact, Int


NDArrayLike: TypeAlias = Array | np.ndarray
BoolLike: TypeAlias = bool | Bool[Array | np.ndarray, ""]
ComplexLike: TypeAlias = complex | Complex[Array | np.ndarray, ""]
FloatLike: TypeAlias = float | Float[Array | np.ndarray, ""]
InexactLike: TypeAlias = complex | float | Inexact[Array | np.ndarray, ""]
IntLike: TypeAlias = int | Int[Array | np.ndarray, ""]
