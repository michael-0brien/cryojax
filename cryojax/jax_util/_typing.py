import typing
from typing import TypeAlias

import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Inexact, Int


if hasattr(typing, "GENERATING_DOCUMENTATION"):

    class NDArrayLike:  # pyright: ignore[reportRedeclaration]
        """A type hint for a JAX or numpy array"""

        pass

    class BoolLike:  # pyright: ignore[reportRedeclaration]
        """A type hint for a python `bool` or a
        JAX/numpy boolean scalar.
        """

        pass

    class ComplexLike:  # pyright: ignore[reportRedeclaration]
        """A type hint for a python `complex` or a
        JAX/numpy complex scalar.
        """

        pass

    class FloatLike:  # pyright: ignore[reportRedeclaration]
        """A type hint for a python `float` or a
        JAX/numpy float scalar.
        """

        pass

    class InexactLike:  # pyright: ignore[reportRedeclaration]
        """A type hint for a python `float` / `complex`, or a
        JAX/numpy float / complex scalar.
        """

        pass

    class IntLike:  # pyright: ignore[reportRedeclaration]
        """A type hint for a python `int`, or a
        JAX/numpy integer scalar.
        """

        pass

else:
    NDArrayLike: TypeAlias = Array | np.ndarray
    BoolLike: TypeAlias = bool | Bool[Array | np.ndarray, ""]
    ComplexLike: TypeAlias = complex | Complex[Array | np.ndarray, ""]
    FloatLike: TypeAlias = float | Float[Array | np.ndarray, ""]
    InexactLike: TypeAlias = complex | float | Inexact[Array | np.ndarray, ""]
    IntLike: TypeAlias = int | Int[Array | np.ndarray, ""]
