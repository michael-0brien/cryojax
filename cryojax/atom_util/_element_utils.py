from typing import TypeVar

import jax
import numpy as np
from jaxtyping import Int, PyTree


T = TypeVar("T")


def split_atoms_by_element(
    atomic_numbers: Int[np.ndarray, " _"],
    atom_pytree: PyTree[np.ndarray, "T"],
    atom_axis: int = 0,
) -> PyTree[tuple[np.ndarray, ...], "T"]:
    """Given atomic numbers, split a pytree of numpy arrays
    representing atom properties into a tuple where each element
    is the pytree of tuples where each element is the array for
    a given atomic number.

    **Arguments:**

    - `atomic_numbers`:
        Atomic numbers as a numpy arrays.
    - `atom_pytree`:
        A pytree of numpy arrays,
    - `atom_axis`:
        The axis representing the atom index. Leading axes
        are assumed to be batched axes.

    **Returns:**

    A pytree with tree structure matching `atom_pytree`,
    where arrays have been replaced with tuples of arrays.
    """
    atom_ids = np.unique(atomic_numbers)
    split_pytree = jax.tree.map(
        lambda x: tuple(
            np.take(x, np.where(atomic_numbers == id), axis=atom_axis) for id in atom_ids
        ),
        atom_pytree,
    )
    return split_pytree
