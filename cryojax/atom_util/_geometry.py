import numpy as np
from jaxtyping import Float


def center_atom_positions(
    atom_positions: Float[np.ndarray, "... n_atoms 3"],
    atom_masses: Float[np.ndarray, "... n_atoms"] | None = None,
    atom_axis: int = 0,
) -> Float[np.ndarray, "... n_atoms 3"]:
    """Center positions of a numpy array of atoms.

    **Arguments:**

    - `atom_positions`:
        The array of atom positions.
    - `atom_masses`:
        Optionally, provide the masses of the atoms
        to center by mass.
    - `atom_axis`:
        The axis representing the atom index. Leading axes
        are assumed to be batched axes. `atom_positions`
        and `atom_masses` are assumed to have the same
        batched axes.

    **Returns:**

    The centered `atom_positions`. This is centered by
    geometry if `atom_masses = None` and by mass otherwise.
    """
    if atom_masses is None:
        n_atoms = atom_positions.shape[atom_axis]
        center_position = np.sum(atom_positions, axis=atom_axis) / n_atoms
    else:
        center_position = (
            np.sum(atom_positions * atom_masses[..., None], axis=atom_axis)
            / atom_masses.sum(axis=atom_axis)[..., None]
        )
    return atom_positions - np.expand_dims(center_position, axis=atom_axis)
