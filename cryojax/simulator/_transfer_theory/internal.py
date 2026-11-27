"""Phase shifts of the CTF, in angstroms and radians. Not currently public API.

Each m-fold aberration is a vector `c (cos(m angle), sin(m angle))`, contracted with the
harmonics `(Re, Im)` of `(q_y + i q_x)^m`. The phase shifts are then polynomials in
`(q_x, q_y)`, so their derivatives stay finite at the origin.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import FloatLike


def compute_even_aberration_phase_shifts(
    frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    wavelength_in_angstroms: Float[Array, ""],
    defocus_in_angstroms: Float[Array, ""],
    astigmatism_xy_in_angstroms: Float[Array, "2"],
    spherical_aberration_in_angstroms: Float[Array, ""],
    offset_z_in_angstroms: FloatLike | None = None,
    nominal_frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"] | None = None,
) -> Float[Array, "y_dim x_dim"]:
    """The defocus, astigmatism, and spherical aberration phase shifts. As in
    CTFFIND4, the defocus is `defocus + astigmatism cos(2 (azimuth - angle)) / 2`.
    Given the nominal grid, the defocus and astigmatism are read on it."""
    if nominal_frequency_grid_in_angstroms is None:
        if offset_z_in_angstroms is not None:
            defocus_in_angstroms = defocus_in_angstroms + jnp.asarray(
                offset_z_in_angstroms, dtype=float
            )
        nominal_frequency_grid_in_angstroms = frequency_grid_in_angstroms
        offset_z_in_angstroms = None
    q_x, q_y = frequency_grid_in_angstroms[..., 0], frequency_grid_in_angstroms[..., 1]
    q_sqr = q_x**2 + q_y**2
    n_x = nominal_frequency_grid_in_angstroms[..., 0]
    n_y = nominal_frequency_grid_in_angstroms[..., 1]
    n_sqr = n_x**2 + n_y**2
    astigmatic_q_sqr = (n_y**2 - n_x**2) * astigmatism_xy_in_angstroms[0] + (
        2.0 * n_x * n_y
    ) * astigmatism_xy_in_angstroms[1]
    defocused_q_sqr = defocus_in_angstroms * n_sqr + 0.5 * astigmatic_q_sqr
    if offset_z_in_angstroms is not None:
        defocused_q_sqr += jnp.asarray(offset_z_in_angstroms, dtype=float) * q_sqr
    defocus_phase_shifts = -0.5 * wavelength_in_angstroms * defocused_q_sqr
    spherical_phase_shifts = (
        0.25
        * spherical_aberration_in_angstroms
        * (wavelength_in_angstroms**3)
        * (q_sqr**2)
    )
    return (2 * jnp.pi) * (defocus_phase_shifts + spherical_phase_shifts)


def compute_odd_aberration_phase_shifts(
    frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    wavelength_in_angstroms: Float[Array, ""],
    coma_xy_in_angstroms: Float[Array, "2"] | None,
    trefoil_xy_in_angstroms: Float[Array, "2"] | None,
) -> Float[Array, "y_dim x_dim"] | None:
    """The axial coma, `2 pi lambda^2 |q|^2 (coma . H_1)`, and trefoil,
    `(2 pi / 3) lambda^2 (trefoil . H_3)`, phase shifts. `None` if there are neither."""
    if coma_xy_in_angstroms is None and trefoil_xy_in_angstroms is None:
        return None
    q_x, q_y = frequency_grid_in_angstroms[..., 0], frequency_grid_in_angstroms[..., 1]
    phase_shifts = jnp.zeros_like(q_x)
    if coma_xy_in_angstroms is not None:
        coma_q = coma_xy_in_angstroms[0] * q_y + coma_xy_in_angstroms[1] * q_x
        phase_shifts += (
            2 * jnp.pi * wavelength_in_angstroms**2 * (q_x**2 + q_y**2) * coma_q
        )
    if trefoil_xy_in_angstroms is not None:
        trefoil_q = trefoil_xy_in_angstroms[0] * (
            q_y**3 - 3.0 * q_y * q_x**2
        ) + trefoil_xy_in_angstroms[1] * (3.0 * q_y**2 * q_x - q_x**3)
        phase_shifts += (2 * jnp.pi / 3) * wavelength_in_angstroms**2 * trefoil_q
    return phase_shifts


def compute_amplitude_contrast_phase_shift(
    amplitude_contrast_ratio: FloatLike,
) -> Float[Array, ""]:
    """The constant phase shift equivalent to an amplitude contrast ratio."""
    amplitude_contrast_ratio = jnp.asarray(amplitude_contrast_ratio, dtype=float)
    return jnp.arctan(
        amplitude_contrast_ratio / jnp.sqrt(1.0 - amplitude_contrast_ratio**2)
    )
