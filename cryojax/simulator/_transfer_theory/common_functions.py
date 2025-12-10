import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import FloatLike
from ...ndimage import cartesian_to_polar


# Not currently public API
def compute_phase_shifts_with_spherical_aberration(
    frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    defocus_in_angstroms: Float[Array, ""],
    astigmatism_in_angstroms: Float[Array, ""],
    astigmatism_angle: Float[Array, ""],
    wavelength_in_angstroms: Float[Array, ""],
    spherical_aberration_in_angstroms: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"]:
    k_sqr, azimuth = cartesian_to_polar(frequency_grid_in_angstroms, square=True)
    defocus = defocus_in_angstroms + 0.5 * astigmatism_in_angstroms * jnp.cos(
        2.0 * (azimuth - astigmatism_angle)
    )
    defocus_phase_shifts = -0.5 * defocus * wavelength_in_angstroms * k_sqr
    aberration_phase_shifts = (
        0.25
        * spherical_aberration_in_angstroms
        * (wavelength_in_angstroms**3)
        * (k_sqr**2)
    )
    phase_shifts = (2 * jnp.pi) * (defocus_phase_shifts + aberration_phase_shifts)

    return phase_shifts


# Not currently public API
def compute_phase_shift_from_amplitude_contrast_ratio(
    amplitude_contrast_ratio: FloatLike,
) -> Float[Array, ""]:
    amplitude_contrast_ratio = jnp.asarray(amplitude_contrast_ratio, dtype=float)
    return jnp.arctan(
        amplitude_contrast_ratio / jnp.sqrt(1.0 - amplitude_contrast_ratio**2)
    )
