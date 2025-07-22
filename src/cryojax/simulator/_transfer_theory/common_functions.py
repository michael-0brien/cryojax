import jax.numpy as jnp
from jaxtyping import Array, Float

from ...coordinates import cartesian_to_polar


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
    amplitude_contrast_ratio: float | Float[Array, ""],
) -> Float[Array, ""]:
    amplitude_contrast_ratio = jnp.asarray(amplitude_contrast_ratio)
    return jnp.arctan(
        amplitude_contrast_ratio / jnp.sqrt(1.0 - amplitude_contrast_ratio**2)
    )

def compute_phase_shift_from_zernike_basis (
    frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    wavelength_in_angstroms: Float[Array, ""],

    coefficients_matrix: Float[Array,""],
    ):

    N, M = coefficients_matrix.shape
    N = jnp.arange(N)
    M = jnp.arange(M)


    k, azimuth = cartesian_to_polar (frequency_grid_in_angstroms, square = False)

    k_powers = (k*wavelength_in_angstroms)**(n + 1) / (n + 1) 

    sum_weights = k_powers[:, None]  # shape (N, 1)

    cos_mphi = jnp.cos(m * azimuth)             
    sin_mphi = jnp.sin(m * azimuth)             

    # Compute the sum with einsum
    term_cos = jnp.einsum('nm,n,m->', coefficients_matrix[:, :, 0], sum_weights[:, 0], cos_mphi)
    term_sin = jnp.einsum('nm,n,m->', coefficients_matrix[:, :, 1], sum_weights[:, 0], sin_mphi)

    phase_shifts = (2 * jnp.pi / wavelength_in_angstroms) * (term_cos + term_sin)

    return phase_shifts
