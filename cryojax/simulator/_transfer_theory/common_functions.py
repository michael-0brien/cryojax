import jax.numpy as jnp
from jaxtyping import Array, Float

from ...jax_util import FloatLike


# Not currently public API
def compute_phase_shifts_with_spherical_aberration(
    frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
    defocus_in_angstroms: Float[Array, ""],
    astigmatism_in_angstroms: Float[Array, ""],
    astigmatism_angle: Float[Array, ""],
    wavelength_in_angstroms: Float[Array, ""],
    spherical_aberration_in_angstroms: Float[Array, ""],
) -> Float[Array, "y_dim x_dim"]:
    # As in CTFFIND4, the defocus is astigmatic: `defocus + astigmatism cos(2 (azimuth
    # - angle)) / 2`, where `azimuth = arctan2(k_x, k_y)` is the direction of the spatial
    # frequency. Because this defocus only ever multiplies `k^2`, the azimuth is never
    # needed explicitly: the product is a polynomial in `(k_x, k_y)` whose derivatives
    # (e.g. in the pixel size) stay finite at the origin, where `arctan2` is not
    # differentiable.
    k_x, k_y = frequency_grid_in_angstroms[..., 0], frequency_grid_in_angstroms[..., 1]
    k_sqr = k_x**2 + k_y**2
    # `cos(2 (azimuth - angle)) k^2`, expanded with `cos(2 azimuth) k^2 = k_y^2 - k_x^2`
    # and `sin(2 azimuth) k^2 = 2 k_x k_y`.
    astigmatic_k_sqr = (k_y**2 - k_x**2) * jnp.cos(2.0 * astigmatism_angle) + (
        2.0 * k_x * k_y
    ) * jnp.sin(2.0 * astigmatism_angle)
    defocus_phase_shifts = (
        -0.5
        * wavelength_in_angstroms
        * (
            defocus_in_angstroms * k_sqr
            + 0.5 * astigmatism_in_angstroms * astigmatic_k_sqr
        )
    )
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
