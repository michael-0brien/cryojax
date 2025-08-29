import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ..jax_util import NDArrayLike


def apply_interaction_constant(
    integrated_potential: Inexact[Array, "y_dim x_dim"],
    wavelength_in_angstroms: Float[NDArrayLike, ""] | float,
    lorenz_factor: Float[NDArrayLike, ""] | float,
) -> Inexact[Array, "y_dim x_dim"]:
    """Given an integrated potential, convert units to the object
    phase shift distribution using the interaction
    constant. For example, the case of the projection approximation,
    compute the phase shift distribution.

    !!! info
        In the projection approximation in cryo-EM, the phase shifts in the
        exit plane are given by

        $$\\eta(x, y) = \\sigma_e \\int dz \\ V(x, y, z),$$

        where $\\sigma_e$ is typically referred to as the interaction
        constant. However, in `cryojax`, the potential is rescaled
        to units of inverse length squared as

        $$U(x, y, z) = \\frac{2 m_0 e}{\\hbar^2} V(x, y, z).$$

        With this rescaling of the potential, the defined as with the
        equation

        $$\\eta(x, y) = \\sigma_e \\int dz \\ U(x, y, z)$$

        with

        $$\\sigma_e = \\frac{\\lambda \\gamma}{4 \\pi},$$

        where $\\lambda$ the relativistic electron wavelength $\\gamma$ is
        the lorenz factor.

        **References**:

        - For the definition of the rescaled potential, see
        Chapter 69, Page 2003, Equation 69.6 from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*
        - For the definition of the phase shifts in terms of the rescaled potential, see
        Chapter 69, Page 2012, Equation 69.34b from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*

    See the documentation on atom-based scattering potentials for more information.
    """  # noqa: E501
    interaction_constant = jnp.asarray(wavelength_in_angstroms * lorenz_factor) / (
        4 * jnp.pi
    )
    return interaction_constant * integrated_potential


def apply_amplitude_contrast_ratio(
    integrated_potential: Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"],
    amplitude_contrast_ratio: Float[Array, ""] | float,
) -> Complex[Array, "y_dim x_dim"]:
    """Apply the amplitude contrast ratio to compute a complex scattering potential,
    given the integrated potential computed just with a weak-phase calculation.

    !!! info
        Given a projected electrostatic potential $\\u(x, y)$, the complex potential
        $\\phi(x, y)$ for amplitude contrast ratio $\\alpha$ is

        $$\\phi(x, y) = \\sqrt{1 - \\alpha^2} \\ u(x, y) + i \\alpha \\ u(x, y)$$
    """
    ac = amplitude_contrast_ratio
    if jnp.iscomplexobj(integrated_potential):
        return jnp.sqrt(1.0 - ac**2) * integrated_potential.real + 1.0j * (
            integrated_potential.imag + ac * integrated_potential.real
        )
    else:
        return (jnp.sqrt(1.0 - ac**2) + 1.0j * ac) * integrated_potential
