import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact


def apply_interaction_constant(
    integrated_potential: Inexact[Array, "y_dim x_dim"],
    wavelength_in_angstroms: Float[Array, ""] | float,
) -> Inexact[Array, "y_dim x_dim"]:
    """Given an integrated potential, convert units to the object
    phase shift distribution using the interaction
    constant. For example, the case of the projection approximation,
    compute the phase shift distribution.

    !!! info
        In the projection approximation in cryo-EM, the phase shifts in the
        exit plane are given by

        $$\\eta(x, y) = \\sigma \\int dz \\ V(x, y, z),$$

        where $\\sigma$ is typically referred to as the interaction
        constant. However, in `cryojax`, the potential is rescaled
        to units of inverse length squared as

        $$U(x, y, z) = \\frac{2 m e}{\\hbar^2} V(x, y, z).$$

        With this rescaling of the potential, the phase shifts are equal to

        $$\\eta(x, y) = \\frac{\\lambda}{4 \\pi} \\int dz \\ U(x, y, z).$$

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
    return integrated_potential * jnp.asarray(wavelength_in_angstroms) / (4 * jnp.pi)


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
        raise NotImplementedError(
            "You may have tried to use a `HighEnergyScatteringTheory` "
            "together with `EwaldSphereExtraction` for simulating images. "
            "This is not implemented!"
        )
        # return jnp.sqrt(1.0 - ac**2) * integrated_potential.real + 1.0j * (
        #     integrated_potential.imag + ac * integrated_potential.real
        # )
    else:
        return (jnp.sqrt(1.0 - ac**2) + 1.0j * ac) * integrated_potential
