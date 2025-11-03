"""Unit conversions."""

import jax.numpy as jnp
import scipy
from jaxtyping import Array, Float

from ..jax_util import FloatLike


def wavelength_from_kilovolts(voltage_in_kilovolts: FloatLike) -> Float[Array, ""]:
    """Get the relativistic electron wavelength at a given accelerating voltage. For
    reference, see Equation 2.5 in Section 2.1 from *Spence, John CH. High-resolution
    electron microscopy. OUP Oxford, 2013.*.

    **Arguments:**

    - `voltage_in_kilovolts`:
        The accelerating voltage given in kilovolts.

    **Returns:**

    The relativistically corrected electron wavelength in Angstroms corresponding to the
    energy `energy_in_keV`.
    """
    accelerating_voltage = 1000.0 * voltage_in_kilovolts  # keV to eV
    return jnp.asarray(
        12.2639 / (accelerating_voltage + 0.97845e-6 * accelerating_voltage**2) ** 0.5,
        dtype=float,
    )


def lorentz_factor_from_kilovolts(voltage_in_kilovolts: FloatLike) -> Float[Array, ""]:
    """Get the Lorentz factor given an accelerating voltage.

    **Arguments:**

    - `voltage_in_kilovolts`:
        The accelerating voltage given in kilovolts.

    **Returns:**

    The Lorentz factor.
    """
    c = scipy.constants.speed_of_light
    m0 = scipy.constants.electron_mass
    e = scipy.constants.elementary_charge
    rest_energy = m0 * c**2
    accelerating_energy = e * (1000.0 * voltage_in_kilovolts)
    return jnp.asarray(1 + accelerating_energy / rest_energy, dtype=float)


def interaction_constant_from_kilovolts(
    voltage_in_kilovolts: FloatLike,
) -> Float[Array, ""]:
    """Get the electron interaction constant given an accelerating voltage.

    The interaction constant is necessary to compute the object
    phase shift distribution from an electrostatic potential integrated
    on the plane.

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

        $$\\sigma_e = \\lambda \\gamma,$$

        where $\\lambda$ the relativistic electron wavelength $\\gamma$ is
        the lorentz factor.

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

    **Arguments:**

    - `voltage_in_kilovolts`:
        The accelerating voltage given in kilovolts.

    **Returns:**

    The electron interaction constant.
    """
    wavelength = wavelength_from_kilovolts(voltage_in_kilovolts)
    lorentz_factor = lorentz_factor_from_kilovolts(voltage_in_kilovolts)
    return wavelength * lorentz_factor
