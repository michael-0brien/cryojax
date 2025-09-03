"""Unit conversions."""

import jax.numpy as jnp
import scipy
from jaxtyping import Array, Float

from ..jax_util import NDArrayLike


def wavelength_from_kilovolts(
    voltage_in_kilovolts: Float[NDArrayLike, ""] | float,
) -> Float[Array, ""]:
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
        12.2639 / (accelerating_voltage + 0.97845e-6 * accelerating_voltage**2) ** 0.5
    )


def lorentz_factor_from_kilovolts(
    voltage_in_kilovolts: Float[NDArrayLike, ""] | float,
) -> Float[Array, ""]:
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
    return 1 + accelerating_energy / rest_energy
