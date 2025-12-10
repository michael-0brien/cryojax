# Physical constants

`cryojax.constants` stores and supports physical constants used when simulating cryo-EM images.

## Scattering factor parameters

Modeling the electron scattering amplitudes of individual atoms is an important component of modeling cryo-EM images, as these are typically used to approximate the electrostatic potential. Typically, the scattering factor for each individual atom is numerically approximated with a fixed functional form but varying parameters for different atoms. These parameters are stored in lookup tables in the literature. This documentation provides these lookup tables and utilities for extracting them so that they may be used to compute electrostatic potentials in cryoJAX.

::: cryojax.constants.PengScatteringFactorParameters
    options:
        members:
            - __init__
            - a
            - b

---

::: cryojax.constants.LobatoScatteringFactorParameters
    options:
        members:
            - __init__
            - a
            - b

---

!!! warning

    Only electron scattering factors for elements found in PDB files (e.g. proteins, DNA/RNA,
    small molecules) are supported when instantiating [`cryojax.constants.PengScatteringFactorParameters`][] or
    [`cryojax.constants.LobatoScatteringFactorParameters`][]. These are the following `atomic_numbers`:

        - 1: Hydrogen
        - 6: Carbon
        - 7: Nitrogen
        - 8: Oxygen
        - 9: Fluorine
        - 11: Sodium
        - 12: Magnesium
        - 15: Phosphorus
        - 16: Sulfur
        - 17: Chlorine
        - 19: Potassium
        - 20: Calcium
        - 25: Manganese
        - 26: Iron
        - 27: Colbalt
        - 29: Copper
        - 30: Zinc

    If `atomic_numbers` contains values not in this list, scattering
    factor parameters returned yield `numpy.nan` values or an
    index out of bounds error. To check if
    `atomic_numbers` is a valid array, use
    [`cryojax.constants.check_atomic_numbers_supported`][].

---

::: cryojax.constants.check_atomic_numbers_supported


## Physical units

Here, convenience methods for working with physical units are described.

::: cryojax.constants.wavelength_from_kilovolts

---

::: cryojax.constants.lorentz_factor_from_kilovolts

---

::: cryojax.constants.interaction_constant_from_kilovolts



## Converting between common conventions

Here, helper functions for converting between common conventions are described.

::: cryojax.constants.b_factor_to_variance

---

::: cryojax.constants.variance_to_b_factor
