# Physical constants

`cryojax.constants` stores and supports physical constants used when simulating cryo-EM images.

## Scattering factor parameters

Modeling the electron scattering amplitudes of individual atoms is an important component of modeling cryo-EM images, as these are typically used to approximate the electrostatic potential. Typically, the scattering factor for each individual atom is numerically approximated with a fixed functional form but varying parameters for different atoms. These parameters are stored in lookup tables in the literature. This documentation provides these lookup tables and utilities for extracting them so that they may be used to compute electrostatic potentials in cryoJAX.

::: cryojax.constants.extract_scattering_factor_parameters

---

::: cryojax.constants.check_atomic_numbers_supported

### Peng scattering factor parameters

::: cryojax.constants.PengScatteringFactorParameters
    options:
        members:
            - __init__
            - a
            - b

---


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
