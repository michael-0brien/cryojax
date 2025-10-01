# Scattering factor parameters

Modeling the electron scattering amplitudes of individual atoms is an important component of modeling cryo-EM images, as these are typically used to approximate the electrostatic potential. Typically, the scattering factor for each individual atom is numerically approximated with a fixed functional form but varying parameters for different atoms. These parameters are stored in lookup tables in the literature. This documentation provides these lookup tables so that they may be used to compute electrostatic potentials in cryoJAX.

## Extracting parameters from a lookup table

::: cryojax.constants.get_tabulated_scattering_factor_parameters

## Lookup tables

::: cryojax.constants.read_peng_scattering_factor_parameter_table
