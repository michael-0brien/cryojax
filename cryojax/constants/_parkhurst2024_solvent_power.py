from collections import namedtuple


Parkhurst2024PowerConstants = namedtuple(
    "Parkhurst2024PowerConstants", ["a_1", "a_2", "s_1", "s_2", "m"]
)

PARKHURST2024_POWER_CONSTANTS = Parkhurst2024PowerConstants(
    a_1=0.199, a_2=0.801, s_1=0.731, s_2=0.081, m=1 / 2.88
)
"""Constants for used to define the power envelope from Parkhurst et al. (2024)

**References:**

- Parkhurst, James M., et al. "Computational models of amorphous ice for accurate
simulation of cryo-EM images of biological samples." Ultramicroscopy 256 (2024): 113882.
"""
