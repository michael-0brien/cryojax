# Rotations in cryo-EM

`cryojax.rotations` implements an engine and utility functions for rotations in cryo-EM.

## Representing rotations

The engine for handling rotations in cryoJAX is the `cryojax.rotations.SO3` class. This is based on the implementation in the package [`jaxlie`](https://github.com/brentyi/jaxlie).

::: cryojax.rotations.SO3
        options:
            members:
                - __init__
                - apply
                - compose
                - inverse
                - from_x_radians
                - from_y_radians
                - from_z_radians
                - identity
                - from_matrix
                - as_matrix
                - exp
                - log
                - adjoint
                - normalize
                - sample_uniform


## Utilities for converting between angular parametrizations
