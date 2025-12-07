# Image and volume manipulation

`cryojax.ndimage` implements routines for image and volume arrays, such coordinate creation, downsampling, filters, and masks. This is a key submodule for supporting `cryojax.simulator`.

## Coordinate systems

This documentation is a collection of functions used to work with coordinate systems in `cryojax`'s conventions. The most important functions are `make_coordinate_grid` and `make_frequency_grid`.

### Creating coordinate systems

::: cryojax.ndimage.make_coordinate_grid

::: cryojax.ndimage.make_frequency_grid

::: cryojax.ndimage.make_frequency_slice

::: cryojax.ndimage.make_1d_coordinate_grid

::: cryojax.ndimage.make_1d_frequency_grid


### Transforming coordinate systems

`cryojax` also provides functions that transform between coordinate conventions.

::: cryojax.ndimage.cartesian_to_polar


## Image transforms (e.g. filters and masks)

??? abstract "`cryojax.ndimage.AbstractImageTransform`"
    ::: cryojax.ndimage.AbstractImageTransform
        options:
            members:
                - __init__
                - is_real_space
                - __call__


### Filters

??? abstract "`cryojax.ndimage.AbstractFilter`"
    ::: cryojax.ndimage.AbstractFilter
        options:
            members:
                - get


### Masks

??? abstract "`cryojax.ndimage.AbstractMask`"
    ::: cryojax.ndimage.AbstractMask
        options:
            members:
                - get

::: cryojax.ndimage.InverseSincMask
        options:
            members:
                - __init__
                - get
                - __call__



## Operators

??? abstract "`cryojax.ndimage.AbstractImageOperator`"
    ::: cryojax.ndimage.AbstractImageOperator
        options:
            members:
                - __call__

### Real-space

???+ abstract "`cryojax.ndimage.AbstractRealOperator`"
    ::: cryojax.ndimage.AbstractRealOperator
        options:
            members:
                - __call__

### Fourier-space

???+ abstract "`cryojax.ndimage.AbstractFourierOperator`"
    ::: cryojax.ndimage.AbstractFourierOperator
        options:
            members:
                - __call__



## Utility functions
