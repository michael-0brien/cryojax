# Image and volume manipulation

`cryojax.ndimage` implements routines for image and volume arrays, such coordinate creation, downsampling, filters, and masks. This is a key submodule for supporting `cryojax.simulator`.

## Coordinate systems

This documentation is a collection of functions used to work with coordinate systems in `cryojax`'s conventions. The most important functions are `make_coordinate_grid` and `make_frequency_grid`.

### Creating coordinate systems

::: cryojax.ndimage.make_coordinate_grid

---

::: cryojax.ndimage.make_frequency_grid

---

::: cryojax.ndimage.make_radial_coordinate_grid

---

::: cryojax.ndimage.make_radial_frequency_grid

---

::: cryojax.ndimage.make_frequency_slice

---

::: cryojax.ndimage.make_1d_coordinate_grid

---

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


::: cryojax.ndimage.ImageScaling
        options:
            members:
                - __init__
                - __call__


### Filters

??? abstract "`cryojax.ndimage.AbstractFilter`"
    ::: cryojax.ndimage.AbstractFilter
        options:
            members:
                - get


::: cryojax.ndimage.LowpassFilter
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.HighpassFilter
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.WhiteningFilter
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.CustomFilter
        options:
            members:
                - __init__
                - get
                - __call__


### Masks

??? abstract "`cryojax.ndimage.AbstractMask`"
    ::: cryojax.ndimage.AbstractMask
        options:
            members:
                - get

::: cryojax.ndimage.CircularCosineMask
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.SphericalCosineMask
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.SquareCosineMask
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.Rectangular2DCosineMask
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.Rectangular3DCosineMask
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.Cylindrical2DCosineMask
        options:
            members:
                - __init__
                - get
                - __call__

---

::: cryojax.ndimage.CustomMask
        options:
            members:
                - __init__
                - get
                - __call__

::: cryojax.ndimage.SincCorrectionMask
        options:
            members:
                - __init__
                - get
                - __call__



## Operators

### Fourier-space

???+ abstract "`cryojax.ndimage.AbstractFourierOperator`"
    ::: cryojax.ndimage.AbstractFourierOperator
        options:
            members:
                - __call__


::: cryojax.ndimage.FourierGaussian
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.ndimage.PeakedFourierGaussian
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.ndimage.FourierConstant
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.ndimage.FourierSinc
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.ndimage.FourierExp2D
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.ndimage.FourierDC
        options:
            members:
                - __init__
                - __call__


---

::: cryojax.ndimage.CustomFourierOperator
        options:
            members:
                - __init__
                - __call__


### Real-space

???+ abstract "`cryojax.ndimage.AbstractRealOperator`"
    ::: cryojax.ndimage.AbstractRealOperator
        options:
            members:
                - __call__

::: cryojax.ndimage.RealGaussian
        options:
            members:
                - __init__
                - __call__

---

::: cryojax.ndimage.RealConstant
        options:
            members:
                - __init__
                - __call__

---


## Utility functions
