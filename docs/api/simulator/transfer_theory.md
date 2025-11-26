# Contrast transfer functions

Applying a contrast transfer function (CTF) to an image in `cryojax` is layered into two classes. The most important class is the `AstigmaticCTF`, which is a function that takes in a grid of in-plane frequency vectors and returns a JAX array of the CTF.

```python
import cryojax.simulator as cxs
from cryojax.ndimage import make_frequency_grid

shape, pixel_size = (100, 100), 1.1
frequency_grid_in_angstroms = make_frequency_grid(shape, pixel_size)
ctf = cxs.AstigmaticCTF(
    defocus_in_angstroms=10000.0,
    astigmatism_in_angstroms=100.0,
    astigmatism_angle=30.0,
    spherical_aberration_in_mm=2.7,
    amplitude_contrast_ratio=0.07,
    phase_shift=0.0,
)
ctf_array = ctf(frequency_grid_in_angstroms, voltage_in_kilovolts=300.0)
```

Further, the `ContrastTransferTheory` is a class that takes in a projection image in a plane below the object and returns the contrast in the plane of the detector.

```python
projection_image_in_fourier_domain = ...
ctf = AstigmaticCTF(...)
transfer_theory = cxs.ContrastTransferTheory(ctf)
contrast_in_fourier_domain = transfer_theory.propagate_object(projection_image_in_fourier_domain)
```

This documentation describes the elements of transfer theory in `cryojax`. More features are included than described above, such as the ability to include a user-specified envelope function to the `ContrastTransferTheory`. Much of the code and theory have been adapted from the Grigorieff Lab's CTFFIND4 program.

*References*

- Rohou, Alexis, and Nikolaus Grigorieff. "CTFFIND4: Fast and accurate defocus estimation from electron micrographs." Journal of structural biology 192.2 (2015): 216-221.

## Transfer Functions

???+ abstract "`cryojax.simulator.AbstractCTF`"
    ::: cryojax.simulator.AbstractCTF
        options:
            members:
                - compute_aberration_phase_shifts

::: cryojax.simulator.AstigmaticCTF
        options:
            members:
                - __init__
                - compute_aberration_phase_shifts
                - __call__

## Transfer Theories

::: cryojax.simulator.ContrastTransferTheory
        options:
            members:
                - __init__
                - propagate_object
