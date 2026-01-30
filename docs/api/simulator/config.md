# Configuring an image

The `AbstractImageConfig` is an object at the core of simulating images in `cryojax`. It stores a configuration for the simulated image and the electron microscope, such as the shape of the desired image and the wavelength of the incident electron beam.

??? abstract "`cryojax.simulator.AbstractImageConfig`"
    ::: cryojax.simulator.AbstractImageConfig
            members:
                - shape
                - pixel_size
                - voltage_in_kilovolts
                - padded_shape

---

::: cryojax.simulator.BasicImageConfig
        options:
            members:
                - __init__
                - wavelength_in_angstroms
                - lorentz_factor
                - interaction_constant
                - get_coordinate_grid
                - get_frequency_grid
                - n_pixels
                - y_dim
                - x_dim
                - padded_n_pixels
                - padded_y_dim
                - padded_x_dim

---

::: cryojax.simulator.DoseImageConfig
        options:
            members:
                - __init__
                - wavelength_in_angstroms
                - lorentz_factor
                - interaction_constant
                - get_coordinate_grid
                - get_frequency_grid
                - n_pixels
                - y_dim
                - x_dim
                - padded_n_pixels
                - padded_y_dim
                - padded_x_dim
