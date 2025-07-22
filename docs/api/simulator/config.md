# Configuring an image

The `AbstractConfig` is an object at the core of simulating images in `cryojax`. It stores a configuration for the simulated image and the electron microscope, such as the shape of the desired image and the wavelength of the incident electron beam.

::: cryojax.simulator.AbstractConfig
        options:
            members:
                - __init__
                - wavelength_in_angstroms
                - wavenumber_in_inverse_angstroms
                - n_pixels
                - y_dim
                - x_dim
                - coordinate_grid_in_pixels
                - coordinate_grid_in_angstroms
                - frequency_grid_in_pixels
                - frequency_grid_in_angstroms
                - full_frequency_grid_in_pixels
                - full_frequency_grid_in_angstroms
                - padded_n_pixels
                - padded_y_dim
                - padded_x_dim
                - padded_coordinate_grid_in_pixels
                - padded_coordinate_grid_in_angstroms
                - padded_frequency_grid_in_pixels
                - padded_frequency_grid_in_angstroms
                - padded_full_frequency_grid_in_pixels
                - padded_full_frequency_grid_in_angstroms
                - crop_to_shape
                - pad_to_padded_shape
                - crop_or_pad_to_padded_shape
