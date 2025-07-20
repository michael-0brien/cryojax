<h1 align='center'>cryoJAX</h1>

![Tests](https://github.com/mjo22/cryojax/actions/workflows/testing.yml/badge.svg)
![Lint](https://github.com/mjo22/cryojax/actions/workflows/ruff.yml/badge.svg)

## Summary

CryoJAX is a library that simulates cryo-electron microscopy (cryo-EM) images in [JAX](https://jax.readthedocs.io/en/latest/). Its purpose is to provide the tools for building downstream data analysis in external workflows and libraries that leverage the statistical inference and machine learning resources of the JAX scientific computing ecosystem. To achieve this, image simulation in cryoJAX is built for reliability and flexibility: it implements a variety of established models and algorithms as well as a framework for implementing new models and algorithms downstream. If your application uses cryo-EM image simulation and it cannot be built downstream, open a [Pull Request](https://github.com/mjo22/cryojax/pulls).

## Documentation

See the documentation at [https://mjo22.github.io/cryojax/](https://mjo22.github.io/cryojax/). It is a work-in-progress, so thank you for your patience!

## Installation

Installing `cryojax` is simple. To start, I recommend creating a new virtual environment. For example, you could do this with `conda`.

```bash
conda create -n cryojax-env -c conda-forge python=3.11
```

Note that `python>=3.10` is required. After creating a new environment, [install JAX](https://github.com/google/jax#installation) with either CPU or GPU support. Then, install `cryojax`. For the latest stable release, install using `pip`.

```bash
python -m pip install cryojax
```

To install the latest commit, you can build the repository directly.

```bash
git clone https://github.com/mjo22/cryojax
cd cryojax
python -m pip install .
```

The [`jax-finufft`](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. These are included as an option for computing image projections. In this case, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Simulating an image

The following is a basic workflow to simulate an image.

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cxs
from cryojax.io import read_array_with_spacing_from_mrc

# Instantiate the scattering potential from a voxel grid. See the documentation
# for how to generate voxel grids from a PDB
filename = "example_scattering_potential.mrc"
real_voxel_grid, voxel_size = read_array_from_mrc(filename, loads_spacing=True)
potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, voxel_size)
# Now, the pose. Angles are given in degrees.
pose = cxs.EulerAnglePose(
    offset_x_in_angstroms=5.0,
    offset_y_in_angstroms=-3.0,
    phi_angle=20.0,
    theta_angle=80.0,
    psi_angle=-10.0,
)
# Next the model for the CTF
ctf = cxs.CTF(
    defocus_in_angstroms=9800.0, astigmatism_in_angstroms=200.0, astigmatism_angle=10.0
)
transfer_theory = cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.1)
# Finally, create the configuration and build the image model
config = cxs.BasicConfig(shape=(320, 320), pixel_size=voxel_size, voltage_in_kilovolts=300.0)
# Instantiate a cryoJAX `image_model` using the `make_image_model` function
image_model, simulate_fn = make_image_model(
    potential, config, pose, transfer_theory, outputs_real_space=True
)
# Simulate an image
image = simulate_fn(image_model)
```

For more advanced image simulation examples and to understand the many features in this library, see the [documentation](https://mjo22.github.io/cryojax/).

## JAX transformations


## Acknowledgements

- `cryojax` implementations of several models and algorithms, such as the CTF, fourier slice extraction, and electrostatic potential computations has been informed by the open-source cryo-EM software [`cisTEM`](https://github.com/timothygrant80/cisTEM).
- `cryojax` is built on [`equinox`](https://github.com/patrick-kidger/equinox/), a popular JAX library for PyTorch-like classes that smoothly integrate with JAX functional programming. We highly recommend learning about `equinox` to fully make use of the power of `jax`.
