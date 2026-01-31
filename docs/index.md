# Welcome to cryoJAX!

CryoJAX is a library that simulates cryo-electron microscopy (cryo-EM) images in [JAX](https://jax.readthedocs.io/en/latest/). Its purpose is to provide the tools for building downstream data analysis in external workflows and libraries that leverage the statistical inference and machine learning resources of the JAX scientific computing ecosystem. To achieve this, image simulation in cryoJAX is built for reliability and flexibility: it implements a variety of established models and algorithms as well as a framework for implementing new models and algorithms downstream. If your application uses cryo-EM image simulation and it cannot be built downstream, open a [pull request](https://github.com/michael-0brien/cryojax/pulls).

This documentation is currently a work-in-progress. Your patience while we get this project properly documented is much appreciated! Feel free to get in touch on github [issues](https://github.com/michael-0brien/cryojax/issues) if you have any questions, bug reports, or feature requests.

## Installation

Installing `cryojax` is simple. To start, I recommend creating a new virtual environment. For example, you could do this with [`uv`](https://docs.astral.sh/uv/pip/environments/#creating-a-virtual-environment).

```bash
uv venv --python=3.11 ~/path/to/venv/
source ~/path/to/venv/bin/activate
```

Note that `python>=3.10` is required. After creating and activating the new environment, [install JAX](https://github.com/google/jax#installation) with either CPU or GPU support. Then, install `cryojax`. For the latest stable release, install using `pip`.

```bash
uv pip install cryojax
```

To install the latest commit, you can build the repository directly.

```bash
git clone https://github.com/michael-0brien/cryojax
cd cryojax
git checkout dev
uv pip install .
```

The [`jax-finufft`](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. This is used in select methods for computing image projections from atoms and voxels. If you would like to use these methods, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Quick example

The following is a basic example for simulating an image:

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cxs

# Instantiate a cryoJAX `image_model`
image_model = cxs.make_image_model(
    # ... load atoms as gaussians mixture from tabulated electron scattering factors
    volume=cxs.load_tabulated_volume(
        "example.pdb", output_type=cxs.GaussianMixtureVolume
    ),
    # ... configure the image
    image_config=cxs.BasicImageConfig(shape=(320, 320), pixel_size=1., voltage_in_kilovolts=300),
    # ... the pose
    pose=cxs.EulerAnglePose(phi_angle=20., theta_angle=80., psi_angle=-10.),
    # ... the CTF
    transfer_theory=cxs.ContrastTransferTheory(
        ctf=cxs.AstigmaticCTF(defocus_in_angstroms=9800., astigmatism_in_angstroms=200., astigmatism_angle=10.),
        amplitude_contrast_ratio=0.1,
    ),
)
# Simulate an image
image = image_model.simulate(outputs_real_space=True)
```

## What next?

Keep reading further for an [overview](https://michael-0brien.github.io/cryojax/overview/) of the library. Then, get started learning about using cryoJAX in practice with the [image simulation](https://michael-0brien.github.io/cryojax/examples/simulate-image/) and [JAX transformation](https://michael-0brien.github.io/cryojax/examples/jax-transformations/) tutorials, as well as the [API documentation](https://michael-0brien.github.io/cryojax/api/simulator/entry-point/)!
