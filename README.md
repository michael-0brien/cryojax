<h1 align='center'>cryoJAX</h1>

[![Continuous Integration](https://github.com/michael-0brien/cryojax/actions/workflows/ci_build.yml/badge.svg)](https://github.com/michael-0brien/cryojax/actions/workflows/ci_build.yml?branch=dev)
[![codecov](https://codecov.io/gh/michael-0brien/cryojax/branch/dev/graph/badge.svg)](https://codecov.io/gh/michael-0brien/cryojax)


## Summary

CryoJAX is a library that simulates cryo-electron microscopy (cryo-EM) images in [JAX](https://jax.readthedocs.io/en/latest/). Its purpose is to provide the tools for building downstream data analysis in external workflows and libraries that leverage the statistical inference and machine learning resources of the JAX scientific computing ecosystem. To achieve this, image simulation in cryoJAX is built for reliability and flexibility; it implements a variety of established models and algorithms as well as a framework for implementing new models and algorithms downstream. If your application uses cryo-EM image simulation and it cannot be built downstream, open a [pull request](https://github.com/michael-0brien/cryojax/pulls).

## Documentation

See the documentation at [https://michael-0brien.github.io/cryojax/](https://michael-0brien.github.io/cryojax/). It is a work-in-progress, so thank you for your patience!

## Installation

Installing `cryojax` is simple. To start, I recommend creating a new virtual environment. For example, you could do this with `conda`.

```bash
conda create -n cryojax-env -c conda-forge python=3.11
```

Note that `python>=3.10` is required. After creating and activating the new environment, [install JAX](https://github.com/google/jax#installation) with either CPU or GPU support. Then, install `cryojax`. For the latest stable release, install using `pip`.

```bash
python -m pip install cryojax
```

To install the latest commit, you can build the repository directly.

```bash
git clone https://github.com/michael-0brien/cryojax
cd cryojax
git checkout dev
python -m pip install .
```

The [`jax-finufft`](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. This is used in select methods for computing image projections from atoms and voxels. If you would like to use these methods, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Quick example

Image simulation in cryoJAX revolves around the `image_model` class. The following is a basic example for instantiating an `image_model` and simulating an image:

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cxs

# Instantiate a cryoJAX `image_model`
image_model = cxs.make_image_model(
    # ... load atoms as gaussians mixture from tabulated electron scattering factors
    volume_parametrization=cxs.load_tabulated_volume(
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

For more advanced image simulation examples and to understand the many features in this library, see the [documentation](https://michael-0brien.github.io/cryojax/).

## JAX transformations

CryoJAX is built on JAX to make use of JIT-compilation, automatic differentiation, and vectorization for cryo-EM data analysis. JAX implements these operations as *function transformations*. If you aren't familiar with this concept, see the [JAX documentation](https://docs.jax.dev/en/latest/key-concepts.html#transformations).

Below are examples of implementing these transformations using [`equinox`](https://docs.kidger.site/equinox/), a popular JAX library for PyTorch-like classes that smoothly integrate with JAX functional programming. To learn more about how `equinox` assists with JAX transformations, see [here](https://docs.kidger.site/equinox/all-of-equinox/#2-filtering).

### Your first JIT compiled function

```python
import equinox as eqx

# Define image simulation function using `equinox.filter_jit`
@eqx.filter_jit
def simulate_fn(image_model):
    """Simulate an image with JIT compilation"""
    return image_model.simulate()

# Simulate an image
image = simulate_fn(image_model)
```

### Computing gradients of a loss function

```python
import equinox as eqx
import jax
import jax.numpy as jnp

# Load observed data
observed_image = ...

# Split the `image_model` by differentiated and non-differentiated
# arguments. Here, differentiate with respect to the pose.
is_pose = lambda x: isinstance(x, cxs.AbstractPose)
filter_spec = jax.tree.map(is_pose, image_model, is_leaf=is_pose)
model_grad, model_nograd = eqx.partition(image_model, filter_spec)

@eqx.filter_value_and_grad
def loss_fn(model_grad, model_nograd, observed_image):
    """Compute gradients with respect to the pose."""
    image_model = eqx.combine(model_grad, model_nograd)
    return jnp.sum((image_model.simulate() - observed_image)**2)

# Compute the loss and gradients
loss, gradients = loss_fn(model_grad, model_nograd, observed_image)
```

### Vectorizing image simulation

```python
import equinox as eqx

# Vectorize model instantiation over poses
@eqx.filter_vmap(in_axes=(0, None, None, None), out_axes=(eqx.if_array(0), None))
def make_model_vmap(wxyz, volume, image_config, transfer_theory):
    pose = cxs.QuaternionPose(wxyz=wxyz)
    image_model = cxs.make_image_model(
        volume, image_config, pose, transfer_theory, normalizes_signal=True
    )
    is_pose = lambda x: isinstance(x, cxs.AbstractPose)
    filter_spec = jax.tree.map(is_pose, image_model, is_leaf=is_pose)
    model_vmap, model_novmap = eqx.partition(image_model, filter_spec)

    return model_vmap, model_novmap


# Define image simulation function with respect to vectorized arguments
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def simulate_fn_vmap(model_vmap, model_novmap):
    image_model = eqx.combine(model_vmap, model_novmap)
    return image_model.simulate()

# Batch image simulation over poses
wxyz = ...  # ... load quaternions
model_vmap, model_novmap = make_model_vmap(wxyz, volume, image_config, transfer_theory)
images = simulate_fn_vmap(model_vmap, model_novmap)
```

## Acknowledgements

- `cryojax` implementations of several models and algorithms, such as the CTF, fourier slice extraction, and electrostatic potential computations has been informed by the open-source cryo-EM software [`cisTEM`](https://github.com/timothygrant80/cisTEM).
- `cryojax` is built using `equinox`, a popular JAX library for PyTorch-like classes that smoothly integrate with JAX functional programming. We highly recommend learning about `equinox` to fully make use of the power of `jax`.
