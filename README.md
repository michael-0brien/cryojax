<h1 align='center'>cryoJAX</h1>

![Tests](https://github.com/mjo22/cryojax/actions/workflows/testing.yml/badge.svg)
![Lint](https://github.com/mjo22/cryojax/actions/workflows/ruff.yml/badge.svg)

cryoJAX is a library that provides tools for simulating and analyzing cryo-electron microscopy (cryo-EM) images. It is built on [`jax`](https://jax.readthedocs.io/en/latest/).

## Summary

Specifically, cryoJAX aims to provide three things in the cryo-EM image-to-structure pipeline.

1. *Physical modeling of image formation*
2. *Statistical modeling of the distributions from which images are drawn*
3. *Easy-to-use utilities for working with real data*

With these tools, `cryojax` aims to appeal to two different communities. Experimentalists can use `cryojax` in order to push the boundaries of what they can extract from their data by interfacing with the `jax` scientific computing ecosystem. Additionally, method developers may use `cryojax` as a backend for an algorithmic research project, such as in cryo-EM structure determination. These two aims are possible because `cryojax` is written to be fully interoperable with anything else in the JAX ecosystem.

Dig a little deeper and you'll find that `cryojax` aims to be a fully extensible modeling language for cryo-EM image formation. It implements a collection of abstract interfaces, which aim to be general enough to support any level of modeling complexity—from simple linear image formation to the most realistic physical models in the field. Best of all, these interfaces are all part of the public API. Users can create their own extensions to `cryojax`, tailored to their specific use-case!

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

The [`jax-finufft`](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. These are included as an option for computing image projections of real-space voxel-based scattering potential representations. In this case, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Simulating an image

The following is a basic workflow to simulate an image.

First, instantiate the spatial potential energy distribution representation and its respective method for computing image projections.

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cxs
from cryojax.io import read_array_with_spacing_from_mrc

# Instantiate the scattering potential
filename = "example_scattering_potential.mrc"
real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(filename)
potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, voxel_size)
# ... now, instantiate the pose. Angles are given in degrees
pose = cxs.EulerAnglePose(
    offset_x_in_angstroms=5.0,
    offset_y_in_angstroms=-3.0,
    phi_angle=20.0,
    theta_angle=80.0,
    psi_angle=-10.0,
)
# ... now, build the ensemble. In this case, the ensemble is just a single structure
structural_ensemble = cxs.SingleStructureEnsemble(potential, pose)
```

Here, the 3D scattering potential array is read from `filename` (see the documentation [here](https://mjo22.github.io/cryojax/examples/compute-potential/) for an example of how to generate the potential). Then, the abstraction of the scattering potential is then loaded in fourier-space into a `FourierVoxelGridPotential`, and subsequently the representation of a biological specimen is instantiated, which also includes a pose and conformational heterogeneity. Here, the `SingleStructureEnsemble` class takes a pose but has no heterogeneity.

Next, build the *scattering theory*. The simplest `scattering_theory` is the `WeakPhaseScatteringTheory`. This represents the usual image formation pipeline in cryo-EM, which forms images by computing projections of the potential and convolving the result with a contrast transfer function.

```python
from cryojax.image import operators as op

# Initialize the scattering theory. First, instantiate fourier slice extraction
potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
# ... next, the contrast transfer theory
ctf = cxs.CTF(defocus_in_angstroms=9800.0, astigmatism_in_angstroms=200.0, astigmatism_angle=10.0)
transfer_theory = cxs.ContrastTransferTheory(
    ctf, envelope=op.FourierGaussian(b_factor=5.0), amplitude_contrast_ratio=0.1)
# ... now for the scattering theory
scattering_theory = cxs.WeakPhaseScatteringTheory(
    structural_ensemble, potential_integrator, transfer_theory
)
```

The `ContrastTransferFunction` has parameters used in CTFFIND4, which take their default values if not
explicitly configured here. Finally, we can instantiate the `image_model`--the highest level of imaging abstraction in `cryojax`--and simulate an image. Here, we choose a `ContrastImageModel`, which simulates image contrast from a linear scattering theory.

```python
# Finally, build the image formation model
# ... first instantiate the instrument configuration
instrument_config = cxs.InstrumentConfig(shape=(320, 320), pixel_size=voxel_size, voltage_in_kilovolts=300.0)
# ... now the imaging pipeline
image_model = cxs.ContrastImageModel(instrument_config, scattering_theory)
# ... finally, simulate an image and return in real-space!
image_without_noise = image_model.render(outputs_real_space=True)
```

`cryojax` also defines a library of distributions from which to sample the data. These distributions define the stochastic model from which images are drawn. For example, instantiate an `IndependentGaussianFourierModes` distribution and either sample from it or compute its log-likelihood.

```python
import cryojax.distributions as dist
from cryojax.image import rfftn, operators as op

# Passing the ImagePipeline and a variance function, instantiate the distribution
distribution = dist.IndependentGaussianFourierModes(
    image_model, variance_function=op.Constant(1.0), normalizes_signal=True
)
# ... then, either simulate an image from this distribution
key = jax.random.key(seed=0)
image_with_noise = distribution.sample(key)
# ... or compute the likelihood
observed = rfftn(...)  # for this example, read in observed data and take FFT
log_likelihood = distribution.log_likelihood(observed)
```

For more advanced image simulation examples and to understand the many features in this library, see the [documentation](https://mjo22.github.io/cryojax/).

## Acknowledgements

- `cryojax` has been greatly informed by the open-source cryo-EM softwares [`cisTEM`](https://github.com/timothygrant80/cisTEM) and [`BioEM`](https://github.com/bio-phys/BioEM).
- `cryojax` relies heavily on and has taken great inspiration from [`equinox`](https://github.com/patrick-kidger/equinox/). We think that `equinox` has great design principles and highly recommend learning about it to fully make use of the power of `jax`.
