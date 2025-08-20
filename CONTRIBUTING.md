# Contributor Guide

Contributions to this repository are welcome and greatly appreciated! We would love
for this package to grow and be supported by a larger community.

## What contributions fit into `cryojax`?

CryoJAX does not try to be a one-stop shop for cryo-EM analysis. Instead, it is a modeling framework for image simulation via its abstract base class (ABC) interfaces that ships with core functionality for image simulation. CryoJAX also supports some utilities for building data analysis or working with JAX downstream.

### What belongs in the cryoJAX core library?

Core functionality for image simulation should be common knowledge in the field and/or demonstrated in real experiments. A good metric of whether or not an image simulation model or algorithm belongs in cryoJAX could be but is not limited to the following: "has this been shown to increase resolution in 3D reconstructions in real experiments?". If you would like to discuss if something is appropriate for cryoJAX core, please open an [issue](https://github.com/mjo22/cryojax/issues) and make a feature request.

### What belongs in a separate library or workflow?

If an image simulation model or algorithm is being prototyped, then it belongs downstream to cryoJAX. Further, if it is not common to many users---such as functionality for particular proteins---it also belongs downstream. If your application cannot be built downstream, it may be necessary to update the cryoJAX ABC interface. In this case, please also open an [issue](https://github.com/mjo22/cryojax/issues).

After discussing the contribution and implementing it either in your local fork of cryoJAX or in an external library, open a [Pull Request](https://github.com/mjo22/cryojax/pulls).

## Getting started

First, fork the library on GitHub. Then clone and install the library in development mode:

```
git clone https://github.com/your-username-here/cryojax.git
cd cryojax
python -m pip install -e .
```

Next, install the pre-commit hooks:

```
python -m pip install pre-commit
pre-commit install
```

This uses `ruff` to format and lint the code.

## Running tests

After making changes, make sure that the tests pass. In the `cryojax` base directory, run

```
python -m pip install -r tests/requirements.txt
python -m pytest
```

**If you are using a non-linux OS, the [`pycistem`](https://github.com/jojoelfe/pycistem) testing dependency cannot be installed**. In this case, in order to run the tests against [`cisTEM`](https://github.com/timothygrant80/cisTEM), run the testing [workflow](https://github.com/mjo22/cryojax/actions/workflows/testing.yml). This can be done manually or will happen automatically when a PR is opened.

## Building documentation

Again in the `cryojax` base directory, build the documentation by installing dependencies using [`mkdocs`](https://www.mkdocs.org/getting-started/#getting-started-with-mkdocs).

```
python -m pip install -r docs/requirements.txt
mkdocs serve
```

Then, navigate to the local webpage by following the instructions in your terminal. In order to run the notebooks in the documentation, it may be necessary to pull large-ish files from [git LFS](https://git-lfs.com/).

```
sudo apt-get install git-lfs  # If using macOS, `brew install git-lfs`
git lfs install; git lfs pull
```

## How to submit changes

Now, if the tests and documentation look okay, push your changes and open a [Pull Request](https://github.com/mjo22/cryojax/pulls)!

## Design principles

`cryojax` is built on [equinox](https://docs.kidger.site/equinox/). In short, `equinox` provides an interface to writing parameterized functions in `jax`. The core object of these parameterized functions is called a [Module](https://docs.kidger.site/equinox/api/module/module/) (yes, this takes inspiration from pytorch). `equinox` ships with features to interact with these `Module`s, and more generally with [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html).

Equinox also provides a recommended pattern for writing `Module`s: https://docs.kidger.site/equinox/pattern/. We think this is a good template for code readability, so `cryojax` tries to adhere to these principles as much as possible.

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/mjo22/cryojax/issues).

When filing an issue, here are some guidelines that may be helpful to know:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to
reproduce the issue. In particular, consider including a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).
