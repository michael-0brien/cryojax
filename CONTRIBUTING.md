# Contributor Guide

Contributions to this repository are welcome. It would be great for cryoJAX to be supported by a larger community.

## Making a feature request

To submit a feature request, open an a thread on the [issues](https://github.com/michael-0brien/cryojax/issues) page. There we can discuss if something is appropriate for the cryoJAX core library, or if it belongs in a separate library or workflow.

After discussing the contribution and please implement a draft of it in your local fork of cryoJAX. Then, open a [pull request](https://github.com/michael-0brien/cryojax/pulls). If this is too cumbersome, you could provide the code from a script where you've implemented the feature in the [issues](https://github.com/michael-0brien/cryojax/issues) discussion thread. The cryoJAX developers can help figure out how to place it in the package. Please open pull requests to the repository's `dev` branch.

### What belongs in the cryoJAX core library?

CryoJAX does not try to be a one-stop shop for cryo-EM analysis. Instead, it is a modeling framework for image simulation via abstract base class (ABC) interfaces that ship with core functionality for image simulation. CryoJAX also supports utilities for building data analysis or working with JAX downstream.

For example, say you'd like to contribute a method for simulating cryo-EM images, such as a new model of the CTF. Your model should be common knowledge in the field and/or demonstrated in real experiments. In general, a good metric of whether or not an image simulation model or algorithm belongs in cryoJAX could be but is not limited to the following: "has this been shown to increase resolution of 3D reconstructions in real experiments?". As another example, let's say you want to contribute a method for computing projections of 3D volumes. If your method is faster or more numerically accurate than something already in cryoJAX, it would also be appropriate for the package.

### What belongs in a separate library or workflow?

CryoJAX is built to support an ecosystem of downstream applications. If an image simulation model or algorithm is being prototyped, then it belongs downstream to cryoJAX. Further, if it is not common to many users---such as functionality for particular proteins---it also belongs downstream. If your application cannot be built downstream, it may be necessary to update the cryoJAX ABC interface. In this case, please also open an [issue](https://github.com/michael-0brien/cryojax/issues).

## Reporting a bug

Make bug reports on the [issues](https://github.com/michael-0brien/cryojax/issues) page. Please provide a test case, and/or steps to reproduce the issue. In particular, consider including a [minimal, reproducible example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to contribute

Let's say you are submitted a bug fix or a feature request to cryoJAX. To contribute, first fork the library on github. Then clone and install the library with dependencies for development and testing:

```
git clone https://github.com/your-username-here/cryojax.git
cd cryojax
git checkout dev
python -m pip install -e '.[dev, tests]'
```

Next, install the pre-commit hooks:

```
pre-commit install
```

This uses `ruff` to format and lint the code. Now, you can push changes to your local fork.

### Running tests

After making changes, make sure that the tests pass. In the `cryojax` base directory, install testing dependencies and run

```
python -m pytest
```

**If you are using a non-linux OS, the [`pycistem`](https://github.com/jojoelfe/pycistem) testing dependency cannot be installed**. In this case, in order to run the tests against [`cisTEM`](https://github.com/timothygrant80/cisTEM), run the testing [workflow](https://github.com/michael-0brien/cryojax/actions/workflows/ci_build.yml). This can be done manually or will happen automatically when a PR is opened.

### Submitting changes

If the tests look okay, open a [pull request](https://github.com/michael-0brien/cryojax/pulls) from your fork the `dev` branch. The developers can review your PR and request changes / add further tweaks if necessary.

### Optional: build documentation

For a given PR it may also be necessary to build the cryoJAX documentation or run jupyter notebook examples. The documentation is easily built using [`mkdocs`](https://www.mkdocs.org/getting-started/#getting-started-with-mkdocs). To make sure the docs build, run the following:

```
python -m pip install -e '.[docs]'
mkdocs build
```

You can also run `mkdocs serve` and follow the instructions in your terminal to inpsect the webpage on your local server.

To run the notebooks in the documentation, it may be necessary to pull large-ish files from [git LFS](https://git-lfs.com/).

```
sudo apt-get install git-lfs  # If using macOS, `brew install git-lfs`
git lfs install; git lfs pull
```

## Design principles

`cryojax` is built on [equinox](https://docs.kidger.site/equinox/). In short, `equinox` provides an interface to writing parameterized functions in `jax`. The core object of these parameterized functions is called a [Module](https://docs.kidger.site/equinox/api/module/module/) (yes, this takes inspiration from pytorch). `equinox` ships with features to interact with these `Module`s, and more generally with [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html).

Equinox also provides a recommended pattern for writing `Module`s: https://docs.kidger.site/equinox/pattern/. We think this is a good template for code readability, so `cryojax` tries to adhere to these principles.
