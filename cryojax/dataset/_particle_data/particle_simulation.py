from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from ...jax_util import NDArrayLike, filter_bscan
from .base_particle_dataset import (
    AbstractParticleParameterFile,
    AbstractParticleStackDataset,
)
from .relion import ParticleParameterInfo


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")
T = TypeVar("T")


def simulate_particle_stack(
    dataset: AbstractParticleStackDataset,
    compute_image_fn: Callable[
        [PyTree, ConstantT, PerParticleT],
        Float[Array, "_ _"],
    ],
    constant_args: ConstantT = None,
    per_particle_args: PerParticleT = None,
    batch_size: int | None = None,
    images_per_file: int | None = None,
    **kwargs: Any,
):
    """Write a stack of images from parameters contained in an
    `AbstractParticleStackDataset`.

    !!! note
        This function works generally for a `compute_image_fn`
        of the form

        ```python
        image = compute_image_fn(
            parameters, constant_args, per_particle_args
        )
        ```

        where `parameters` is the pytree read from the
        `AbstractParticleStackDataset.parameter_file`,
        `constant_args` is a parameter that does not change
        between images, and `per_particle_args` is a pytree whose
        leaves have a batch dimension equal to the number of particles
        to be simulated.

    *Example 1*: Basic usage such as instantiating an
    `AbstractParticleStackDataset` and writing a
    `compute_image_fn`

    ```python
    import cryojax.simulator as cxs
    import jax
    from cryojax.data import RelionParticleStackDataset
    from jaxtyping import PyTree

    # Load a `RelionParticleStackDataset` object. This loads
    # parameters and writes images
    dataset = RelionParticleStackDataset(..., mode='w')

    # Write your `compute_image_fn` function, building an
    # `AbstractImageModel` (see tutorials for details)

    def compute_image_fn(
        parameters: RelionParticleParameters,
        constant_args: PyTree,
        _,
    ) -> jax.Array:
        # `constant_args` do not change between images. For
        # example, include the method of taking projections
        integrator, ... = constant_args
        # Using the pose, CTF, and image config from the
        # `parameters`, build image simulation model
        image_model = cxs.make_image_model(...)
        # ... and compute
        return image_model.simulate()

    # Simulate images and write to disk
    simulate_particle_stack(
        dataset,
        compute_image_fn,
        constant_args=(integrator, ...)
        per_particle_args=None, # default
        batch_size=10,
    )
    ```

    *Example 2*: More-advanced usage, writing a
    `compute_image_fn` that simulates images with noise.
    Uses `per_particle_args` as well as `constant_args`.

    ```python
    import cryojax.simulator as cxs
    import jax
    from cryojax.data import RelionParticleStackDataset
    from jaxtyping import Array, PyTree, Shaped

    # Load a `RelionParticleStackDataset` object. This loads
    # parameters and writes images
    dataset = RelionParticleStackDataset(..., mode='w')

    # Instantiate per-particle arguments. First, the RNG keys used
    # to generate the noise
    seed = 0
    key = jax.random.key(seed)
    key, *keys_noise = jax.random.split(key, n_images+1)
    keys_noise = jnp.array(keys_noise)
    # ... then, add a scaling parameter for the images
    key, subkey = jax.random.split(key)
    scaling_params = jax.random.uniform(subkey, shape=(n_images,))

    # Now write your `compute_image_fn` function, building a
    # `cryojax.simulator.UncorrelationGaussianNoiseModel` to
    # simulate images with white noise (see tutorials for details)

    def compute_image_fn(
        particle_parameters: RelionParticleParameters,
        constant_args: PyTree,
        per_particle_args: PyTree[Shaped[Array, "_ ..."]],
    ) -> jax.Array:
        ... = constant_args
        key, scale = per_particle_args

        # Combine two previously split PyTrees
        image_model = cxs.make_image_model(...)
        distribution = cxs.UncorrelatedGaussianNoiseModel(image_model, ...)

        return scale * distribution.sample(key)

    simulate_particle_stack(
        dataset,
        compute_image_fn,
        constant_args=(...)
        per_particle_args=(keys_noise, scaling_params)
        batch_size=10,
    )
    ```

    **Arguments:**

    - `dataset`:
        The `AbstractParticleStackDataset` dataset. Note that this must be
        passed in *writing mode*, i.e. `mode = 'w'`.
    - `compute_image_fn`:
        A callable that computes the image stack from the parameters contained
        in the STAR file.
    - `constant_args`:
        The constant arguments to pass to the `compute_image_fn` function.
        These must be the same for all images.
    - `per_particle_args`:
        Arguments to pass to the `compute_image_fn` function.
        This is a pytree with leaves having a batch size with equal dimension
        to the number of images.
    - `batch_size`:
        The number images to compute in parallel using `jax.vmap`.
        If `None`, simulate images in a python for-loop. This is
        useful if the user isn't yet familiar with debugging JIT
        compilation.
    - `images_per_file`:
        The number of images to write in a single image file. By default,
        set this as the number of particles in the dataset.
    - `kwargs`:
        Keyword arguments passed to
        `AbstractParticleStackDataset.parameter_dataset.save`.
    """
    if dataset.mode == "r":
        raise ValueError(
            "Found that the `dataset` was in reading mode "
            "(`mode = 'r'`), but this must be instantiated in "
            "writing mode (`mode = 'w'`)."
        )
    n_particles = len(dataset)
    images_per_file = n_particles if images_per_file is None else images_per_file
    # Get function that simulates batch of images
    compute_image_stack_fn = _configure_simulation_fn(
        compute_image_fn,
        batch_size,
        images_per_file,
    )
    # Run control flow
    n_iterations, remainder = (
        n_particles // images_per_file,
        n_particles % images_per_file,
    )
    parameter_file = dataset.parameter_file
    for file_index in range(n_iterations):
        dataset_index = np.arange(
            file_index * images_per_file, (file_index + 1) * images_per_file, dtype=int
        )
        images, parameters = _simulate_images(
            dataset_index,
            parameter_file,
            compute_image_stack_fn,
            constant_args,
            per_particle_args,
        )
        dataset.write_images(dataset_index, images, parameters)
    # ... handle remainder
    if remainder > 0:
        compute_image_stack_fn = _configure_simulation_fn(
            compute_image_fn, batch_size, remainder
        )
        index_array = np.arange(n_particles - remainder, n_particles, dtype=int)
        images, parameters = _simulate_images(
            index_array,
            parameter_file,
            compute_image_stack_fn,
            constant_args,
            per_particle_args,
        )
        dataset.write_images(index_array, images, parameters)
    # Finally, save metadata file
    parameter_file.save(**kwargs)


def _simulate_images(
    index: Int[np.ndarray, " _"],
    parameter_file: AbstractParticleParameterFile,
    compute_image_stack_fn: Callable[
        [PyTree, ConstantT, PerParticleT],
        Float[NDArrayLike, "_ _ _"],
    ],
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
) -> tuple[Float[NDArrayLike, "_ _ _"], ParticleParameterInfo]:
    parameters = parameter_file[index]
    args = (constant_args, _index_pytree(index, per_particle_args))
    image_stack = compute_image_stack_fn(parameters, *args)

    return image_stack, parameters


def _configure_simulation_fn(
    compute_image_fn: Callable[
        [PyTree, ConstantT, PerParticleT],
        Float[Array, "_ _"],
    ],
    batch_size: int | None,
    images_per_file: int,
) -> Callable[
    [PyTree, ConstantT, PerParticleT],
    Float[NDArrayLike, "_ _ _"],
]:
    if batch_size is None:

        def compute_image_stack_fn(parameters, constant_args, per_particle_args):  # type: ignore
            shape = parameters["image_config"].shape
            image_stack = np.empty((images_per_file, *shape))
            for i in range(images_per_file):
                parameters_at_i = _index_pytree(i, parameters)
                per_particle_args_at_i = _index_pytree(i, per_particle_args)
                image = compute_image_fn(
                    parameters_at_i, constant_args, per_particle_args_at_i
                )
                image_stack[i] = np.asarray(image)
            return image_stack

    else:
        batch_size = min(images_per_file, batch_size)
        compute_vmap = eqx.filter_vmap(
            compute_image_fn, in_axes=(eqx.if_array(0), None, eqx.if_array(0))
        )
        if batch_size == images_per_file:
            compute_image_stack_fn = eqx.filter_jit(compute_vmap)
        else:

            @eqx.filter_jit
            def compute_image_stack_fn(parameters, constant_args, per_particle_args):
                # Compute with `jax.lax.scan` via `cryojax.jax_util.filter_bscan`
                init = constant_args
                xs = (parameters, per_particle_args)

                def f_scan(carry, xs):
                    _constant_args = carry
                    _parameters, _per_particle_args = xs
                    image_stack = compute_vmap(
                        _parameters, _constant_args, _per_particle_args
                    )
                    return carry, image_stack

                _, image_stack = filter_bscan(f_scan, init, xs, batch_size=batch_size)

                return image_stack

    return compute_image_stack_fn  # type: ignore


def _index_pytree(
    index: int | Int[np.ndarray, ""] | Int[np.ndarray, " _"], pytree: T
) -> T:
    dynamic, static = eqx.partition(pytree, eqx.is_array)
    dynamic_at_index = jax.tree.map(lambda x: x[index], dynamic)
    return eqx.combine(dynamic_at_index, static)
