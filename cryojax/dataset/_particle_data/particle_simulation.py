from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from ...jax_util import NDArrayLike, filter_bscan
from .base_particle_dataset import AbstractParticleDataset, AbstractParticleParameterFile


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")
T = TypeVar("T")


def simulate_particle_stack(
    dataset: AbstractParticleDataset,
    simulate_fn: Callable[
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
    `AbstractParticleDataset`.

    !!! note
        This function works generally for a `simulate_fn`
        of the form

        ```python
        image = simulate_fn(
            parameter_info, constant_args, per_particle_args
        )
        ```

        where `parameter_info` is the pytree read from the
        `AbstractParticleDataset.parameter_file`,
        `constant_args` is a parameter that does not change
        between images, and `per_particle_args` is a pytree whose
        leaves have a batch dimension equal to the number of particles
        to be simulated.

    *Example 1*: Basic usage such as instantiating an
    `AbstractParticleDataset` and writing a
    `simulate_fn`

    ```python
    import cryojax.simulator as cxs
    import jax
    from cryojax.data import RelionParticleDataset
    from jaxtyping import PyTree

    # Load a `RelionParticleDataset` object. This loads
    # parameters and writes images
    dataset = RelionParticleDataset(..., mode='w')

    # Write your `simulate_fn` function, building an
    # `AbstractImageModel` (see tutorials for details)

    def simulate_fn(
        parameter_info: PyTree, # loaded from `dataset.parameter_file`
        constant_args: PyTree,
        _,
    ) -> jax.Array:
        # `constant_args` do not change between images. For
        # example, include the method of taking projections
        ... = constant_args
        # Using the pose, CTF, and image config from the
        # `parameter_info`, build image simulation model
        image_model = cxs.make_image_model(...)
        # ... and compute
        return image_model.simulate()

    # Simulate images and write to disk
    simulate_particle_stack(
        dataset,
        simulate_fn,
        constant_args=(...)
        per_particle_args=None, # default
        batch_size=10,
    )
    ```

    *Example 2*: More-advanced usage, writing a
    `simulate_fn` that simulates images with noise.
    Uses `per_particle_args` as well as `constant_args`.

    ```python
    import cryojax.simulator as cxs
    import jax
    from cryojax.data import RelionParticleDataset
    from jaxtyping import Array, PyTree, Shaped

    # Load a `RelionParticleDataset` object. This loads
    # parameters and writes images
    dataset = RelionParticleDataset(..., mode='w')

    # Instantiate per-particle arguments. First, the RNG keys used
    # to generate the noise
    seed = 0
    key = jax.random.key(seed)
    key, *keys_noise = jax.random.split(key, n_images+1)
    keys_noise = jnp.array(keys_noise)
    # ... then, add a scaling parameter for the images
    key, subkey = jax.random.split(key)
    scaling_params = jax.random.uniform(subkey, shape=(n_images,))

    # Now write your `simulate_fn` function, building a
    # `cryojax.simulator.GaussianWhiteNoiseModel` to
    # simulate images with white noise (see tutorials for details)

    def simulate_fn(
        parameter_info: PyTree,
        constant_args: PyTree,
        per_particle_args: PyTree[Shaped[Array, "_ ..."]],
    ) -> jax.Array:
        ... = constant_args
        key, scale = per_particle_args

        # Combine two previously split PyTrees
        image_model = cxs.make_image_model(...)
        distribution = cxs.GaussianWhiteNoiseModel(image_model, ...)

        return scale * distribution.sample(key)

    simulate_particle_stack(
        dataset,
        simulate_fn,
        constant_args=(...)
        per_particle_args=(keys_noise, scaling_params)
        batch_size=10,
    )
    ```

    **Arguments:**

    - `dataset`:
        The `AbstractParticleDataset` dataset. Note that this must be
        passed in *writing mode*, i.e. `mode = 'w'`.
    - `simulate_fn`:
        A callable that computes the image stack from the parameters contained
        in the STAR file.
    - `constant_args`:
        The constant arguments to pass to the `simulate_fn` function.
        These must be the same for all images.
    - `per_particle_args`:
        Arguments to pass to the `simulate_fn` function.
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
        `AbstractParticleDataset.parameter_file.save`.
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
    simulate_batch_fn = _configure_simulation_fn(
        simulate_fn,
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
        images, parameter_info = _simulate_images(
            dataset_index,
            parameter_file,
            simulate_batch_fn,
            constant_args,
            per_particle_args,
        )
        dataset.write_images(dataset_index, images, parameter_info)
    # ... handle remainder
    if remainder > 0:
        simulate_batch_fn = _configure_simulation_fn(simulate_fn, batch_size, remainder)
        index_array = np.arange(n_particles - remainder, n_particles, dtype=int)
        images, parameter_info = _simulate_images(
            index_array,
            parameter_file,
            simulate_batch_fn,
            constant_args,
            per_particle_args,
        )
        dataset.write_images(index_array, images, parameter_info)
    # Finally, save metadata file
    parameter_file.save(**kwargs)


def _simulate_images(
    index: Int[np.ndarray, " _"],
    parameter_file: AbstractParticleParameterFile,
    simulate_batch_fn: Callable[
        [PyTree, ConstantT, PerParticleT],
        Float[NDArrayLike, "_ _ _"],
    ],
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
) -> tuple[Float[NDArrayLike, "_ _ _"], PyTree]:
    parameter_info = parameter_file[index]
    args = (constant_args, _index_pytree(index, per_particle_args))
    image_stack = simulate_batch_fn(parameter_info, *args)

    return image_stack, parameter_info


def _configure_simulation_fn(
    simulate_fn: Callable[
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

        def simulate_batch_fn(parameter_info, constant_args, per_particle_args):  # type: ignore
            parameter_info_at_0, per_particle_args_at_0 = (
                _index_pytree(0, parameter_info),
                _index_pytree(0, per_particle_args),
            )
            shape = _determine_output_shape(
                simulate_fn, parameter_info_at_0, constant_args, per_particle_args_at_0
            )
            image_stack = np.empty((images_per_file, *shape))
            for i in range(images_per_file):
                parameter_info_at_i = _index_pytree(i, parameter_info)
                per_particle_args_at_i = _index_pytree(i, per_particle_args)
                image = simulate_fn(
                    parameter_info_at_i, constant_args, per_particle_args_at_i
                )
                image_stack[i] = np.asarray(image)
            return image_stack

    else:
        batch_size = min(images_per_file, batch_size)
        compute_vmap = eqx.filter_vmap(
            simulate_fn, in_axes=(eqx.if_array(0), None, eqx.if_array(0))
        )
        if batch_size == images_per_file:
            simulate_batch_fn = eqx.filter_jit(compute_vmap)
        else:

            @eqx.filter_jit
            def simulate_batch_fn(parameter_info, constant_args, per_particle_args):
                # Compute with `jax.lax.scan` via `cryojax.jax_util.filter_bscan`
                init = constant_args
                xs = (parameter_info, per_particle_args)

                def f_scan(carry, xs):
                    _constant_args = carry
                    _parameter_info, _per_particle_args = xs
                    image_stack = compute_vmap(
                        _parameter_info, _constant_args, _per_particle_args
                    )
                    return carry, image_stack

                _, image_stack = filter_bscan(f_scan, init, xs, batch_size=batch_size)

                return image_stack

    return simulate_batch_fn  # type: ignore


def _index_pytree(
    index: int | Int[np.ndarray, ""] | Int[np.ndarray, " _"], pytree: T
) -> T:
    dynamic, static = eqx.partition(pytree, eqx.is_array)
    dynamic_at_index = jax.tree.map(lambda x: x[index], dynamic)
    return eqx.combine(dynamic_at_index, static)


def _determine_output_shape(_fn, *_args):
    jaxpr_fn = eqx.filter_make_jaxpr(_fn)
    _, out_dynamic, out_static = jaxpr_fn(*_args)
    out_struct = eqx.combine(out_dynamic, out_static)
    return out_struct.shape
