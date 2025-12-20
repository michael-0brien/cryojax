# Extra JAX/Equinox

`cryojax.jax_util` supports downstream applications with helpers for common patterns using cryoJAX for cryo-EM data analysis.

## Equinox extensions

### Helpers for filtering

To make use of the full power of JAX, it is highly recommended to learn about [equinox](https://docs.kidger.site/equinox/all-of-equinox/). Using `equinox`, cryoJAX implements its [models as pytrees](https://docs.kidger.site/equinox/all-of-equinox/#1-models-as-pytrees) using `equinox.Module`s. These pytrees can be operated on similarly to any pytree with JAX (e.g. with `jax.tree.map`). Complementary to the `equinox.Module` interface, `equinox` introduces the idea of [*filtering*](https://docs.kidger.site/equinox/all-of-equinox/#2-filtering) in order to separate pytree leaves into different groups, a core task in using JAX (e.g. separating traced and static arguments to `jax.jit`). In particular, this grouping is achieved with the functions [eqx.partition and eqx.combine](https://docs.kidger.site/equinox/api/manipulation/#equinox.partition). This documentation describes utilities in `cryojax` for working with `equinox.partition` and `equinox.combine`.

::: cryojax.jax_util.make_filter_spec

### Batched loops

`cryojax` provides a collection of useful functions not found in JAX that tend to be important for tasks in cryo-EM. Depending on developments with core JAX/Equinox and other factors, these functions could be removed in future releases of `cryojax`. Use with caution!

::: cryojax.jax_util.filter_bmap

::: cryojax.jax_util.filter_bscan

## Grid search interface


## Reparameterizing pytrees


## Extra type hints using `jaxtyping`

::: cryojax.jax_util.NDArrayLike
    options:
        members: false
        show_symbol_type_heading: false

::: cryojax.jax_util.FloatLike
    options:
        members: false
        show_symbol_type_heading: false

::: cryojax.jax_util.ComplexLike
    options:
        members: false
        show_symbol_type_heading: false

::: cryojax.jax_util.InexactLike
    options:
        members: false
        show_symbol_type_heading: false

::: cryojax.jax_util.IntLike
    options:
        members: false
        show_symbol_type_heading: false

::: cryojax.jax_util.BoolLike
    options:
        members: false
        show_symbol_type_heading: false
