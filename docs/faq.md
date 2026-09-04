# FAQs

## Why JAX and not PyTorch?

First of all, we recommend reading the [Equinox FAQs](https://docs.kidger.site/equinox/faq/#pytorch) for description of how JAX+Equinox compares with PyTorch in general.

In context of cryo-EM, PyTorch is often an good choice for writing simulation/analysis code, has already been an engine for cryo-EM research, and in many contexts has advantages over JAX given its huge base of users. In fact, [teamtomo](https://teamtomo.org/) is building an ecosystem of extensible code for cryo-EM based on PyTorch, and we highly encourage you to check it out if you haven't already!

However, the specific goal of the cryoJAX library is to be a flexible platform for downstream data analysis applications that need a differentiable, scalable forward model and the ability to hack its implementation. To this end, there are a few reasons why we like JAX:

**JAX function transformations are extremely flexible**

JAX is capable of transforming *entire programs* with `jax.jit`, `jax.grad`, and `jax.vmap` across many accelerators. Cryo-EM is exploding with new exciting research applications, which are highly varied in how exactly to deploy and optimize a forward model. Function transformations happen to be an extremely useful programming model towards this end: all we need to do is write the forward model, and users can take control of how to use it in practice.

Inspired from JAX, PyTorch has recently implemented function transformations via [`torch.func`](https://docs.pytorch.org/docs/stable/func.html). To our knowledge, these cannot yet handle of the same complexity as in JAX (especially enabled by Equinox), and at this stage designing a package that fully relies on `torch.func` for its usability could be problematic. At some point, this would be very interesting.

**CryoJAX is simple to maintain**

A collorary of the previous point is that cryoJAX becomes relatively straightforward to maintain! We don't need to include any assumptions in the library of how your code will be batched, what parameters you would like to infer from data, or how you'd like to distribute your computation over devices. This is all handled at runtime.

Arguably, these facts enable the existence of cryoJAX; the difficulty of maintaining a range of models and algorithms for image simulation is already challenge enough!

**The JAX scientific computing ecosystem**

JAX has a growing ecosystem for scientific computing in the physical sciences, such as for statistical inference. These can be leveraged for building exciting new cryo-EM data analysis applications. Check out [`optimistix`](https://github.com/patrick-kidger/optimistix) for non-linear optimization, [`lineax`](https://github.com/patrick-kidger/lineax) for linear solvers, and [`blackjax`](https://github.com/blackjax-devs/blackjax) for sampling.

Additionally, cryoJAX is a part of a growing number of libraries for physical modeling, and some of these could be used with cryoJAX for cryo-EM. For example, see [`diffrax`](https://github.com/patrick-kidger/diffrax) for differential equation solvers and [`jax-md`](https://github.com/jax-md/jax-md) for molecular dynamics.

See the Equinox [awesome list](https://docs.kidger.site/equinox/awesome-list/#awesome-list) for more libraries in the JAX ecosystem.

## I need help with debugging!

It can be challenging to debug JAX JIT-compiled code. To start learning about this, I recommend learning about [`jax.debug`](https://docs.jax.dev/en/latest/debugging/index.html#debugging-runtime-values) and [`equinox.error_if`](https://docs.kidger.site/equinox/api/errors/#equinox.error_if).

CryoJAX implements internal runtime checks for things such as if positive quantities (e.g. pixel size) have somehow become negative, such as during gradient descent. By default, cryoJAX will not perform these checks, but enable them by setting the environmental variable `CRYOJAX_ENABLE_CHECKS=true`.

These checks are performed using the function [`cryojax.jax_util.maybe_error_if`][], which wraps [`equinox.error_if`](https://docs.kidger.site/equinox/api/errors/#equinox.error_if). To change the behavior of run-time error checks, see the `error_if` documentation to learn about the `EQX_ON_ERROR` environmental variable.

Also, note that [`cryojax.jax_util.maybe_error_if`][] is public API so that developers of downstream cryoJAX libraries may also use `CRYOJAX_ENABLE_CHECKS`!

## Can I give parameters a batch dimension?

Yes. Classes such as the
[`cryojax.simulator.EulerAnglePose`][], [`cryojax.simulator.AstigmaticCTF`][], and [`cryojax.simulator.BasicImageConfig`][] accept array parameters with a leading batch dimension. For example,

```python
import numpy as np
import cryojax.simulator as cxs

# A "stack" of 100 poses
pose = cxs.EulerAnglePose(phi_angle=np.linspace(0.0, 360.0, 100))
```

is a perfectly valid object to construct.

!!! warning "Batched objects must be used inside `vmap`"

    A batched object like the one above can only be *used* to simulate an image
    inside a `jax.vmap` (or `equinox.filter_vmap`) region, which maps over the
    batch dimension. Passing a batched object directly to a simulation routine
    outside of `vmap` is a common gotcha: it will either raise a shape error
    or—more dangerously—**silently produce an incorrect result**, because the
    compute-time code assumes its array leaves are unbatched.

    For a concrete example, consider building 100 poses and simulating an image
    for each. The following is **wrong**:

    ```python
    # 100 poses, each with a different phi angle
    pose = cxs.EulerAnglePose(phi_angle=np.linspace(0.0, 360.0, 100))
    image_model = cxs.make_image_model(volume, image_config, pose, transfer_theory)

    # WRONG: `pose` is batched but `simulate` is not mapped over the batch.
    # Inside the compute code, `pose.offset_in_angstroms[0]` now indexes the
    # *batch* axis instead of the x-component, so the phase-shift translation is
    # silently computed from the wrong numbers. You get a single (H, W) image
    # back with no error, even though 100 were intended.
    image = image_model.simulate()
    ```

    Instead, map over the batch dimension explicitly. The **correct** version:

    ```python
    import equinox as eqx

    # Batch only over the leaves that actually have a batch dimension (the pose)
    @eqx.filter_vmap(in_axes=(eqx.if_array(0),))
    def simulate_stack(pose):
        image_model = cxs.make_image_model(
            volume, image_config, pose, transfer_theory
        )
        return image_model.simulate()

    images = simulate_stack(pose)  # shape (100, H, W), as intended
    ```

## When are my parameters on the CPU versus the GPU?

CryoJAX classes do not move their inputs between the host (CPU) and a
device (e.g. GPU) at construction time. Instead, they store array parameters as
[`cryojax.jax_util.NDArrayLike`][] leaves and preserve the backend of whatever
you pass in:

- A **JAX array** (including a tracer inside `jit`/`vmap`) is kept on its
  device.
- A **NumPy array** is kept on the host.
- A **Python scalar/sequence** (such as a default argument) is converted to a
  NumPy array on the host.

Both NumPy and JAX arrays are treated as *traced*
parameters by Equinox's filtered transformations (e.g. `equinox.filter_jit`),
so both backends behave as parameters. The stored leaves
are cast to JAX arrays at compute time, so a NumPy leaf built inside a JIT trace
is simply lifted onto the device where it is used.
