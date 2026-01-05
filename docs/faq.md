# FAQ

## Why JAX and not PyTorch?

PyTorch is an excellent choice for writing cryo-EM simulation/analysis code, has already been an engine for cryo-EM research, and in many contexts has advantages over JAX given its huge base of users. In fact, [teamtomo](https://teamtomo.org/) is building an ecosystem of extensible code for cryo-EM based on PyTorch, and we highly encourage you to check it out if you haven't already!

However, the specific goal of the cryoJAX library is to be a flexible platform for downstream data analysis applications that need a differentiable, scalable forward model and the ability to hack its implementation. To this end, there are a few reasons why we like JAX:

**JAX function transformations are extremely flexible**

JAX is capable of transforming *entire programs* with `jax.jit`, `jax.grad`, and `jax.vmap` across many accelerators. Cryo-EM is exploding with new exciting research applications, which are highly varied in how exactly to deploy and optimize a forward model. Deploying function transformations happens to be an extremely useful programming model towards this end: all we need to do is write the forward model, and users can take control of how to use it in practice.

Inspired from JAX, PyTorch has recently implemented function transformations via [`torch.func`](https://docs.pytorch.org/docs/stable/func.html). To our knowledge, these cannot yet handle of the same complexity as in JAX, and at this stage designing a package that fully relies on `torch.func` for its utility could be problem. At some point, this would be very interesting.

**CryoJAX is simple to maintain**

A collorary of the previous point is that cryoJAX becomes relatively straightforward to maintain! We don't need to include any assumptions in the library of how your code will be batched, what parameters you would like to infer from data, or how you'd like to distribute your computation over devices. This is all handled at runtime.

Arguably, these facts enable the existence of cryoJAX; the difficulty of maintaining a range of models and algorithms for image simulation is already challenge enough!

**The JAX scientific computing ecosystem**

JAX has a growing ecosystem for scientific computing, physical modeling, and statistical inference in the physical sciences. For example, check out [`optimistix`](https://github.com/patrick-kidger/optimistix), [`lineax`](https://github.com/patrick-kidger/lineax), and [`diffrax`](https://github.com/patrick-kidger/diffrax).
