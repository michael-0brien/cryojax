# Overview

CryoJAX is a cryo-EM image simulation library for building downstream data analysis applications that leverage the scientific computing resources of JAX. The scope of what is in cryoJAX is 1) Its framework for simulating cryo-EM images and 2) Utilities for building data analysis applications. These are organized into a handful of submodules.

## TL;DR - a full breakdown of cryoJAX

| Submodule | Description |
| :-------- | :---------- |
| [`cryojax.simulator`](./api/simulator/entry-point.md) |  Image simulation models and algorithms |
| [`cryojax.io`](./api/io.md) |  Basic I/O for cryo-EM file formats |
| [`cryojax.ndimage`](./api/ndimage.md) | Image and volume manipulation |
| [`cryojax.jax_util`](./api/jax_util.md) | Functions improving JAX/Equinox user experience |
| [`cryojax.atom_util`](./api/atom_util.md)  | Operate on atoms and their coordinates |
| [`cryojax.rotations`](./api/rotations.md) | Backend for coordinate rotations |
| [`cryojax.constants`](./api/constants.md) | Handle physical constants |

<!-- ### [`cryojax.simulator`](./api/simulator/entry-point.md)

This contains algorithms and models for cryo-EM image simulation, as well as a framework for implementing new models and algorithms. See the [simulate an image](./examples/simulate-image.ipynb) tutorial to get started.

### [`cryojax.io`](./api/io.md)

For example, read/write MRC files and PDB/PDBx files.

### [`cryojax.ndimage`](./api/ndimage.md)

This includes cropping / padding, masks / filters, and downsampling. -->
