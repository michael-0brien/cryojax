"""Kernel-agnostic Pallas/Triton machinery shared by the spreading and NUFFT
kernels: block sizing, lane masks, `enable_pallas` resolution, and the helpers
the forward-mode (JVP) rules are built from."""

import operator
from collections.abc import Mapping
from functools import reduce

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array

from .._config import CRYOJAX_ENABLE_PALLAS, CRYOJAX_PALLAS_BLOCK_SIZE


# The Triton compiler-params class has been renamed across jax versions
# (`TritonCompilerParams` up to jax 0.5.3, `CompilerParams` from 0.9.1+) --
# isolate the version-compat shim to this one place.
_CompilerParams = getattr(pltriton, "CompilerParams", None) or getattr(
    pltriton, "TritonCompilerParams"
)


def _choose_block_size(n_spread: int, ndim: int) -> int:
    """Number of points each Pallas program handles.

    Empirically tuned across three GPU generations (Ampere RTX 3090, Hopper
    H100 PCIe, Blackwell RTX PRO 6000; both 2D and 3D -- see project memory
    for the full sweep) -- unlike the original per-instruction-budget
    formula this replaces, whose rationale (bounding Python-unrolled
    instruction count) no longer applies now that the kernel uses
    `jax.lax.fori_loop`. A flat `block_size=128` is the best overall choice
    across all three architectures and both dimensionalities. Override via
    the `CRYOJAX_PALLAS_BLOCK_SIZE` environment variable if you've
    benchmarked a better value for your own (GPU, M, n_spread) --
    deliberately not a function parameter, to keep the number of ways to
    configure this down to one.
    """
    del n_spread, ndim  # kept for API stability; the flat default doesn't use them
    if CRYOJAX_PALLAS_BLOCK_SIZE is not None:
        return CRYOJAX_PALLAS_BLOCK_SIZE
    return 128


def _valid_lane_mask(block_size: int, m_total: int) -> Array:
    """Boolean mask marking which lanes in the current grid program
    correspond to real points vs. padding past `m_total` (when `m_total`
    isn't a multiple of `block_size`)."""
    prog = pl.program_id(0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (block_size,), 0)
    return (prog * block_size + lane) < m_total


def resolve_enable_pallas(
    enable_pallas: bool | Mapping[str, bool] | None,
) -> tuple[bool, bool]:
    """Resolve `enable_pallas` (see `api.py`) to `(use_pallas_fwd,
    use_pallas_bwd)`, against the `CRYOJAX_ENABLE_PALLAS` env var default,
    and fail fast if Pallas is requested without a GPU available (rather
    than deferring to whatever error `pallas_call` itself would raise)."""
    if enable_pallas is None:
        use_fwd = use_bwd = CRYOJAX_ENABLE_PALLAS
    elif isinstance(enable_pallas, bool):
        use_fwd = use_bwd = enable_pallas
    elif isinstance(enable_pallas, Mapping):
        use_fwd = enable_pallas.get("fwd", CRYOJAX_ENABLE_PALLAS)
        use_bwd = enable_pallas.get("bwd", CRYOJAX_ENABLE_PALLAS)
    else:
        raise TypeError(
            "`enable_pallas` must be `None`, a `bool`, or a mapping with "
            f"'fwd'/'bwd' keys, got {type(enable_pallas)}."
        )
    if (use_fwd or use_bwd) and jax.default_backend() != "gpu":
        raise RuntimeError(
            "`enable_pallas` requires a CUDA GPU (the Pallas/Triton backend), "
            f"but the current JAX default backend is {jax.default_backend()!r}. "
            "Pass `enable_pallas=False` (or leave it unset, along with "
            "`CRYOJAX_ENABLE_PALLAS`) to use the pure-JAX backend instead."
        )
    return use_fwd, use_bwd


def _is_zero(tangent) -> bool:
    return isinstance(tangent, SymbolicZero)


def _add(*terms):
    """Sum of the non-`None` terms; `None` when there are none."""
    terms = [t for t in terms if t is not None]
    return reduce(operator.add, terms) if terms else None


def _mul(a, b):
    return None if a is None or b is None else a * b


def _neg(x):
    return None if x is None else -x


def _tprod2(a, b, ta, tb):
    """Tangent of `a b` by the product rule, `None` tangents dropped."""
    return _add(_mul(ta, b), _mul(tb, a))


def _tprod3(a, b, c, ta, tb, tc):
    """Tangent of `a b c` by the product rule, `None` tangents dropped."""
    return _add(_mul(ta, b * c), _mul(tb, a * c), _mul(tc, a * b))


def _jvp_over_nonzero(fn, args, tangents):
    """`jax.jvp(fn, args, tangents)` over only the arguments whose tangent is not
    `None`; the rest are closed over as constants, so no zero term is traced."""
    nonzero = [k for k, t in enumerate(tangents) if t is not None]

    def fn_nonzero(*nonzero_vals):
        full = list(args)
        for k, v in zip(nonzero, nonzero_vals):
            full[k] = v
        return fn(*full)

    return jax.jvp(
        fn_nonzero, tuple(args[k] for k in nonzero), tuple(tangents[k] for k in nonzero)
    )


def _like_operands(tangents, operand, dtype):
    """A stack of directions laid out like the kernel operand it pairs with: `tangents`
    has a leading direction axis and the result is `(n_directions, *operand.shape)`,
    flattened if the operand is, broadcast if the operand is."""
    tangents = jnp.asarray(tangents, dtype=dtype)
    n_directions = tangents.shape[0]
    if tangents.ndim - 1 > operand.ndim:
        tangents = tangents.reshape(n_directions, -1)
    lead = (1,) * (operand.ndim - (tangents.ndim - 1))
    tangents = tangents.reshape(n_directions, *lead, *tangents.shape[1:])
    return jnp.broadcast_to(tangents, (n_directions, *operand.shape))


#: Directions one tangent launch handles; a larger batch of directions is chunked across
#: launches, never refused. The direction loop is unrolled inside the neighbour loop --
#: each direction adds a per-point tangent block and, in the gather, its own accumulators
#: of `block_size` lanes -- so this is a register budget, not a capability. Measured on
#: an RTX PRO 5000 (float64, 100k points, 192^2, n_spread 7): the cost per added
#: direction is flat from 2 to 12 directions and rises ~1.8x at 16, so 12 sits below
#: the knee with float32 having more headroom.
_MAX_JVP_DIRECTIONS = 12


def _read_directions(refs, nonzero, names, n_directions):
    """Per-direction dicts `{name: value | None}` of the nonzero tangents, read once
    from their refs: a per-point ref is `(n_directions, block)`, a scalar ref
    `(n_directions,)`."""
    nonzero_names = [name for name, is_nonzero in zip(names, nonzero) if is_nonzero]
    directions = [dict.fromkeys(names) for _ in range(n_directions)]
    for name, ref in zip(nonzero_names, refs):
        for d in range(n_directions):
            directions[d][name] = ref[d]
    return directions


def _direction_specs(
    nonzero, names, block_size, n_directions, scalar_names, image_names, n_pixels
):
    """One `BlockSpec` per nonzero tangent, matching `_read_directions`' layout."""
    specs = []
    for name, is_nonzero in zip(names, nonzero):
        if not is_nonzero:
            continue
        if name in scalar_names:
            specs.append(pl.BlockSpec((n_directions,), lambda p: (0,)))
        elif name in image_names:
            specs.append(pl.BlockSpec((n_directions * n_pixels,), lambda p: (0,)))
        else:
            specs.append(pl.BlockSpec((n_directions, block_size), lambda p: (0, p)))
    return specs


def _tangent_operands(nonzero_tangents, nonzero_operands, nonzero_names, dtype):
    """Kernel operands for a stack of directions: scalars `(n_directions,)`, images
    flattened, everything else `(n_directions, *operand.shape)`."""
    n_directions = nonzero_tangents[0].shape[0]
    operands = []
    for t, operand, name in zip(nonzero_tangents, nonzero_operands, nonzero_names):
        if name == "pix":
            operands.append(jnp.asarray(t, dtype=dtype).reshape(n_directions))
        elif name == "g":
            operands.append(_like_operands(t, operand, dtype).reshape(-1))
        else:
            operands.append(_like_operands(t, operand, dtype))
    return tuple(operands)


def _chunk_directions(impl, n_directions, tangents):
    """Run `impl` on chunks of at most `_MAX_JVP_DIRECTIONS` directions and
    concatenate the tangents along the direction axis."""
    outs = [
        impl(tuple(x[start : start + _MAX_JVP_DIRECTIONS] for x in tangents))
        for start in range(0, n_directions, _MAX_JVP_DIRECTIONS)
    ]
    if len(outs) == 1:
        return outs[0]
    return jax.tree.map(lambda *ts: jnp.concatenate(ts, axis=0), *outs)


def _batchable(single, n_primals, impl):
    """`single` (one direction) as a `custom_vmap` function whose batching rule sends a
    `jax.vmap` over tangents at ONE primal point to `impl` (all directions in one
    kernel, chunked by `_MAX_JVP_DIRECTIONS`). A `vmap` that also batches the primals --
    different point sets -- has nothing to share and takes Pallas's generic batching."""
    fn = custom_vmap(single)

    @fn.def_vmap
    def _rule(axis_size, in_batched, *args):
        primals, tangents = args[:n_primals], args[n_primals:]
        batched = lambda out: (out, jax.tree.map(lambda _: True, out))
        if any(in_batched[:n_primals]):
            in_axes = tuple(0 if b else None for b in in_batched)
            return batched(jax.vmap(single, in_axes=in_axes)(*args))
        tangents = tuple(
            t if b else jnp.broadcast_to(t, (axis_size, *jnp.shape(t)))
            for t, b in zip(tangents, in_batched[n_primals:])
        )
        return batched(
            _chunk_directions(lambda ts: impl(primals, ts), axis_size, tangents)
        )

    return fn
