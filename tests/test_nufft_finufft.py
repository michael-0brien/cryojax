"""The native NUFFT against finufft, the ground truth, and against exact identities."""

import warnings

import cryojax.ndimage as im
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage._nufft import api as nufft_api
from cryojax.ndimage._nufft.kernel import resolve_upsample_factor


finufft = pytest.importorskip("finufft")

try:
    import jax_finufft
except ModuleNotFoundError:
    jax_finufft = None


SHAPES = [(64, 64), (5, 8), (7, 6), (16, 15, 14)]
SIGMAS = [1.25, 2.0]
PIXEL_SIZE = 0.7
NUM_POINTS = 200


#
# References
#
def _points(shape, num, seed):
    """Positions `(x, y[, z])` spread over the whole box, in units of `PIXEL_SIZE`."""
    n = np.array(shape[::-1])
    return np.random.default_rng(seed).uniform(-n / 2, n / 2, (num, n.size)) * PIXEL_SIZE


def _random_spectrum(shape, seed, batch=()):
    rng = np.random.default_rng(seed)
    half = (*batch, *shape[:-1], shape[-1] // 2 + 1)
    return rng.normal(size=half) + 1j * rng.normal(size=half)


def _grid_points(shape):
    """Pixel centres, origin at `N//2`, as `(x, y[, z])` in C order of `shape`."""
    mesh = np.meshgrid(*[np.arange(n) - n // 2 for n in shape], indexing="ij")
    return np.stack([m.ravel() for m in mesh[::-1]], axis=-1).astype(float)


def _half_grid_frequencies(shape):
    """Integer frequencies of the rfft half grid, one broadcastable array per axis."""
    freqs = [np.fft.ifftshift(np.arange(n) - n // 2) for n in shape[:-1]]
    freqs.append(np.arange(shape[-1] // 2 + 1))
    return np.meshgrid(*freqs, indexing="ij", sparse=True)


def _hermitian_weights(shape):
    weights = np.full(shape[-1] // 2 + 1, 2.0)
    weights[0] = 1.0
    if shape[-1] % 2 == 0:
        weights[-1] = 1.0
    return weights


def _angles(positions, shape, pixel_size=PIXEL_SIZE):
    n = np.array(shape[::-1])
    angles = 2 * np.pi * (n // 2 + np.asarray(positions, float) / pixel_size) / n
    return [np.ascontiguousarray(angles[:, a]) for a in reversed(range(len(shape)))]


def _odd_last_axis(shape):
    """`shape` with the last axis padded to odd size, holding `0 .. n_x // 2` first."""
    return (*shape[:-1], shape[-1] + 1 - shape[-1] % 2)


def _finufft_type1(
    positions, strengths, shape, modes, eps=1e-14, upsampfac=None, dtype=np.float64
):
    transform = finufft.nufft2d1 if len(shape) == 2 else finufft.nufft3d1
    kwargs = {} if upsampfac is None else {"upsampfac": upsampfac}
    return transform(
        *[angles.astype(dtype) for angles in _angles(positions, shape)],
        np.asarray(strengths, np.result_type(dtype, 1j)),
        n_modes=modes,
        eps=eps,
        isign=-1,
        modeord=1,
        **kwargs,
    )


def _type1_reference(positions, strengths, shape):
    """finufft at the half grid's own frequencies. `convert_fftn_to_rfftn` of the
    `shape` output differs where an even full axis's `-n/2` meets the `+n_x/2` column:
    the full grid has no `+n/2` row to conjugate, so it returns `(+n/2, +n_x/2)`."""
    full = _finufft_type1(positions, strengths, shape, _odd_last_axis(shape))
    return full[..., : shape[-1] // 2 + 1]


def _hermitian_grid(spectrum, shape, modes=None):
    """`G[k] = w_k F[k] / 2`, `G[-k] = conj(G[k])`, padded to `n + 1` on even axes
    unless `modes` is given."""
    padded = modes or tuple(n + 1 - n % 2 for n in shape)
    half = spectrum * _hermitian_weights(shape) / 2
    k = [np.broadcast_to(f, half.shape) for f in _half_grid_frequencies(shape)]
    grid = np.zeros(padded, dtype=complex)
    np.add.at(grid, tuple(f % p for f, p in zip(k, padded)), half)
    np.add.at(grid, tuple(-f % p for f, p in zip(k, padded)), np.conj(half))
    return grid


def _finufft_type2(positions, grid, shape, eps=1e-14, upsampfac=None, dtype=np.float64):
    transform = finufft.nufft2d2 if len(shape) == 2 else finufft.nufft3d2
    kwargs = {} if upsampfac is None else {"upsampfac": upsampfac}
    values = transform(
        *[angles.astype(dtype) for angles in _angles(positions, shape)],
        grid.astype(np.result_type(dtype, 1j)),
        eps=eps,
        isign=1,
        modeord=1,
        **kwargs,
    )
    return values.real


def _type2_reference(positions, spectrum, shape):
    grid = _hermitian_grid(spectrum, shape)
    return _finufft_type2(positions, grid, shape) / np.prod(shape)


def _finufft_half_grid(positions, strengths, shape, **kwargs):
    """finufft on `shape`'s own modes, so on native's fine grid, as the rfft half grid,
    and where that determines it: an even last axis's `+n_x/2` column is
    `conj(F[-k])`, unknown where `-k` is another even axis's `+n/2`."""
    full = _finufft_type1(positions, strengths, shape, shape, **kwargs)
    half = full[..., : shape[-1] // 2 + 1].copy()
    known = np.ones(half.shape, bool)
    if shape[-1] % 2 == 0:
        axes = tuple(range(len(shape)))
        negated = np.conj(np.roll(np.flip(full, axes), 1, axes))
        half[..., -1] = negated[..., shape[-1] // 2]
        for k, n in zip(_half_grid_frequencies(shape)[:-1], shape[:-1]):
            if n % 2 == 0:
                known[..., -1] &= (k != -(n // 2))[..., 0]
    return half, known


def _without_nyquist(spectrum, shape):
    """`spectrum` zeroed on even axes' `n/2` modes, so that its Hermitian sum lies in
    finufft's `shape` modes, on native's fine grid."""
    keep = np.ones(spectrum.shape, bool)
    for k, n in zip(_half_grid_frequencies(shape), shape):
        if n % 2 == 0:
            keep &= np.abs(k) != n // 2
    return spectrum * keep


def _finufft_values(positions, spectrum, shape, **kwargs):
    """finufft type 2 of a `_without_nyquist` spectrum on `shape`'s own modes."""
    grid = _hermitian_grid(spectrum, shape, modes=shape)
    return _finufft_type2(positions, grid, shape, **kwargs) / np.prod(shape)


def _finufft_half_sum(positions, spectrum, shape):
    """`Re sum_{k in half grid} F[k] exp(+2 pi i k t / n)`, no weights."""
    transform = finufft.nufft2d2 if len(shape) == 2 else finufft.nufft3d2
    grid = np.zeros(_odd_last_axis(shape), dtype=complex)
    grid[..., : shape[-1] // 2 + 1] = spectrum
    values = transform(*_angles(positions, shape), grid, eps=1e-14, isign=1, modeord=1)
    return values.real


def _relative_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


#
# Agreement with finufft
#
# Native and finufft 2.5.1 (whose rules are 2.6's in float64 up to width 16) at one `eps`
# and `upsample_factor` are one approximation: native is within RATIO of finufft's own
# error of it. Measured: at most 0.047 (3D, eps 1e-9, sigma 1.4713)
RATIO = 0.1
# At widths 15-16 both errors are mostly the tap polynomials' truncation at 0.05 eps,
# which fit rounding decides (3D: 15 coefficients here, 17 in finufft 2.5.1)
_TRUNCATION_BOUND = pytest.mark.xfail(
    strict=True, reason="measured ratio 0.22-0.41 in 2D, 2.5-2.9 in 3D"
)
FLOAT64_CASES = [
    pytest.param(
        shape,
        eps,
        sigma,
        marks=_TRUNCATION_BOUND if (eps, sigma) == (1e-9, 1.25) else (),
    )
    for shape in SHAPES
    for eps in (1e-3, 1e-6, 1e-9)
    for sigma in (1.25, 1.4713, 2.0)
]
# float32 where finufft 2.5.1's widths are 2.6's (sigma 1.25, or widths up to 8), with
# 3D at sigma 1.25 and eps 1e-6, where the float32 guard narrows the width from 11 to 8
FLOAT32_CASES = [
    (shape, eps, sigma)
    for shape in [*SHAPES, (64, 64, 64)]
    for eps in (1e-3, 1e-4, 1e-6)
    for sigma in (1.25, 2.0)
]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("shape, eps, sigma", FLOAT64_CASES)
def test_type1_is_finufft_approximation(shape, eps, sigma):
    positions = _points(shape, NUM_POINTS, seed=0)
    strengths = np.random.default_rng(1).normal(size=NUM_POINTS)
    exact, known = _finufft_half_grid(positions, strengths, shape)
    approx, _ = _finufft_half_grid(positions, strengths, shape, eps=eps, upsampfac=sigma)
    native = im.dispatch_nufft1(
        jnp.asarray(positions),
        jnp.asarray(strengths),
        shape,
        pixel_size=PIXEL_SIZE,
        eps=eps,
        upsample_factor=sigma,
    )
    assert native.dtype == jnp.complex128
    assert native.shape == exact.shape
    native, exact, approx = np.asarray(native)[known], exact[known], approx[known]
    assert _relative_error(native, approx) <= RATIO * _relative_error(approx, exact)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("shape, eps, sigma", FLOAT64_CASES)
def test_type2_is_finufft_approximation(shape, eps, sigma):
    positions = _points(shape, NUM_POINTS, seed=0)
    spectrum = _without_nyquist(_random_spectrum(shape, seed=1), shape)
    exact = _finufft_values(positions, spectrum, shape)
    approx = _finufft_values(positions, spectrum, shape, eps=eps, upsampfac=sigma)
    native = im.dispatch_nufft2(
        jnp.asarray(positions),
        jnp.asarray(spectrum),
        shape,
        pixel_size=PIXEL_SIZE,
        eps=eps,
        upsample_factor=sigma,
    )
    assert native.dtype == jnp.float64
    assert native.shape == (NUM_POINTS,)
    assert _relative_error(native, approx) <= RATIO * _relative_error(approx, exact)


@pytest.mark.parametrize("shape, eps, sigma", FLOAT32_CASES)
def test_float32_within_twice_finufft_single(shape, eps, sigma):
    """Both against finufft in float64 at eps 1e-12, on the same points; one warning per
    limit `nufft_accuracy_limits` finds failed, per transform."""
    positions = _points(shape, NUM_POINTS, seed=0).astype(np.float32)
    strengths = np.random.default_rng(1).normal(size=NUM_POINTS).astype(np.float32)
    spectrum = _without_nyquist(_random_spectrum(shape, seed=2), shape)
    spectrum = spectrum.astype(np.complex64)
    exact, known = _finufft_half_grid(positions, strengths, shape, eps=1e-12)
    single, _ = _finufft_half_grid(
        positions, strengths, shape, eps=eps, upsampfac=sigma, dtype=np.float32
    )
    exact_values = _finufft_values(positions, spectrum, shape, eps=1e-12)
    single_values = _finufft_values(
        positions, spectrum, shape, eps=eps, upsampfac=sigma, dtype=np.float32
    )
    limits = im.nufft_accuracy_limits(
        shape, eps=eps, upsample_factor=sigma, dtype=np.float32
    )
    num_failed = (eps < limits.rounding_floor) + (sigma < limits.min_upsample_factor)
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=eps, upsample_factor=sigma)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = im.dispatch_nufft1(
            jnp.asarray(positions), jnp.asarray(strengths), shape, **kwargs
        )
        native_values = im.dispatch_nufft2(
            jnp.asarray(positions), jnp.asarray(spectrum), shape, **kwargs
        )
    assert [w.category for w in caught] == [UserWarning] * (2 * num_failed)
    assert native.dtype == jnp.complex64 and native_values.dtype == jnp.float32
    native, exact, single = np.asarray(native)[known], exact[known], single[known]
    assert _relative_error(native, exact) <= 2 * _relative_error(single, exact)
    assert _relative_error(native_values, exact_values) <= 2 * _relative_error(
        single_values, exact_values
    )


@pytest.mark.parametrize("shape", [(8, 6), (7, 9), (6, 5, 4), (5, 7, 6)])
def test_type2_on_grid_equals_irfftn(shape):
    spectrum = _random_spectrum(shape, seed=0)
    values = im.dispatch_nufft2(
        jnp.asarray(_grid_points(shape)),
        jnp.asarray(spectrum),
        shape,
        eps=1e-12,
        upsample_factor=2.0,
    )
    expected = np.fft.irfftn(spectrum, s=shape, axes=range(len(shape))).ravel()
    np.testing.assert_allclose(values, expected, atol=1e-10 * np.abs(expected).max())


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("shape", SHAPES)
def test_type2_is_weighted_adjoint(shape, sigma):
    positions = jnp.asarray(_points(shape, NUM_POINTS, seed=0))
    values = np.random.default_rng(1).normal(size=NUM_POINTS)
    spectrum = _random_spectrum(shape, seed=2)
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=1e-6, upsample_factor=sigma)
    lhs = np.sum(
        values * im.dispatch_nufft2(positions, jnp.asarray(spectrum), shape, **kwargs)
    )
    type1 = np.asarray(
        im.dispatch_nufft1(positions, jnp.asarray(values), shape, **kwargs)
    )
    rhs = np.real(np.sum(_hermitian_weights(shape) * np.conj(type1) * spectrum))
    np.testing.assert_allclose(lhs, rhs / np.prod(shape), rtol=1e-10)


#
# Edge cases
#
@pytest.mark.parametrize("shape", [(5, 8), (16, 15, 14)])
def test_type1_periodic_in_positions(shape):
    positions = _points(shape, NUM_POINTS, seed=0)
    strengths = jnp.asarray(np.random.default_rng(1).normal(size=NUM_POINTS))
    spectrum = jnp.asarray(_random_spectrum(shape, seed=2))
    periods = np.random.default_rng(3).integers(-3, 4, positions.shape)
    shifted = positions + periods * np.array(shape[::-1]) * PIXEL_SIZE
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=1e-9, upsample_factor=1.25)
    type1, type1_shifted = (
        im.dispatch_nufft1(jnp.asarray(p), strengths, shape, **kwargs)
        for p in (positions, shifted)
    )
    type2, type2_shifted = (
        im.dispatch_nufft2(jnp.asarray(p), spectrum, shape, **kwargs)
        for p in (positions, shifted)
    )
    assert _relative_error(type1_shifted, type1) <= 1e-10
    assert _relative_error(type2_shifted, type2) <= 1e-10


@pytest.mark.parametrize(
    "position_dtype, strength_dtype, expected",
    [
        (jnp.float32, jnp.float32, jnp.complex64),
        (jnp.float32, jnp.float64, jnp.complex128),
        (jnp.float64, jnp.float32, jnp.complex128),
        (jnp.float64, jnp.float64, jnp.complex128),
    ],
)
def test_type1_dtypes(position_dtype, strength_dtype, expected):
    shape = (8, 6)
    positions = jnp.asarray(_points(shape, 10, seed=0), dtype=position_dtype)
    strengths = jnp.ones(10, dtype=strength_dtype)
    spectrum = im.dispatch_nufft1(positions, strengths, shape, eps=1e-4)
    assert spectrum.dtype == expected
    values = im.dispatch_nufft2(positions, spectrum, shape, eps=1e-4)
    assert values.dtype == jnp.finfo(expected).dtype


def test_type1_rejects_complex_strengths():
    with pytest.raises(ValueError, match=r"dispatch_nufft1[\s\S]*strengths"):
        im.dispatch_nufft1(jnp.zeros((3, 2)), jnp.ones(3, dtype=complex), (6, 6))


@pytest.mark.parametrize("shape", [(8, 6), (5, 7, 6)])
def test_type1_empty_and_single_point(shape):
    ndim, half = len(shape), (*shape[:-1], shape[-1] // 2 + 1)
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=1e-9)
    empty = im.dispatch_nufft1(jnp.zeros((0, ndim)), jnp.zeros(0), shape, **kwargs)
    assert empty.shape == half
    assert np.all(np.asarray(empty) == 0)
    no_values = im.dispatch_nufft2(
        jnp.zeros((0, ndim)), jnp.asarray(_random_spectrum(shape, 0)), shape, **kwargs
    )
    assert no_values.shape == (0,)

    position = np.array([[0.3, -1.2, 0.7][:ndim]])
    spectrum = im.dispatch_nufft1(
        jnp.asarray(position), jnp.asarray([2.5]), shape, **kwargs
    )
    t = np.array(shape[::-1]) // 2 + position[0] / PIXEL_SIZE
    phase = sum(
        k * t_a / n for k, t_a, n in zip(_half_grid_frequencies(shape), t[::-1], shape)
    )
    plane_wave = 2.5 * np.exp(-2j * np.pi * phase)
    assert _relative_error(spectrum, plane_wave) <= 3e-9


@pytest.mark.parametrize("shape", [(7, 6), (16, 15, 14)])
def test_batched_strengths_equal_unbatched(shape):
    positions = jnp.asarray(_points(shape, NUM_POINTS, seed=0))
    strengths = jnp.asarray(np.random.default_rng(1).normal(size=(3, NUM_POINTS)))
    spectra = jnp.asarray(_random_spectrum(shape, seed=2, batch=(3,)))
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=1e-9, upsample_factor=1.25)
    batched = im.dispatch_nufft1(positions, strengths, shape, **kwargs)
    assert batched.shape == (3, *spectra.shape[1:])
    for r in range(3):
        unbatched = im.dispatch_nufft1(positions, strengths[r], shape, **kwargs)
        assert _relative_error(batched[r], unbatched) <= 1e-12
    batched = im.dispatch_nufft2(positions, spectra, shape, **kwargs)
    assert batched.shape == (3, NUM_POINTS)
    for r in range(3):
        unbatched = im.dispatch_nufft2(positions, spectra[r], shape, **kwargs)
        assert _relative_error(batched[r], unbatched) <= 1e-12


#
# Derivatives
#
# A derivative of the approximation: its aliased terms oscillate faster than the signal,
# and measure 5-13 eps at sigma = 2
VJP_CASES = [
    ((64, 64), 1.25, 1e-6),
    ((5, 8), 1.25, 1e-9),
    ((5, 8), 2.0, 1e-9),
    ((7, 6), 2.0, 1e-9),
    ((16, 15, 14), 1.25, 1e-6),
    ((16, 15, 14), 2.0, 1e-9),
]


def _frequency_factors(shape):
    """`2 pi i k_a / (n_a pixel_size)` on the half grid, in `(x, y[, z])` order."""
    freqs = _half_grid_frequencies(shape)
    return [
        2j * np.pi * freqs[axis] / (shape[axis] * PIXEL_SIZE)
        for axis in reversed(range(len(shape)))
    ]


@pytest.mark.parametrize("shape, sigma, eps", VJP_CASES)
def test_type1_position_vjp_identity(shape, sigma, eps):
    positions = _points(shape, NUM_POINTS, seed=0)
    strengths = np.random.default_rng(1).normal(size=NUM_POINTS)
    cotangent = _random_spectrum(shape, seed=2)

    def loss(positions, strengths):
        spectrum = im.dispatch_nufft1(
            positions,
            strengths,
            shape,
            pixel_size=PIXEL_SIZE,
            eps=eps,
            upsample_factor=sigma,
        )
        return jnp.real(jnp.sum(jnp.conj(cotangent) * spectrum))

    positions_bar, strengths_bar = jax.grad(loss, argnums=(0, 1))(
        jnp.asarray(positions), jnp.asarray(strengths)
    )
    expected_positions_bar = np.stack(
        [
            strengths * _finufft_half_sum(positions, factor * cotangent, shape)
            for factor in _frequency_factors(shape)
        ],
        axis=-1,
    )
    expected_strengths_bar = _finufft_half_sum(positions, cotangent, shape)
    assert _relative_error(positions_bar, expected_positions_bar) <= 20 * eps
    assert _relative_error(strengths_bar, expected_strengths_bar) <= 3 * eps


@pytest.mark.parametrize("shape, sigma, eps", VJP_CASES)
def test_type2_position_vjp_identity(shape, sigma, eps):
    positions = _points(shape, NUM_POINTS, seed=0)
    spectrum = _random_spectrum(shape, seed=1)
    cotangent = np.random.default_rng(2).normal(size=NUM_POINTS)

    def loss(positions):
        values = im.dispatch_nufft2(
            positions,
            jnp.asarray(spectrum),
            shape,
            pixel_size=PIXEL_SIZE,
            eps=eps,
            upsample_factor=sigma,
        )
        return jnp.sum(cotangent * values)

    positions_bar = jax.grad(loss)(jnp.asarray(positions))
    expected = np.stack(
        [
            cotangent * _type2_reference(positions, factor * spectrum, shape)
            for factor in _frequency_factors(shape)
        ],
        axis=-1,
    )
    assert _relative_error(positions_bar, expected) <= 20 * eps


@pytest.mark.parametrize("shape", [(12, 10), (6, 5, 4)])
def test_hvp_runs_and_is_symmetric(shape):
    rng = np.random.default_rng(0)
    num = 30
    params = (
        jnp.asarray(_points(shape, num, seed=1)),
        jnp.asarray(rng.normal(size=num)),
        jnp.asarray(PIXEL_SIZE),
    )
    spectrum = jnp.asarray(_random_spectrum(shape, seed=2))
    weights = jnp.asarray(rng.uniform(size=spectrum.shape))
    kwargs = dict(eps=1e-9, upsample_factor=2.0)

    def loss(params):
        positions, strengths, pixel_size = params
        type1 = im.dispatch_nufft1(
            positions, strengths, shape, pixel_size=pixel_size, **kwargs
        )
        type2 = im.dispatch_nufft2(
            positions, spectrum, shape, pixel_size=pixel_size, **kwargs
        )
        return jnp.sum(weights * jnp.abs(type1) ** 2) + jnp.sum(strengths * type2**2)

    def hvp(direction):
        return jax.jvp(jax.grad(loss), (params,), (direction,))[1]

    def dot(a, b):
        return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

    v1 = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape)), params)
    v2 = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape)), params)
    v1_h_v2, v2_h_v1 = dot(v1, hvp(v2)), dot(v2, hvp(v1))
    assert np.isfinite(v1_h_v2)
    np.testing.assert_allclose(v1_h_v2, v2_h_v1, rtol=1e-8)


#
# upsample_factor="auto"
#
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_auto_is_static(monkeypatch, name):
    """`"auto"` is the default and resolves from static shapes, `eps` and dtype: one
    trace for new values, no leaked tracers, the kernel of the resolved factor, here the
    smallest that reaches `eps` at its width, plus finufft's 1e-3 pad."""
    shape = (32, 30, 28)
    positions = jnp.asarray(_points(shape, NUM_POINTS, seed=0), jnp.float32)
    if name == "dispatch_nufft1":
        second = jnp.asarray(
            np.random.default_rng(1).normal(size=NUM_POINTS), jnp.float32
        )
    else:
        second = jnp.asarray(_random_spectrum(shape, seed=1), jnp.complex64)
    sigma = resolve_upsample_factor(
        "auto", n_points=NUM_POINTS, shape=shape, eps=1e-5, dtype=np.float32, pallas=False
    )
    limits = im.nufft_accuracy_limits(
        shape, eps=1e-5, upsample_factor=sigma, dtype=np.float32
    )
    assert sigma == pytest.approx(limits.min_upsample_factor + 1e-3)
    assert sigma > 1.251
    kernel_factors = []

    def make_pswf_kernel(eps, upsample_factor, ndim, dtype):
        kernel_factors.append(upsample_factor)
        return make_pswf_kernel_unpatched(eps, upsample_factor, ndim, dtype)

    make_pswf_kernel_unpatched = nufft_api.make_pswf_kernel
    monkeypatch.setattr(nufft_api, "make_pswf_kernel", make_pswf_kernel)
    transform = getattr(im, name)

    @jax.jit
    def run(positions, second):
        return transform(
            positions, second, shape, pixel_size=PIXEL_SIZE, eps=1e-5, enable_pallas=False
        )

    with jax.checking_leaks():
        run(positions, second)
        run(positions + 0.1, 2 * second)
    assert kernel_factors == [sigma]


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_auto_uses_pure_jax_costs(monkeypatch, name):
    backends = []

    def resolve_upsample_factor(upsample_factor, **kwargs):
        backends.append(kwargs["pallas"])
        return resolve_upsample_factor_unpatched(upsample_factor, **kwargs)

    resolve_upsample_factor_unpatched = nufft_api.resolve_upsample_factor
    monkeypatch.setattr(nufft_api, "resolve_upsample_factor", resolve_upsample_factor)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    getattr(im, name)(jnp.zeros((3, 2)), second, (8, 6), enable_pallas=False)
    assert backends == [False]


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_upsample_factor_other_string_raises(name):
    shape = (8, 6)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    with pytest.raises(ValueError, match=rf"{name}[\s\S]*upsample_factor[\s\S]*fast"):
        getattr(im, name)(jnp.zeros((3, 2)), second, shape, upsample_factor="fast")


#
# Errors and warnings
#
def _call(name, shape, dtype=np.float64, **kwargs):
    """`name` on 3 points of `shape` in `dtype`."""
    positions = jnp.zeros((3, len(shape)), dtype)
    if name == "dispatch_nufft1":
        second = jnp.ones(3, dtype)
    else:
        half = (*shape[:-1], shape[-1] // 2 + 1)
        second = jnp.ones(half, jnp.result_type(dtype, 1j))
    return getattr(im, name)(positions, second, shape, **kwargs)


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
@pytest.mark.parametrize("upsample_factor", [1.0, 0.5])
def test_upsample_factor_at_most_one_raises(name, upsample_factor):
    with pytest.raises(
        ValueError, match=rf"{name}[\s\S]*upsample_factor[\s\S]*{upsample_factor}"
    ):
        _call(name, (8, 6), upsample_factor=upsample_factor)


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_eps_at_most_zero_raises(name, eps):
    with pytest.raises(ValueError, match=rf"{name}[\s\S]*eps[\s\S]*positive"):
        _call(name, (8, 6), eps=eps)


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
@pytest.mark.parametrize("upsample_factor", [1.1, 3.5])
def test_upsample_factor_outside_useful_range_warns(name, upsample_factor):
    """finufft include/finufft/makeplan.hpp:147; `eps` is met at both."""
    with pytest.warns(
        UserWarning,
        match=rf"{name}[\s\S]*upsample_factor = {upsample_factor}[\s\S]*1.15, 3.0",
    ) as record:
        _call(name, (8, 6), eps=1e-6, upsample_factor=upsample_factor)
    assert len(record) == 1


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_eps_below_eps_mach_warns_and_runs(monkeypatch, name):
    """One warning, though the floor and sigma limit fail at `eps_mach` too."""
    kernel_eps = []

    def make_pswf_kernel(eps, upsample_factor, ndim, dtype):
        kernel_eps.append(eps)
        return make_pswf_kernel_unpatched(eps, upsample_factor, ndim, dtype)

    make_pswf_kernel_unpatched = nufft_api.make_pswf_kernel
    monkeypatch.setattr(nufft_api, "make_pswf_kernel", make_pswf_kernel)
    with pytest.warns(
        UserWarning, match=rf"{name}[\s\S]*eps = 1e-09[\s\S]*float32[\s\S]*1.2e-07"
    ) as record:
        result = _call(name, (8, 6), np.float32, eps=1e-9)
    assert len(record) == 1
    assert kernel_eps == [float(jnp.finfo(jnp.float32).eps)]
    assert np.all(np.isfinite(np.asarray(result)))


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_warns_below_rounding_floor(name):
    """finufft include/finufft/setpts.hpp:22; the floor, 6.04e-5, rounded up."""
    with pytest.warns(
        UserWarning,
        match=rf"{name}[\s\S]*eps = 1e-06[\s\S]*\(1024, 1024\)[\s\S]*float32"
        r"[\s\S]*6\.1e-05[\s\S]*jax_enable_x64",
    ) as record:
        _call(name, (1024, 1024), np.float32, eps=1e-6)
    assert len(record) == 1


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_warns_below_min_upsample_factor(name):
    """finufft include/finufft/setpts.hpp:22; the width clipped to 16 needs 1.2675."""
    with pytest.warns(
        UserWarning,
        match=rf"{name}[\s\S]*upsample_factor = 1.25[\s\S]*eps = 1e-10[\s\S]*"
        r"\(64, 64\)[\s\S]*float64[\s\S]*at least 1\.27[\s\S]*auto",
    ) as record:
        _call(name, (64, 64), eps=1e-10, upsample_factor=1.25)
    assert len(record) == 1


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_float32_guard_warns_once(name):
    """The guard narrows the width from 11 to 8, below rule 1's, so the sigma limit
    fails: at width 8, eps 1e-6 needs 2.0397."""
    with pytest.warns(
        UserWarning,
        match=rf"{name}[\s\S]*upsample_factor = 1.25[\s\S]*\(8, 8, 8\)[\s\S]*float32"
        r"[\s\S]*at least 2\.04",
    ) as record:
        _call(name, (8, 8, 8), np.float32, eps=1e-6, upsample_factor=1.25)
    assert len(record) == 1


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_auto_unreachable_warns(monkeypatch, name):
    """No sigma up to 2.5 reaches 3e-7 in float32, though the floor (2.4e-7) does."""
    kernel_factors = []

    def make_pswf_kernel(eps, upsample_factor, ndim, dtype):
        kernel_factors.append(upsample_factor)
        return make_pswf_kernel_unpatched(eps, upsample_factor, ndim, dtype)

    make_pswf_kernel_unpatched = nufft_api.make_pswf_kernel
    monkeypatch.setattr(nufft_api, "make_pswf_kernel", make_pswf_kernel)
    with pytest.warns(
        UserWarning, match=rf"{name}[\s\S]*eps = 3e-07[\s\S]*float32[\s\S]*2.5"
    ) as record:
        result = _call(name, (4, 4), np.float32, eps=3e-7)
    assert len(record) == 1
    assert kernel_factors == [2.5]
    assert np.all(np.isfinite(np.asarray(result)))


@pytest.mark.skipif(jax_finufft is None, reason="jax-finufft not installed")
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "eps, upsample_factor", [(1e-10, 1.25), (1e-13, "auto"), (1e-6, 1.1)]
)
def test_jax_finufft_branch_runs_no_native_checks(monkeypatch, eps, upsample_factor):
    """finufft takes widths past the native cap, and none of the native warnings."""
    monkeypatch.setattr(nufft_api, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    shape = (16, 15)
    positions = _points(shape, NUM_POINTS, seed=0)
    strengths = np.random.default_rng(1).normal(size=NUM_POINTS)
    spectrum = _random_spectrum(shape, seed=2)
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=eps, upsample_factor=upsample_factor)
    type1 = im.dispatch_nufft1(
        jnp.asarray(positions), jnp.asarray(strengths), shape, **kwargs
    )
    type2 = im.dispatch_nufft2(
        jnp.asarray(positions), jnp.asarray(spectrum), shape, **kwargs
    )
    reference = _type1_reference(positions, strengths, shape)
    assert _relative_error(type1, reference) <= 1e-9
    reference = _type2_reference(positions, spectrum, shape)
    assert _relative_error(type2, reference) <= 1e-9


#
# jax-finufft backend
#
@pytest.mark.skipif(jax_finufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize("fft_ordered", [True, False])
@pytest.mark.parametrize("shape", [(64, 64), (7, 6), (16, 15, 14)])
def test_jax_finufft_branch_agrees(monkeypatch, shape, fft_ordered):
    """Values and every gradient, with cryoJAX's `Opts(modeord=True)` or user `opts`
    in finufft's centred order."""
    from jax_finufft.options import NestedOpts, Opts

    positions = jnp.asarray(_points(shape, NUM_POINTS, seed=0))
    strengths = jnp.asarray(np.random.default_rng(1).normal(size=(2, NUM_POINTS)))
    spectrum = jnp.asarray(_random_spectrum(shape, seed=2, batch=(2,)))
    kwargs = dict(pixel_size=PIXEL_SIZE, eps=1e-9, upsample_factor=2.0)
    if not fft_ordered:
        opts = Opts(upsampfac=2.0)
        kwargs["options"] = {"opts": NestedOpts(forward=opts, backward=opts)}

    def loss(positions, strengths, spectrum):
        type1 = im.dispatch_nufft1(positions, strengths, shape, **kwargs)
        type2 = im.dispatch_nufft2(positions, spectrum, shape, **kwargs)
        value = jnp.sum(jnp.abs(type1) ** 2) + jnp.sum(type2**2)
        return value, (type1, type2)

    def run():
        grads, outputs = jax.grad(loss, argnums=(0, 1, 2), has_aux=True)(
            positions, strengths, spectrum
        )
        return (*outputs, *grads)

    monkeypatch.setattr(nufft_api, "CRYOJAX_FINUFFT_BACKEND", "native")
    native = run()
    monkeypatch.setattr(nufft_api, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    for expected, actual in zip(native, run()):
        assert _relative_error(actual, expected) <= 1e-7


@pytest.mark.skipif(jax_finufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_jax_finufft_does_not_mutate_options(monkeypatch, name):
    from jax_finufft.options import Opts

    monkeypatch.setattr(nufft_api, "CRYOJAX_FINUFFT_BACKEND", "jax-finufft")
    opts = Opts(modeord=True)
    options = {"opts": opts}
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((6, 4), complex)
    getattr(im, name)(jnp.zeros((3, 2)), second, (6, 6), options=options)
    assert options == {"opts": opts}
