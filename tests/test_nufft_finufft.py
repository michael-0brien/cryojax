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
DTYPES_AND_EPS = [
    (np.float64, 1e-3),
    (np.float64, 1e-6),
    (np.float64, 1e-9),
    (np.float32, 1e-3),
    (np.float32, 1e-4),
]
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


def _finufft_type1(positions, strengths, shape, modes, eps=1e-14, upsampfac=None):
    transform = finufft.nufft2d1 if len(shape) == 2 else finufft.nufft3d1
    kwargs = {} if upsampfac is None else {"upsampfac": upsampfac}
    return transform(
        *_angles(positions, shape),
        np.asarray(strengths, dtype=complex),
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


def _hermitian_grid(spectrum, shape):
    """`G[k] = w_k F[k] / 2`, `G[-k] = conj(G[k])`, padded to `n + 1` on even axes."""
    padded = tuple(n + 1 - n % 2 for n in shape)
    half = spectrum * _hermitian_weights(shape) / 2
    k = [np.broadcast_to(f, half.shape) for f in _half_grid_frequencies(shape)]
    grid = np.zeros(padded, dtype=complex)
    np.add.at(grid, tuple(f % p for f, p in zip(k, padded)), half)
    np.add.at(grid, tuple(-f % p for f, p in zip(k, padded)), np.conj(half))
    return grid


def _finufft_type2(positions, grid, shape, eps=1e-14, upsampfac=None):
    transform = finufft.nufft2d2 if len(shape) == 2 else finufft.nufft3d2
    kwargs = {} if upsampfac is None else {"upsampfac": upsampfac}
    values = transform(
        *_angles(positions, shape), grid, eps=eps, isign=1, modeord=1, **kwargs
    )
    return values.real


def _type2_reference(positions, spectrum, shape):
    grid = _hermitian_grid(spectrum, shape)
    return _finufft_type2(positions, grid, shape) / np.prod(shape)


def _finufft_error(positions, shape, eps, upsampfac, strengths=None, spectrum=None):
    """finufft's own type-1 or type-2 error at `eps` and `upsampfac`, on `shape` itself:
    padding an axis for the references would enlarge its fine grid."""
    if spectrum is None:
        exact, approx = (
            _finufft_type1(positions, strengths, shape, shape, eps=e, upsampfac=u)
            for e, u in ((1e-14, None), (eps, upsampfac))
        )
    else:
        grid = np.asarray(im.convert_rfftn_to_fftn(jnp.asarray(spectrum), shape), complex)
        exact, approx = (
            _finufft_type2(positions, grid, shape, eps=e, upsampfac=u)
            for e, u in ((1e-14, None), (eps, upsampfac))
        )
    return _relative_error(approx, exact)


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


def _bound(eps, finufft_error, dtype):
    bound = max(3 * eps, 2 * finufft_error)
    return max(bound, 5e-4) if dtype == np.float32 else bound


#
# Agreement with finufft
#
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("dtype, eps", DTYPES_AND_EPS)
@pytest.mark.parametrize("shape", SHAPES)
def test_type1_matches_finufft(shape, dtype, eps, sigma):
    positions = _points(shape, NUM_POINTS, seed=0).astype(dtype)
    strengths = np.random.default_rng(1).normal(size=NUM_POINTS).astype(dtype)
    reference = _type1_reference(positions, strengths, shape)
    finufft_error = _finufft_error(positions, shape, eps, sigma, strengths=strengths)
    native = im.dispatch_nufft1(
        jnp.asarray(positions),
        jnp.asarray(strengths),
        shape,
        pixel_size=PIXEL_SIZE,
        eps=eps,
        upsample_factor=sigma,
    )
    assert native.dtype == (jnp.complex64 if dtype == np.float32 else jnp.complex128)
    assert native.shape == reference.shape
    assert _relative_error(native, reference) <= _bound(eps, finufft_error, dtype)


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("dtype, eps", DTYPES_AND_EPS)
@pytest.mark.parametrize("shape", SHAPES)
def test_type2_matches_finufft(shape, dtype, eps, sigma):
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    positions = _points(shape, NUM_POINTS, seed=0).astype(dtype)
    spectrum = _random_spectrum(shape, seed=1).astype(complex_dtype)
    reference = _type2_reference(positions, spectrum, shape)
    finufft_error = _finufft_error(positions, shape, eps, sigma, spectrum=spectrum)
    native = im.dispatch_nufft2(
        jnp.asarray(positions),
        jnp.asarray(spectrum),
        shape,
        pixel_size=PIXEL_SIZE,
        eps=eps,
        upsample_factor=sigma,
    )
    assert native.dtype == dtype
    assert native.shape == (NUM_POINTS,)
    assert _relative_error(native, reference) <= _bound(eps, finufft_error, dtype)


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
@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_auto_is_static(monkeypatch, name):
    """`"auto"` is the default and resolves from static shapes, `eps` and dtype: one
    trace for new values, no leaked tracers, the kernel of the resolved factor. In
    float32 at eps 1e-6 this 3D case resolves away from 1.25."""
    shape = (32, 30, 28)
    positions = jnp.asarray(_points(shape, NUM_POINTS, seed=0), jnp.float32)
    if name == "dispatch_nufft1":
        second = jnp.asarray(
            np.random.default_rng(1).normal(size=NUM_POINTS), jnp.float32
        )
    else:
        second = jnp.asarray(_random_spectrum(shape, seed=1), jnp.complex64)
    sigma = resolve_upsample_factor(
        "auto", n_points=NUM_POINTS, shape=shape, eps=1e-6, dtype=np.float32, pallas=False
    )
    assert sigma != 1.25
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
            positions, second, shape, pixel_size=PIXEL_SIZE, eps=1e-6, enable_pallas=False
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
# Errors
#
@pytest.mark.xfail(reason="Task 2")
@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_eps_too_small_raises(name):
    shape = (8, 6)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    with pytest.raises(ValueError, match=rf"{name}[\s\S]*eps[\s\S]*1e-12"):
        getattr(im, name)(
            jnp.zeros((3, 2)), second, shape, eps=1e-12, upsample_factor=1.25
        )


@pytest.mark.xfail(reason="Task 2")
@pytest.mark.parametrize("upsample_factor", ["auto", 2.0])
@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_eps_below_fit_precision_raises(name, upsample_factor):
    """Legal widths at sigma 2, but an ill-conditioned or failing tap fit."""
    shape = (8, 6)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=rf"{name}[\s\S]*eps[\s\S]*1e-13"):
            getattr(im, name)(
                jnp.zeros((3, 2)),
                second,
                shape,
                eps=1e-13,
                upsample_factor=upsample_factor,
            )


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_kernel_error_names_dispatch_and_resolved_factor(monkeypatch, name):
    def make_pswf_kernel(eps, upsample_factor, ndim, dtype):
        raise ValueError("Found invalid value for `make_pswf_kernel(...)`.")

    monkeypatch.setattr(nufft_api, "make_pswf_kernel", make_pswf_kernel)
    shape = (8, 6)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    sigma = resolve_upsample_factor(
        "auto", n_points=3, shape=shape, eps=1e-6, dtype=np.float64, pallas=False
    )
    with pytest.raises(
        ValueError, match=rf"{name}[\s\S]*eps = 1e-06[\s\S]*upsample_factor = {sigma}`"
    ):
        getattr(im, name)(jnp.zeros((3, 2)), second, shape, enable_pallas=False)


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
def test_auto_model_errors_pass_through(monkeypatch, name):
    def resolve_upsample_factor(upsample_factor, **kwargs):
        raise np.linalg.LinAlgError("cost model")

    monkeypatch.setattr(nufft_api, "resolve_upsample_factor", resolve_upsample_factor)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    with pytest.raises(np.linalg.LinAlgError, match="cost model"):
        getattr(im, name)(jnp.zeros((3, 2)), second, (8, 6))


@pytest.mark.skipif(jax_finufft is None, reason="jax-finufft not installed")
@pytest.mark.parametrize("eps, upsample_factor", [(1e-10, 1.25), (1e-13, "auto")])
def test_jax_finufft_skips_native_kernel_limits(monkeypatch, eps, upsample_factor):
    """finufft takes widths past the native cap and `eps` below the native fit's."""
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


@pytest.mark.parametrize("name", ["dispatch_nufft1", "dispatch_nufft2"])
@pytest.mark.parametrize("upsample_factor", [1.0, 0.5])
def test_upsample_factor_at_most_one_raises(name, upsample_factor):
    shape = (8, 6)
    second = jnp.ones(3) if name == "dispatch_nufft1" else jnp.ones((8, 4), complex)
    with pytest.raises(
        ValueError, match=rf"{name}[\s\S]*upsample_factor[\s\S]*{upsample_factor}"
    ):
        getattr(im, name)(
            jnp.zeros((3, 2)), second, shape, upsample_factor=upsample_factor
        )


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
