"""`cryojax.ndimage.compute_spread_order`: a bin-sorting permutation that leaves every
spread and type-1 NUFFT unchanged."""

import cryojax.ndimage as im
import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


def _points(m, shape, seed=0):
    """`(M, d)` positions as `(x, y[, z])`, in pixels from grid index `n // 2`."""
    rng = np.random.default_rng(seed)
    extent = np.array(shape[::-1])
    return jnp.asarray(rng.uniform(-0.45, 0.45, (m, len(shape))) * extent)


@pytest.mark.parametrize("shape", [(32, 48), (16, 24, 32)])
def test_order_is_a_permutation(shape):
    order = im.compute_spread_order(_points(500, shape), shape)
    assert order.shape == (500,)
    np.testing.assert_array_equal(np.sort(np.asarray(order)), np.arange(500))


@pytest.mark.parametrize(
    "shape, bin_shape", [((32, 48), (4, 16)), ((16, 24, 32), (4, 4, 16))]
)
def test_sorted_points_visit_bins_x_fastest(shape, bin_shape):
    positions = _points(2000, shape)
    order = im.compute_spread_order(positions, shape)
    grid = np.asarray(positions)[np.asarray(order)] + np.array(shape[::-1]) // 2
    bins = np.floor(grid[:, ::-1] / np.array(bin_shape)).astype(int)  # (z,) y, x
    n_bins = -(-np.array(shape) // np.array(bin_shape))
    key = np.ravel_multi_index(tuple(bins.T), tuple(n_bins))
    assert np.all(np.diff(key) >= 0)


def test_upsample_factor_bins_the_fine_grid():
    # 5 pixels apart in x: one 16-cell bin on the grid, two on a 4x fine grid
    positions = jnp.array([[5.0, 0.0], [0.0, 0.0]])
    order = im.compute_spread_order(positions, (64, 64))
    np.testing.assert_array_equal(order, [0, 1])
    order = im.compute_spread_order(positions, (64, 64), upsample_factor=4.0)
    np.testing.assert_array_equal(order, [1, 0])


def test_pixel_size_converts_positions():
    positions = jnp.array([[10.0, 0.0], [0.0, 0.0]])  # 5 and 0 pixels at pixel size 2
    a = im.compute_spread_order(positions, (64, 64), pixel_size=2.0, upsample_factor=4.0)
    b = im.compute_spread_order(positions / 2, (64, 64), upsample_factor=4.0)
    np.testing.assert_array_equal(a, b)


def test_spreading_is_unchanged_by_the_order():
    shape, pixel_size = (32, 48), 1.5
    positions = _points(400, shape) * pixel_size
    amplitude = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.0, 400))
    order = im.compute_spread_order(positions, shape, pixel_size=pixel_size)

    def spread(p, a):
        return im.spread_gaussians_2d(
            p[:, 0], p[:, 1], a, 2.0, shape, pixel_size=pixel_size, n_spread=7
        )

    np.testing.assert_allclose(
        spread(positions[order], amplitude[order]),
        spread(positions, amplitude),
        atol=1e-12,
    )


@pytest.mark.parametrize("shape", [(32, 48), (16, 24, 32)])
def test_type1_nufft_is_unchanged_by_the_order(shape):
    positions = _points(300, shape)
    strengths = jnp.asarray(np.random.default_rng(2).standard_normal((2, 300)))
    order = im.compute_spread_order(positions, shape, upsample_factor=2.0)
    kwargs = dict(eps=1e-8, upsample_factor=2.0)
    np.testing.assert_allclose(
        im.dispatch_nufft1(positions[order], strengths[:, order], shape, **kwargs),
        im.dispatch_nufft1(positions, strengths, shape, **kwargs),
        atol=1e-10,
    )


def test_inverse_restores_type2_values():
    shape = (32, 48)
    positions = _points(300, shape)
    rng = np.random.default_rng(3)
    spectrum = jnp.asarray(
        rng.standard_normal((32, 25)) + 1j * rng.standard_normal((32, 25))
    )
    order = im.compute_spread_order(positions, shape)
    inverse = jnp.argsort(order)
    kwargs = dict(eps=1e-8, upsample_factor=2.0)
    values = im.dispatch_nufft2(positions[order], spectrum, shape, **kwargs)[inverse]
    np.testing.assert_allclose(
        values, im.dispatch_nufft2(positions, spectrum, shape, **kwargs), atol=1e-10
    )


def test_wrapped_positions_sort_with_their_periodic_image():
    shape = (16, 64)
    positions = jnp.array([[-40.0, 0.0], [24.0, 0.0], [0.0, 0.0]])  # -40 wraps to 24
    order = np.asarray(im.compute_spread_order(positions, shape))
    assert (
        abs(int(np.flatnonzero(order == 0)[0]) - int(np.flatnonzero(order == 1)[0])) == 1
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(positions=jnp.zeros((3, 3)), shape=(8, 8)), "positions"),
        (dict(positions=jnp.zeros((3, 2)), shape=(8,)), "shape"),
        (dict(positions=jnp.zeros((3, 2)), shape=(8, 8), bin_shape=(4,)), "bin_shape"),
        (dict(positions=jnp.zeros((3, 2)), shape=(8, 8), bin_shape=(0, 4)), "bin_shape"),
        (
            dict(positions=jnp.zeros((3, 2)), shape=(8, 8), upsample_factor=0.5),
            "upsample",
        ),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        im.compute_spread_order(**kwargs)


def test_jit_compatible():
    shape = (32, 48)
    positions = _points(100, shape)
    np.testing.assert_array_equal(
        jax.jit(im.compute_spread_order, static_argnums=1)(positions, shape),
        im.compute_spread_order(positions, shape),
    )
