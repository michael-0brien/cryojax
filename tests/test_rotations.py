"""Tests for `cryojax.rotations.SO3` and `cryojax.rotations.SO2`.

The batched compute logic is validated against the ground truth of mapping the
unbatched implementation with `equinox.filter_vmap`, and standard group axioms
(identity, inverse, matrix round-trips) are checked as well.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.rotations import SO2, SO3


BATCH = 7


def _random_quaternions(key, shape=(BATCH,)):
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def _random_unit_complex(key, shape=(BATCH,)):
    theta = jax.random.uniform(key, shape, minval=0.0, maxval=2 * jnp.pi)
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)


# ----------------------------------------------------------------------------
# Backend preservation (the `leaf_asarray` convention)
# ----------------------------------------------------------------------------


def test_so3_preserves_numpy_backend():
    R = SO3(np.array([1.0, 0.0, 0.0, 0.0]))
    assert isinstance(R.wxyz, np.ndarray)  # host-resident, not moved to device
    # ... but compute methods still return JAX arrays
    assert isinstance(R.as_matrix(), jax.Array)


def test_so2_preserves_numpy_backend():
    S = SO2(np.array([1.0, 0.0]))
    assert isinstance(S.unit_complex, np.ndarray)
    assert isinstance(S.as_matrix(), jax.Array)


def test_so3_python_scalar_sequence_is_host():
    R = SO3([1.0, 0.0, 0.0, 0.0])
    assert isinstance(R.wxyz, np.ndarray)


# ----------------------------------------------------------------------------
# Unbatched sanity checks
# ----------------------------------------------------------------------------


def test_so3_identity_and_roundtrips():
    R = SO3.identity()
    np.testing.assert_allclose(R.as_matrix(), np.eye(3), atol=1e-6)
    # matrix round-trip
    q = _random_quaternions(jax.random.PRNGKey(0), shape=())
    R = SO3(q)
    R2 = SO3.from_matrix(R.as_matrix())
    np.testing.assert_allclose(R.as_matrix(), R2.as_matrix(), atol=1e-5)
    # exp/log round-trip
    tangent = jax.random.normal(jax.random.PRNGKey(1), (3,)) * 0.5
    np.testing.assert_allclose(SO3.exp(tangent).log(), tangent, atol=1e-5)


def test_so3_inverse_is_group_inverse():
    q = _random_quaternions(jax.random.PRNGKey(2), shape=())
    R = SO3(q)
    np.testing.assert_allclose((R @ R.inverse()).as_matrix(), np.eye(3), atol=1e-5)


def test_so2_identity_and_roundtrips():
    S = SO2.identity()
    np.testing.assert_allclose(S.as_matrix(), np.eye(2), atol=1e-6)
    theta = 0.7
    S = SO2.from_radians(theta)
    np.testing.assert_allclose(S.as_radians(), theta, atol=1e-6)
    # `inverse` undoes `apply`
    v = jnp.asarray([1.0, 2.0])
    np.testing.assert_allclose(S.inverse().apply(S.apply(v)), v, atol=1e-6)
    # group inverse: `S @ S.inverse()` is the identity
    np.testing.assert_allclose((S @ S.inverse()).as_matrix(), np.eye(2), atol=1e-6)
    # `from_matrix` inverts `as_matrix`
    np.testing.assert_allclose(
        SO2.from_matrix(S.as_matrix()).as_radians(), theta, atol=1e-6
    )


def test_so2_compose_adds_angles():
    a, b = 0.3, 1.1
    composed = SO2.from_radians(a) @ SO2.from_radians(b)
    np.testing.assert_allclose(composed.as_radians(), a + b, atol=1e-6)


# ----------------------------------------------------------------------------
# Batched compute logic == vmap over the unbatched implementation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["as_matrix", "log", "normalize_wxyz", "inverse_wxyz"],
)
def test_so3_unary_batched_matches_vmap(method):
    wxyz = _random_quaternions(jax.random.PRNGKey(3))
    R = SO3(wxyz)

    fns = {
        "as_matrix": lambda r: r.as_matrix(),
        "log": lambda r: r.log(),
        "normalize_wxyz": lambda r: r.normalize().wxyz,
        "inverse_wxyz": lambda r: r.inverse().wxyz,
    }
    fn = fns[method]
    batched = fn(R)
    vmapped = eqx.filter_vmap(lambda q: fn(SO3(q)))(wxyz)
    np.testing.assert_allclose(batched, vmapped, atol=1e-5)


def test_so3_compose_batched_matches_vmap():
    wxyz0 = _random_quaternions(jax.random.PRNGKey(4))
    wxyz1 = _random_quaternions(jax.random.PRNGKey(5))
    batched = SO3(wxyz0).compose(SO3(wxyz1)).wxyz
    vmapped = eqx.filter_vmap(lambda a, b: SO3(a).compose(SO3(b)).wxyz)(wxyz0, wxyz1)
    np.testing.assert_allclose(batched, vmapped, atol=1e-5)


def test_so3_apply_batched_matches_vmap():
    wxyz = _random_quaternions(jax.random.PRNGKey(6))
    targets = jax.random.normal(jax.random.PRNGKey(7), (BATCH, 3))
    batched = SO3(wxyz).apply(targets)
    vmapped = eqx.filter_vmap(lambda q, t: SO3(q).apply(t))(wxyz, targets)
    np.testing.assert_allclose(batched, vmapped, atol=1e-5)


def test_so3_exp_and_from_matrix_batched_matches_vmap():
    tangent = jax.random.normal(jax.random.PRNGKey(8), (BATCH, 3)) * 0.4
    np.testing.assert_allclose(
        SO3.exp(tangent).wxyz,
        eqx.filter_vmap(lambda t: SO3.exp(t).wxyz)(tangent),
        atol=1e-5,
    )
    wxyz = _random_quaternions(jax.random.PRNGKey(9))
    matrices = SO3(wxyz).as_matrix()
    np.testing.assert_allclose(
        SO3.from_matrix(matrices).wxyz,
        eqx.filter_vmap(lambda m: SO3.from_matrix(m).wxyz)(matrices),
        atol=1e-5,
    )


def test_so3_from_axis_radians_batched_matches_vmap():
    angles = jnp.linspace(-2.0, 2.0, BATCH)
    for ctor in (SO3.from_x_radians, SO3.from_y_radians, SO3.from_z_radians):
        np.testing.assert_allclose(
            ctor(angles).wxyz,
            eqx.filter_vmap(lambda a: ctor(a).wxyz)(angles),
            atol=1e-5,
        )


def test_so3_single_rotation_broadcasts_over_grid():
    # A single (unbatched) rotation applied to a coordinate grid must equal
    # nested vmaps over the grid axes. This is the `AbstractPose.rotate_coordinates`
    # usage pattern.
    R = SO3(_random_quaternions(jax.random.PRNGKey(10), shape=()))
    grid = jax.random.normal(jax.random.PRNGKey(11), (4, 5, 6, 3))
    batched = R.apply(grid)
    vmapped = jax.vmap(jax.vmap(jax.vmap(R.apply)))(grid)
    np.testing.assert_allclose(batched, vmapped, atol=1e-5)


@pytest.mark.parametrize(
    "method",
    ["as_matrix", "log", "normalize_uc", "inverse_uc"],
)
def test_so2_unary_batched_matches_vmap(method):
    uc = _random_unit_complex(jax.random.PRNGKey(12))
    S = SO2(uc)
    fns = {
        "as_matrix": lambda s: s.as_matrix(),
        "log": lambda s: s.log(),
        "normalize_uc": lambda s: s.normalize().unit_complex,
        "inverse_uc": lambda s: s.inverse().unit_complex,
    }
    fn = fns[method]
    np.testing.assert_allclose(
        fn(S), eqx.filter_vmap(lambda u: fn(SO2(u)))(uc), atol=1e-5
    )


def test_so2_compose_and_apply_batched_matches_vmap():
    uc0 = _random_unit_complex(jax.random.PRNGKey(13))
    uc1 = _random_unit_complex(jax.random.PRNGKey(14))
    np.testing.assert_allclose(
        SO2(uc0).compose(SO2(uc1)).unit_complex,
        eqx.filter_vmap(lambda a, b: SO2(a).compose(SO2(b)).unit_complex)(uc0, uc1),
        atol=1e-5,
    )
    targets = jax.random.normal(jax.random.PRNGKey(15), (BATCH, 2))
    np.testing.assert_allclose(
        SO2(uc0).apply(targets),
        eqx.filter_vmap(lambda u, t: SO2(u).apply(t))(uc0, targets),
        atol=1e-5,
    )


def test_so2_from_radians_and_from_matrix_batched_matches_vmap():
    angles = jnp.linspace(0.0, 2 * jnp.pi, BATCH)
    np.testing.assert_allclose(
        SO2.from_radians(angles).unit_complex,
        eqx.filter_vmap(lambda a: SO2.from_radians(a).unit_complex)(angles),
        atol=1e-5,
    )
    uc = _random_unit_complex(jax.random.PRNGKey(16))
    matrices = SO2(uc).as_matrix()
    np.testing.assert_allclose(
        SO2.from_matrix(matrices).unit_complex,
        eqx.filter_vmap(lambda m: SO2.from_matrix(m).unit_complex)(matrices),
        atol=1e-5,
    )


# ----------------------------------------------------------------------------
# Batched objects compose under vmap in a full pose usage
# ----------------------------------------------------------------------------


def test_batched_so3_apply_under_vmap_matches_loop():
    wxyz = _random_quaternions(jax.random.PRNGKey(17))
    grid = jax.random.normal(jax.random.PRNGKey(18), (3, 3, 3, 3))

    # Map a batch of rotations over a shared grid
    batched = eqx.filter_vmap(lambda q: SO3(q).apply(grid))(wxyz)
    expected = jnp.stack([SO3(wxyz[i]).apply(grid) for i in range(BATCH)], axis=0)
    np.testing.assert_allclose(batched, expected, atol=1e-5)
