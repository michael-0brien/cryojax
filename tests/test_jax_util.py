import cryojax.jax_util as jxu
import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.io import read_array_from_mrc
from jaxtyping import Array


try:
    import lineax as lx
except ModuleNotFoundError:
    lx = None

_skip_without_lineax = pytest.mark.skipif(lx is None, reason="lineax not installed")


#
# Test PyTree transforms
#
class Exp(eqx.Module):
    a: Array = eqx.field(converter=jnp.asarray)

    def __call__(self, x):
        return jnp.exp(-self.a * x)


def test_resolve_transform():
    pytree = Exp(a=1.0)
    pytree_with_transform = eqx.tree_at(
        lambda fn: fn.a,
        pytree,
        replace_fn=lambda a: jxu.CustomTransform(jnp.exp, jnp.log(a)),
    )
    assert eqx.tree_equal(pytree, jxu.resolve_transforms(pytree_with_transform))


def test_nested_resolve_transform():
    pytree = Exp(a=1.0)
    pytree_with_transform = eqx.tree_at(
        lambda fn: fn.a,
        pytree,
        replace_fn=lambda a: jxu.CustomTransform(lambda b: 2 * b, a / 2),
    )
    pytree_with_nested_transform = eqx.tree_at(
        lambda fn: fn.a.args[0],
        pytree_with_transform,
        replace_fn=lambda a_scaled: jxu.CustomTransform(jnp.exp, jnp.log(a_scaled)),
    )
    assert eqx.tree_equal(
        pytree,
        jxu.resolve_transforms(pytree_with_transform),
        jxu.resolve_transforms(pytree_with_nested_transform),
    )


def test_stop_gradient():
    @jax.value_and_grad
    def objective_fn(pytree):
        exp, x = jxu.resolve_transforms(pytree)
        return exp(x)

    x = jnp.asarray(np.random.random())
    exp = Exp(a=1.0)
    exp_with_stop_gradient = eqx.tree_at(
        lambda fn: fn.a, exp, replace_fn=jxu.StopGradientTransform
    )
    _, grads = objective_fn((exp_with_stop_gradient, x))
    grads = jxu.resolve_transforms(grads)
    assert grads[0].a == 0.0


#
# Test `filter_bscan` / `filter_bmap`
#
@pytest.mark.parametrize(
    "batch_size,dim",
    [
        (1, 200),
        (10, 200),
        (33, 200),
        (200, 200),
    ],
)
def test_bscan_remainder(batch_size, dim):
    @jax.jit
    @jax.vmap
    def f(x):
        return x + 1

    xs = jnp.zeros(dim)
    np.testing.assert_allclose(jxu.filter_bmap(f, xs, batch_size=batch_size), f(xs))


#
# Linear operators
#
@pytest.fixture
def voxel_info(sample_mrc_path):
    return read_array_from_mrc(sample_mrc_path, loads_grid_spacing=True)


@pytest.fixture
def voxel_volume(voxel_info):
    return cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_info[0], pad_scale=1.3)


@pytest.fixture
def voxel_size(voxel_info):
    return voxel_info[1]


@pytest.fixture
def image_config(voxel_volume, voxel_size):
    shape = voxel_volume.shape[0:2]
    return cxs.BasicImageConfig(
        shape=(int(0.9 * shape[0]), int(0.9 * shape[1])),
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
        padded_shape=shape,
    )


@pytest.fixture
def image_model(voxel_volume, image_config):
    pose = cxs.EulerAnglePose()
    return cxs.make_image_model(voxel_volume, image_config, pose)


@_skip_without_lineax
def test_simulate_equality(image_model):
    linear_operator, vector = jxu.make_linear_operator(
        fn=lambda x: x.simulate(),
        args=image_model,
        where_vector=lambda x: x.volume.values,
    )
    image_cxs = image_model.simulate()
    image_lx = linear_operator.mv(vector)
    np.testing.assert_allclose(image_cxs, image_lx)


@_skip_without_lineax
def test_linear_transpose(image_model):
    where_vector = lambda x: x.volume.values
    linear_operator, _ = jxu.make_linear_operator(
        fn=lambda x: x.simulate(),
        args=image_model,
        where_vector=where_vector,
    )
    voxel_grid = where_vector(image_model)
    backprojection = where_vector(
        linear_operator.T.mv(jnp.zeros(image_model.image_config.shape))
    )
    assert voxel_grid.shape == backprojection.shape


@_skip_without_lineax
def test_bad_linear_transpose(sample_pdb_path, image_config):
    image_model = cxs.make_image_model(
        cxs.load_tabulated_volume(sample_pdb_path, output_type=cxs.GaussianMixtureVolume),
        image_config,
        pose=cxs.EulerAnglePose(),
    )
    where_vector = lambda x: x.volume.positions
    linear_operator, _ = jxu.make_linear_operator(
        fn=lambda x: x.simulate(),
        args=image_model,
        where_vector=where_vector,
    )
    with pytest.raises(Exception):
        linear_operator.T
