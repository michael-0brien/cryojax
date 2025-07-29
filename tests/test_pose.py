import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import cryojax.simulator as cs
from cryojax.rotations import SO3


config.update("jax_enable_x64", True)


def test_default_pose_arguments():
    euler = cs.EulerAnglePose()
    quat = cs.QuaternionPose()
    axis_angle = cs.AxisAnglePose()
    np.testing.assert_allclose(euler.rotation.as_matrix(), quat.rotation.as_matrix())
    np.testing.assert_allclose(
        euler.rotation.as_matrix(), axis_angle.rotation.as_matrix()
    )


def test_translation_agreement():
    rotation = SO3(jnp.asarray((1.0, 0.0, 0.0, 0.0)))
    offset = jnp.asarray((0.0, -1.4))
    quat = cs.QuaternionPose.from_rotation_and_translation(rotation, offset)
    axis_angle = cs.AxisAnglePose.from_rotation_and_translation(rotation, offset)
    np.testing.assert_allclose(quat.rotation.as_matrix(), axis_angle.rotation.as_matrix())
    np.testing.assert_allclose(quat.offset_in_angstroms, axis_angle.offset_in_angstroms)


def test_pose_conversion():
    wxyz = jnp.asarray((1.0, 2.0, 3.0, 0.5))
    rotation = SO3(wxyz).normalize()
    quat = cs.QuaternionPose.from_rotation(rotation)
    euler = cs.EulerAnglePose.from_rotation(rotation)
    axis_angle = cs.AxisAnglePose.from_rotation(rotation)
    np.testing.assert_allclose(quat.rotation.as_matrix(), euler.rotation.as_matrix())
    np.testing.assert_allclose(quat.rotation.as_matrix(), axis_angle.rotation.as_matrix())


def test_invert_pose():
    wxyz = jnp.asarray((1.0, 2.0, 3.0, 0.5))
    offset = jnp.asarray((-1.0, 1.0))
    rotation = SO3(wxyz).normalize()
    quat = cs.QuaternionPose.from_rotation_and_translation(rotation, offset)
    quat_inverse = quat.to_inverse_rotation()
    np.testing.assert_allclose(quat.wxyz, quat_inverse.wxyz.at[1:].mul(-1))
    np.testing.assert_allclose(quat.offset_in_angstroms, quat_inverse.offset_in_angstroms)


def test_axis_angle_euler_agreement():
    angle = 2.0
    angle_in_radians = jnp.deg2rad(angle)
    rotation_x = SO3.from_x_radians(angle_in_radians)
    rotation_y = SO3.from_y_radians(angle_in_radians)
    rotation_z = SO3.from_z_radians(angle_in_radians)
    aa_x = cs.AxisAnglePose(euler_vector=(angle, 0.0, 0.0))
    aa_y = cs.AxisAnglePose(euler_vector=(0.0, angle, 0.0))
    aa_z = cs.AxisAnglePose(euler_vector=(0.0, 0.0, angle))
    np.testing.assert_allclose(rotation_x.as_matrix(), aa_x.rotation.as_matrix())
    np.testing.assert_allclose(rotation_y.as_matrix(), aa_y.rotation.as_matrix())
    np.testing.assert_allclose(rotation_z.as_matrix(), aa_z.rotation.as_matrix())


@pytest.mark.parametrize(
    "phi, theta, psi",
    [
        (2.0, 15.0, -40.0),
        (10.0, 90.0, 170.0),
        (-10.0, 120.0, 140.0),
        (-120.0, 40.0, -80.0),
    ],
)
def test_euler_angle_conversion(phi, theta, psi):
    pose = cs.EulerAnglePose(phi_angle=phi, theta_angle=theta, psi_angle=psi)
    converted_pose = cs.EulerAnglePose.from_rotation(pose.rotation)
    np.testing.assert_allclose(
        np.asarray((phi, theta, psi)),
        np.asarray(
            (
                converted_pose.phi_angle,
                converted_pose.theta_angle,
                converted_pose.psi_angle,
            )
        ),
    )
