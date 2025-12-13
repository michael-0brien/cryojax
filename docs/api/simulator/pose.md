# Representing poses

`cryojax` provides different parameterizations for the pose of a structure. These are captured through the abstract base class called `AbstractPose`.

!!! info "Rotation and translation conventions"
    Pose transformations in `cryojax` can be captured by the equation

    $$\vec{x}' = R \vec{x} + \vec{t},$$

    where $\vec{x}'$ and $\vec{x}$ are the 3D coordinate vectors in the rotated and unrotated frames, and $R$ and $\vec{t} = (t_x, t_y, 0)$ are the rotation and in-plane translation vector.

    Standard softwares such as RELION and cisTEM define the rotation and translation to
    "undo" the observed pose. This can be captured by a translation to the center, followed by a rotation:

    $$\vec{x} = (\vec{x}' + \vec{t}^*) R^*.$$

    When $R^* = R^T$ and $\vec{t}^* = -\vec{t}$, this equation can be inverted for $\vec{x}'$ to recover the cryoJAX convention.

    Additional considerations are required to convert between RELION and cisTEM euler angles since this is a composition
    of three rotations. See the `EulerAnglePose` documentation for more information.

!!! info "Degrees vs radians conventions"
    Angular quantities in `cryojax` are always in *degrees*.
    Therefore concrete classes of the `AbstractPose` have
    angles in degrees, e.g.

    ```python
    import cryojax.simulator as cxs

    phi_in_degrees, theta_in_degrees, psi_in_degrees = 10.0, 30.0, 40.0
    pose = cxs.EulerAnglePose(
        phi_angle=phi_in_degrees,
        theta_angle=theta_in_degrees,
        psi_angle=psi_in_degrees,
    )
    ```

???+ abstract "`cryojax.simulator.AbstractPose`"
    ::: cryojax.simulator.AbstractPose
        options:
            members:
                - rotation
                - from_rotation
                - to_inverse_rotation

::: cryojax.simulator.EulerAnglePose
        options:
            members:
                - __init__
                - compute_translation_operator
                - translate_image
                - rotate_coordinates
                - offset_in_angstroms
                - rotation
                - from_rotation
                - from_rotation_and_translation
                - to_inverse_rotation

---

::: cryojax.simulator.QuaternionPose
        options:
            members:
                - __init__
                - compute_translation_operator
                - translate_image
                - rotate_coordinates
                - offset_in_angstroms
                - rotation
                - from_rotation
                - from_rotation_and_translation
                - to_inverse_rotation

---

::: cryojax.simulator.AxisAnglePose
        options:
            members:
                - __init__
                - compute_translation_operator
                - translate_image
                - rotate_coordinates
                - offset_in_angstroms
                - rotation
                - from_rotation
                - from_rotation_and_translation
                - to_inverse_rotation
