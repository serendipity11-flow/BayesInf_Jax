"""Differentiable, inference-time geometry features for the planar pusher."""

import jax
import jax.numpy as jnp


ARM_BASE = jnp.asarray((-0.72, 0.0))
LINK_LENGTHS = jnp.asarray((0.34, 0.30, 0.22))
PUSHER_RADIUS = 0.055
OBJECT_HALF_EXTENTS = jnp.asarray((0.09, 0.055))


def pusher_contact_features(state, *, temperature=0.02):
    """Return distance, normal relative speed, proximity, and face normal.

    ``state`` is the physical 12-D MuJoCo state
    ``[q_arm(3), q_object(3), v_arm(3), v_object(3)]``.  Distance is from the
    spherical fingertip surface to the nearest point on the oriented cuboid;
    it is negative for overlap.  The final two values are the selected face
    normal in the cuboid frame (a continuous alternative to a face one-hot).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q, qd = state[:6], state[6:]
    angles = jnp.cumsum(q[:3])
    directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
    tip = ARM_BASE + jnp.sum(LINK_LENGTHS[:, None] * directions, axis=0)
    # Analytic planar manipulator Jacobian.
    derivatives = LINK_LENGTHS[:, None] * jnp.stack(
        (-jnp.sin(angles), jnp.cos(angles)), axis=-1
    )
    jacobian = jnp.stack(
        (jnp.sum(derivatives[0:], axis=0),
         jnp.sum(derivatives[1:], axis=0), derivatives[2]), axis=1
    )
    tip_velocity = jacobian @ qd[:3]

    yaw = q[5]
    rotation = jnp.asarray(((jnp.cos(yaw), -jnp.sin(yaw)),
                            (jnp.sin(yaw), jnp.cos(yaw))))
    relative_local = rotation.T @ (tip - q[3:5])
    closest_local = jnp.clip(relative_local, -OBJECT_HALF_EXTENTS,
                             OBJECT_HALF_EXTENTS)
    outside = relative_local - closest_local
    outside_sq = jnp.sum(outside ** 2)
    # A raw norm has an undefined derivative at exactly zero, which occurs
    # whenever the tip centre projects inside/on the box.  Smooth only the
    # denominator; retain the exact branch predicate and signed distance.
    safe_outside_norm = jnp.sqrt(outside_sq + 1e-12)
    is_outside = outside_sq > 1e-12
    face_clearance = OBJECT_HALF_EXTENTS - jnp.abs(relative_local)
    face_axis = jnp.argmin(face_clearance)
    face_sign = jnp.where(relative_local[face_axis] >= 0, 1.0, -1.0)
    inside_normal = jnp.zeros(2).at[face_axis].set(face_sign)
    outside_normal = outside / safe_outside_norm
    normal_local = jnp.where(is_outside, outside_normal, inside_normal)
    point_distance = jnp.where(
        is_outside, safe_outside_norm, -face_clearance[face_axis]
    )
    signed_distance = point_distance - PUSHER_RADIUS
    normal_world = rotation @ normal_local
    contact_arm = closest_local
    rotational_velocity = qd[5] * jnp.asarray((-contact_arm[1], contact_arm[0]))
    object_point_velocity = qd[3:5] + rotation @ rotational_velocity
    relative_normal_velocity = jnp.dot(
        tip_velocity - object_point_velocity, normal_world
    )
    proximity = jax.nn.sigmoid(-signed_distance / temperature)
    return jnp.concatenate((
        jnp.asarray((signed_distance, relative_normal_velocity, proximity)),
        normal_local,
    ))
