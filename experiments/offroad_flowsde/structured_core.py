"""Kinematic reconstruction utilities for the learned dynamics core."""

import jax.numpy as jnp

CONTEXT_DIM, CORE_DIM, ACTION_DIM = 16, 14, 2


def world_velocity(core):
    psi, longitudinal, lateral = core[..., 0], core[..., 1], core[..., 2]
    return jnp.stack((longitudinal*jnp.cos(psi)-lateral*jnp.sin(psi),
                      longitudinal*jnp.sin(psi)+lateral*jnp.cos(psi)), axis=-1)


def reconstruct_next(current, next_core, dt):
    """Advance XY by trapezoidal integration; append the predicted core."""
    velocity = .5*(world_velocity(current[..., 2:]) + world_velocity(next_core))
    return jnp.concatenate((current[..., :2] + dt*velocity, next_core), axis=-1)
