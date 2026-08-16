"""Shared closed-loop objective for differential-drive MPC comparisons."""
import jax.numpy as jnp
import numpy as np


def _trajectory_cost(array_module, paths, actions, goal, heading_weight=0.0):
    """Cost one path per leading batch index.

    ``paths`` excludes the initial state and has shape ``(..., horizon, state)``;
    ``actions`` has the corresponding shape ``(..., horizon, 2)``.
    """
    distance = array_module.linalg.norm(paths[..., :2] - goal, axis=-1)
    effort = .015 * array_module.mean(
        actions[..., 0] ** 2 + .2 * actions[..., 1] ** 2, axis=-1)
    smooth = .02 * array_module.mean(
        array_module.diff(actions, axis=-2) ** 2, axis=(-2, -1))
    direction = goal - paths[..., :2]
    desired_heading = array_module.arctan2(direction[..., 1], direction[..., 0])
    heading_cost = heading_weight * array_module.mean(
        1.0 - array_module.cos(paths[..., 2] - desired_heading), axis=-1)
    return (array_module.mean(distance, axis=-1) + 2.5 * distance[..., -1]
            + effort + smooth + heading_cost)


def trajectory_cost_jax(paths, actions, goal, heading_weight=0.0):
    return _trajectory_cost(jnp, paths, actions, goal, heading_weight)


def trajectory_cost_numpy(paths, actions, goal, heading_weight=0.0):
    return _trajectory_cost(np, paths, actions, goal, heading_weight)
