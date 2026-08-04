import jax.numpy as jnp
import jax.random as jr

from experiments.jackal_obstacle_avoidance.model import (
    JackalConfig, generate_identification_data, minimum_clearance, nominal_step,
    obstacle_cost, state_from_pose, transition_windows, yaw_from_state,
)


def test_jackal_transition_and_training_alignment():
    states, actions = generate_identification_data(
        jr.PRNGKey(0), episodes=3, steps=8
    )
    windows_x, windows_u = transition_windows(states, actions, 3, 2)
    assert states.shape == (3, 9, 6)
    assert actions.shape == (3, 8, 2)
    assert windows_x.shape[1:] == (5, 6)
    assert windows_u.shape[1:] == (4, 2)


def test_heading_is_continuous_and_obstacle_cost_is_larger_near_obstacle():
    config = JackalConfig()
    state = nominal_step(
        state_from_pose(0, 0, jnp.pi - 0.01), jnp.array([0.5, 0.5]), config
    )
    assert -jnp.pi <= yaw_from_state(state) <= jnp.pi
    cost = obstacle_cost((3.0, 0.0), [(1.0, 0.0, 0.3)], config)
    near = cost(state_from_pose(0.65, 0, 0), jnp.zeros(2), 0)
    far = cost(state_from_pose(0, 1.5, 0), jnp.zeros(2), 0)
    assert near > far
    assert minimum_clearance(
        jnp.stack([state_from_pose(0, 0, 0), state_from_pose(0.8, 0, 0)]),
        [(1.0, 0.0, 0.3)],
    ) < 0
