"""Jackal dynamics data and risk-sensitive obstacle-avoidance MPC utilities.

State: ``[x, y, sin(yaw), cos(yaw), linear_velocity, angular_velocity]``.
Action: ``[linear_velocity_command, angular_velocity_command]``.
"""

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


@dataclass(frozen=True)
class JackalConfig:
    dt: float = 0.1
    linear_time_constant: float = 0.35
    angular_time_constant: float = 0.25
    linear_limits: tuple = (0.0, 1.0)
    angular_limits: tuple = (-1.5, 1.5)
    robot_radius: float = 0.27
    safety_margin: float = 0.18


@dataclass(frozen=True)
class HighFidelityJackalConfig:
    """Hidden plant parameters unavailable to the nominal-model controller."""

    dt: float = 0.1
    wheel_base: float = 0.43
    motor_time_constant: float = 0.32
    payload_factor: float = 0.2
    command_delay_steps: int = 1
    base_traction: float = 0.85
    terrain_amplitude: float = 0.12
    terrain_frequency: float = 1.3
    terrain_phase: float = 0.0
    left_wheel_slip: float = 0.05
    right_wheel_slip: float = 0.05
    lateral_slip: float = 0.04
    disturbance_x: float = 0.0
    disturbance_y: float = 0.0
    disturbance_yaw: float = 0.0
    process_noise: float = 0.01
    robot_radius: float = 0.27
    safety_margin: float = 0.18


def state_from_pose(x, y, yaw, linear_velocity=0.0, angular_velocity=0.0):
    """Construct the learning state without a discontinuous yaw coordinate."""
    return jnp.asarray([
        x, y, jnp.sin(yaw), jnp.cos(yaw), linear_velocity, angular_velocity
    ])


def yaw_from_state(state):
    return jnp.arctan2(state[..., 2], state[..., 3])


def nominal_step(state, action, config=JackalConfig()):
    """A lagged unicycle used only to generate a runnable demonstration."""
    v_cmd = jnp.clip(action[0], *config.linear_limits)
    w_cmd = jnp.clip(action[1], *config.angular_limits)
    v = state[4] + config.dt * (v_cmd - state[4]) / config.linear_time_constant
    w = state[5] + config.dt * (w_cmd - state[5]) / config.angular_time_constant
    yaw = yaw_from_state(state) + config.dt * w
    x = state[0] + config.dt * v * jnp.cos(yaw)
    y = state[1] + config.dt * v * jnp.sin(yaw)
    return state_from_pose(x, y, yaw, v, w)


def terrain_traction(position, config: HighFidelityJackalConfig):
    """Smooth spatial traction field, standing in for changing terrain."""
    pattern = jnp.sin(
        config.terrain_frequency * position[0] + config.terrain_phase
    ) * jnp.cos(
        0.7 * config.terrain_frequency * position[1] - config.terrain_phase
    )
    return jnp.clip(
        config.base_traction + config.terrain_amplitude * pattern, 0.15, 1.05
    )


def high_fidelity_step(state, delayed_action, key,
                       config: HighFidelityJackalConfig):
    """Differential-drive plant with hidden terrain, slip, payload and force."""
    v_command = jnp.clip(delayed_action[0], 0.0, 1.0)
    w_command = jnp.clip(delayed_action[1], -1.5, 1.5)
    nominal_left = v_command - 0.5 * config.wheel_base * w_command
    nominal_right = v_command + 0.5 * config.wheel_base * w_command
    traction = terrain_traction(state[:2], config)
    left = traction * (1.0 - config.left_wheel_slip) * nominal_left
    right = traction * (1.0 - config.right_wheel_slip) * nominal_right
    target_v = 0.5 * (left + right)
    target_w = (right - left) / config.wheel_base
    # Payload increases inertia without changing the steady-state kinematics.
    tau = config.motor_time_constant * (1.0 + config.payload_factor)
    v = state[4] + config.dt * (target_v - state[4]) / tau
    w = state[5] + config.dt * (target_w - state[5]) / tau
    noise_v, noise_w, noise_lateral = jr.normal(key, (3,))
    v = v + config.process_noise * noise_v
    w = w + config.process_noise * noise_w + config.disturbance_yaw
    yaw = yaw_from_state(state) + config.dt * w
    lateral = config.lateral_slip * jnp.abs(v) * jnp.sin(
        config.terrain_frequency * state[0] - config.terrain_phase
    ) + config.process_noise * 0.25 * noise_lateral
    x_velocity = (
        v * jnp.cos(yaw) - lateral * jnp.sin(yaw) + config.disturbance_x
    )
    y_velocity = (
        v * jnp.sin(yaw) + lateral * jnp.cos(yaw) + config.disturbance_y
    )
    return state_from_pose(
        state[0] + config.dt * x_velocity,
        state[1] + config.dt * y_velocity,
        yaw, v, w,
    )


def delayed_high_fidelity_step(state, command_queue, action, key, config):
    """Apply the oldest queued command and append the newly issued command."""
    delayed_action = command_queue[0]
    next_state = high_fidelity_step(state, delayed_action, key, config)
    next_queue = jnp.concatenate([command_queue[1:], action[None]], axis=0)
    return next_state, next_queue


class ExactJackalModel:
    """CEM-compatible stochastic rollout wrapper around the known simulator."""

    def __init__(self, config=JackalConfig(), process_noise=0.01):
        self.config = config
        self.process_noise = process_noise
        self.history_len = 1

    def rollout_deterministic(self, state_context, actions):
        state = jnp.asarray(state_context)
        if state.ndim == 2:
            state = state[-1]
        def step(state, action):
            state = nominal_step(state, action, self.config)
            return state, state
        return jax.lax.scan(step, state, actions)[1]

    def sample_rollout(self, state_history, control_history, actions, key,
                       n_particles=1):
        del control_history

        def particle(particle_key):
            state = state_history[-1]
            noise_keys = jr.split(particle_key, actions.shape[0])

            def step(state, inputs):
                action, noise_key = inputs
                state = nominal_step(state, action, self.config)
                state = state.at[4:].add(
                    self.process_noise * jr.normal(noise_key, (2,))
                )
                return state, state
            return jax.lax.scan(step, state, (actions, noise_keys))[1]
        return jax.vmap(particle)(jr.split(key, n_particles))


def generate_identification_data(key, episodes=128, steps=100,
                                 config=JackalConfig(), process_noise=0.01):
    """Generate persistently exciting trajectories for a smoke-test/demo."""
    episode_keys = jr.split(key, episodes)

    def episode(episode_key):
        command_key, noise_key = jr.split(episode_key)
        raw = jr.uniform(command_key, (steps, 2), minval=-1.0, maxval=1.0)
        actions = raw.at[:, 0].set((raw[:, 0] + 1.0) / 2.0)
        actions = actions.at[:, 1].multiply(config.angular_limits[1])

        def transition(state, inputs):
            action, noise = inputs
            next_state = nominal_step(state, action, config)
            # Noise is added to velocities; pose then remains geometrically valid.
            next_state = next_state.at[4:].add(process_noise * noise)
            return next_state, next_state

        initial = state_from_pose(0.0, 0.0, jr.uniform(episode_key, (), minval=-jnp.pi,
                                                       maxval=jnp.pi))
        _, next_states = jax.lax.scan(
            transition, initial, (actions, jr.normal(noise_key, (steps, 2)))
        )
        return jnp.concatenate([initial[None], next_states]), actions

    return jax.vmap(episode)(episode_keys)


def sample_training_plant(key):
    """Draw an in-distribution high-fidelity plant for identification."""
    values = jr.uniform(key, (10,))
    return HighFidelityJackalConfig(
        payload_factor=float(0.05 + 0.45 * values[0]),
        command_delay_steps=int(1 + (values[1] > 0.55)),
        base_traction=float(0.68 + 0.27 * values[2]),
        terrain_amplitude=float(0.04 + 0.14 * values[3]),
        terrain_frequency=float(0.8 + 1.0 * values[4]),
        terrain_phase=float(2 * jnp.pi * values[5]),
        left_wheel_slip=float(0.02 + 0.12 * values[6]),
        right_wheel_slip=float(0.02 + 0.12 * values[7]),
        lateral_slip=float(0.01 + 0.07 * values[8]),
        disturbance_x=float(0.025 * (2 * values[9] - 1)),
        process_noise=0.012,
    )


def sample_robust_training_plant(key):
    """Broad identification distribution spanning the deployment envelope."""
    values = jr.uniform(key, (13,))
    return HighFidelityJackalConfig(
        motor_time_constant=float(0.25 + 0.15 * values[0]),
        payload_factor=float(0.05 + 1.20 * values[1]),
        command_delay_steps=int(1 + jnp.floor(4 * values[2])),
        base_traction=float(0.45 + 0.50 * values[3]),
        terrain_amplitude=float(0.03 + 0.19 * values[4]),
        terrain_frequency=float(0.7 + 1.4 * values[5]),
        terrain_phase=float(2 * jnp.pi * values[6]),
        left_wheel_slip=float(0.01 + 0.20 * values[7]),
        right_wheel_slip=float(0.01 + 0.20 * values[8]),
        lateral_slip=float(0.01 + 0.13 * values[9]),
        disturbance_x=float(0.06 * (values[10] - 0.5)),
        disturbance_y=float(0.24 * (values[11] - 0.5)),
        disturbance_yaw=float(0.10 * (values[12] - 0.5)),
        process_noise=0.012,
    )


def generate_high_fidelity_identification_data(
        key, episodes=256, steps=120, *, robust=False):
    """Generate randomized plant data; parameters change between episodes."""
    states, actions = [], []
    for episode_index, episode_key in enumerate(jr.split(key, episodes)):
        plant_key, command_key, initial_key, rollout_key = jr.split(episode_key, 4)
        config = (
            sample_robust_training_plant(plant_key)
            if robust else sample_training_plant(plant_key)
        )
        # Piecewise-smoothed excitation is closer to commands sent to a robot
        # than independent white noise while retaining persistent excitation.
        # Mix stepwise and held commands. MPC proposes both smooth sequences
        # and abrupt candidate changes, so identification must cover both.
        hold_steps = (1, 2, 5)[episode_index % 3] if robust else 5
        knots = max(2, (steps + hold_steps - 1) // hold_steps)
        raw = jr.uniform(command_key, (knots, 2), minval=-1.0, maxval=1.0)
        raw = raw.at[:, 0].set(0.05 + 0.9 * (raw[:, 0] + 1.0) / 2.0)
        raw = raw.at[:, 1].multiply(1.35)
        episode_actions = jnp.repeat(raw, hold_steps, axis=0)[:steps]
        initial = state_from_pose(
            0.0, 0.0, jr.uniform(initial_key, (), minval=-jnp.pi, maxval=jnp.pi)
        )
        queue = jnp.zeros((max(1, config.command_delay_steps), 2))
        episode_states = [initial]
        state = initial
        for action, step_key in zip(
                episode_actions, jr.split(rollout_key, steps)):
            state, queue = delayed_high_fidelity_step(
                state, queue, action, step_key, config
            )
            episode_states.append(state)
        states.append(jnp.stack(episode_states))
        actions.append(episode_actions)
    return jnp.stack(states), jnp.stack(actions)


def generate_control_task_identification_data(key, episodes=256, steps=120):
    """Balanced identification data for the five evaluated plant families."""
    configs = (
        HighFidelityJackalConfig(),
        HighFidelityJackalConfig(
            command_delay_steps=4, base_traction=0.82,
            terrain_amplitude=0.10, terrain_phase=2.2,
        ),
        HighFidelityJackalConfig(
            payload_factor=1.15, motor_time_constant=0.38,
            base_traction=0.80, terrain_phase=0.4,
        ),
        HighFidelityJackalConfig(
            disturbance_y=0.11, disturbance_yaw=0.045,
            base_traction=0.82, terrain_phase=2.8,
        ),
        HighFidelityJackalConfig(
            payload_factor=0.95, command_delay_steps=3,
            base_traction=0.48, terrain_amplitude=0.20,
            left_wheel_slip=0.19, right_wheel_slip=0.11,
            lateral_slip=0.12, disturbance_y=-0.08,
            disturbance_yaw=-0.035, terrain_phase=1.7,
        ),
    )
    states, actions = [], []
    for episode_index, episode_key in enumerate(jr.split(key, episodes)):
        command_key, initial_key, rollout_key = jr.split(episode_key, 3)
        config = configs[episode_index % len(configs)]
        raw = jr.uniform(command_key, (steps, 2), minval=-1.0, maxval=1.0)
        episode_actions = raw.at[:, 0].set(
            0.05 + 0.9 * (raw[:, 0] + 1.0) / 2.0
        )
        episode_actions = episode_actions.at[:, 1].multiply(1.35)
        initial = state_from_pose(
            0.0, 0.0,
            jr.uniform(initial_key, (), minval=-0.8, maxval=0.8),
        )
        queue = jnp.zeros((max(1, config.command_delay_steps), 2))
        episode_states = [initial]
        state = initial
        for action, step_key in zip(
                episode_actions, jr.split(rollout_key, steps)):
            state, queue = delayed_high_fidelity_step(
                state, queue, action, step_key, config
            )
            episode_states.append(state)
        states.append(jnp.stack(episode_states))
        actions.append(episode_actions)
    return jnp.stack(states), jnp.stack(actions)


def transition_windows(states, actions, history_len, rollout_steps=1):
    """Create aligned ``H+L`` state and ``H+L-1`` control training windows."""
    states, actions = np.asarray(states), np.asarray(actions)
    length = history_len + rollout_steps
    xs, us = [], []
    for episode_states, episode_actions in zip(states, actions):
        for start in range(episode_states.shape[0] - length + 1):
            xs.append(episode_states[start:start + length])
            us.append(episode_actions[start:start + length - 1])
    if not xs:
        raise ValueError("episodes are too short for history_len + rollout_steps")
    return jnp.asarray(np.stack(xs)), jnp.asarray(np.stack(us))


def obstacle_cost(goal, obstacles: Sequence[Sequence[float]], config=JackalConfig(),
                  goal_weight=4.0, heading_weight=0.15, control_weight=0.03,
                  collision_weight=250.0, progress_weight=0.4):
    """Return an MPC stage cost for circular obstacles ``(x, y, radius)``."""
    goal = jnp.asarray(goal)
    obstacles = jnp.asarray(obstacles).reshape((-1, 3))

    def cost(state, action, time):
        del time
        position = state[:2]
        goal_delta = goal - position
        goal_distance = jnp.linalg.norm(goal_delta)
        desired_yaw = jnp.arctan2(goal_delta[1], goal_delta[0])
        heading_error = jnp.arctan2(
            jnp.sin(yaw_from_state(state) - desired_yaw),
            jnp.cos(yaw_from_state(state) - desired_yaw),
        )
        center_distance = jnp.linalg.norm(
            position[None] - obstacles[:, :2], axis=-1
        )
        clearance = center_distance - obstacles[:, 2]
        required = config.robot_radius + config.safety_margin
        # Smooth, nonzero gradient before collision; compatible with sampling MPC.
        collision = jnp.sum(jax.nn.softplus((required - clearance) / 0.05) ** 2)
        reverse = jax.nn.relu(-action[0]) ** 2
        return (
            goal_weight * goal_distance ** 2
            + heading_weight * heading_error ** 2
            + control_weight * jnp.sum(action ** 2)
            + collision_weight * collision
            + progress_weight * reverse
        )
    return cost


def minimum_clearance(states, obstacles):
    obstacles = jnp.asarray(obstacles).reshape((-1, 3))
    distance = jnp.linalg.norm(
        states[:, None, :2] - obstacles[None, :, :2], axis=-1
    ) - obstacles[None, :, 2]
    return jnp.min(distance)
