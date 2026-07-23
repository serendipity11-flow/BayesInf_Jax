"""Fitted causal joint-servo model used as the ALOHA physical baseline."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class SecondOrderServo(eqx.Module):
    momentum: jnp.ndarray
    command_gain: jnp.ndarray
    action_to_state_scale: jnp.ndarray
    action_to_state_bias: jnp.ndarray
    history_len: int = eqx.field(static=True)

    def __init__(self, momentum, command_gain, action_to_state_scale,
                 action_to_state_bias, history_len=8):
        self.momentum = jnp.asarray(momentum)
        self.command_gain = jnp.asarray(command_gain)
        self.action_to_state_scale = jnp.asarray(action_to_state_scale)
        self.action_to_state_bias = jnp.asarray(action_to_state_bias)
        self.history_len = history_len

    def predict_next(self, history, action):
        velocity = history[-1] - history[-2]
        command = self.action_to_state_scale * action + self.action_to_state_bias
        return (history[-1] + self.momentum * velocity
                + self.command_gain * (command - history[-1]))

    def rollout_deterministic_history(self, history, controls):
        def step(current, action):
            next_state = self.predict_next(current, action)
            next_history = jnp.concatenate([current[1:], next_state[None]], axis=0)
            return next_history, next_state
        _, states = jax.lax.scan(step, history, controls)
        return states

    def sample_rollout(self, state_history, previous_controls, future_controls,
                       key, n_particles=1):
        del previous_controls, key
        path = self.rollout_deterministic_history(state_history, future_controls)
        return jnp.broadcast_to(path, (n_particles,) + path.shape)


def fit_servo(episodes, actions, train_indices, x_mean, x_std, u_mean, u_std,
              history_len=8):
    """Fit stable per-joint momentum and command gains by least squares."""
    action_to_state_scale = u_std / x_std
    action_to_state_bias = (u_mean - x_mean) / x_std
    features, labels = [], []
    for index in train_indices:
        x = (episodes[index] - x_mean) / x_std
        u = (actions[index] - u_mean) / u_std
        velocity = x[1:-1] - x[:-2]
        command = action_to_state_scale * u[1:] + action_to_state_bias
        features.append(np.stack([velocity, command - x[1:-1]], axis=-1))
        labels.append(x[2:] - x[1:-1])
    phi, y = np.concatenate(features), np.concatenate(labels)
    coefficients = np.stack([
        np.linalg.lstsq(phi[:, joint], y[:, joint], rcond=None)[0]
        for joint in range(y.shape[1])
    ])
    momentum = np.clip(coefficients[:, 0], 0.0, 0.99)
    gain = np.clip(coefficients[:, 1], 1e-4, 1.0)
    return (SecondOrderServo(momentum, gain, action_to_state_scale,
                             action_to_state_bias, history_len),
            momentum, gain)
