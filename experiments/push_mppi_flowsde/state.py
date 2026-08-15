"""Canonical local state layout and history-window utilities."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


HISTORY, ACTION_DIM, STATE_DIM = 3, 3, 24
INPUT_DIM = HISTORY * STATE_DIM + (HISTORY - 1) * ACTION_DIM
EE_POSITION = slice(0, 3)
EE_VELOCITY = slice(3, 6)
OBJECT_POSITION = slice(6, 9)
OBJECT_YAW = 9
OBJECT_VELOCITY = slice(10, 13)
OBJECT_OMEGA = slice(13, 16)
OBJECT_MOVEMENT = slice(10, 16)
CONTACT_VECTOR = slice(16, 19)
GATE_VECTOR = slice(19, 22)
WALL_DISTANCE = slice(22, 24)


def pack_history(states, actions):
    """Pack newest-first (..., 3, 24) states and (..., 2, 3) actions."""
    if states.shape[-2:] != (HISTORY, STATE_DIM):
        raise ValueError(f"states must end with {(HISTORY, STATE_DIM)}")
    if actions.shape[-2:] != (HISTORY - 1, ACTION_DIM):
        raise ValueError(f"actions must end with {(HISTORY - 1, ACTION_DIM)}")
    return jnp.concatenate((states.reshape(states.shape[:-2] + (-1,)),
                            actions.reshape(actions.shape[:-2] + (-1,))), -1)


def latest_frame(history):
    return history[..., :STATE_DIM]


def update_history(history, next_state, action):
    states = history[..., :HISTORY * STATE_DIM].reshape(
        history.shape[:-1] + (HISTORY, STATE_DIM)
    )
    actions = history[..., HISTORY * STATE_DIM:].reshape(
        history.shape[:-1] + (HISTORY - 1, ACTION_DIM)
    )
    return pack_history(jnp.concatenate((next_state[..., None, :], states[..., :-1, :]), -2),
                        jnp.concatenate((action[..., None, :], actions[..., :-1, :]), -2))


@dataclass
class HistoryBuffer:
    states: np.ndarray
    actions: np.ndarray

    @classmethod
    def from_state(cls, state):
        state = np.asarray(state, dtype=np.float32)
        return cls(np.repeat(state[None], HISTORY, 0),
                   np.zeros((HISTORY - 1, ACTION_DIM), np.float32))

    @property
    def vector(self):
        return np.asarray(pack_history(self.states, self.actions), dtype=np.float32)

    def append(self, state, action):
        self.states[1:] = self.states[:-1]; self.states[0] = state
        self.actions[1:] = self.actions[:-1]; self.actions[0] = action
