"""Episode-safe loading, splitting, and normalization for ALOHA parquet data."""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd


def load_episodes(path: Path):
    frame = pd.read_parquet(
        path, columns=["observation.state", "action", "episode_index", "frame_index"]
    ).sort_values(["episode_index", "frame_index"])
    states, actions, ids = [], [], []
    for episode_id, episode in frame.groupby("episode_index", sort=True):
        x = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
        u = np.stack(episode["action"].to_numpy()).astype(np.float32)
        if len(x) < 2 or len(x) != len(u):
            raise ValueError(f"invalid episode {episode_id}: x={x.shape}, u={u.shape}")
        states.append(x); actions.append(u[:-1]); ids.append(int(episode_id))
    if not states:
        raise ValueError(f"no episodes found in {path}")
    state_dim, control_dim = states[0].shape[1], actions[0].shape[1]
    if any(x.shape[1] != state_dim for x in states):
        raise ValueError("inconsistent observation.state dimensions")
    if any(u.shape[1] != control_dim for u in actions):
        raise ValueError("inconsistent action dimensions")
    return states, actions, np.asarray(ids), state_dim, control_dim


def split_episode_indices(n_episodes, seed):
    order = np.asarray(jr.permutation(jr.PRNGKey(seed), n_episodes))
    n_train = max(1, int(0.8 * n_episodes))
    n_validation = max(1, int(0.1 * n_episodes))
    return (order[:n_train], order[n_train:n_train + n_validation],
            order[n_train + n_validation:])


def normalization(episodes, actions, indices):
    x = np.concatenate([episodes[i] for i in indices])
    u = np.concatenate([actions[i] for i in indices])
    return tuple(jnp.asarray(value) for value in (
        x.mean(0), np.maximum(x.std(0), 1e-4),
        u.mean(0), np.maximum(u.std(0), 1e-4),
    ))


def augment_with_causal_velocity(episodes, dt=0.02, window=3):
    """Return [position, velocity] episodes using backward causal differences.

    ``velocity[t] = (q[t] - q[t-window]) / (window*dt)`` where possible;
    shorter backward differences are used at the beginning of each episode.
    No future sample contributes to an estimate.
    """
    if dt <= 0 or window < 1:
        raise ValueError("dt must be positive and velocity window at least one")
    augmented = []
    for positions in episodes:
        velocity = np.zeros_like(positions)
        for step in range(1, len(positions)):
            lag = min(step, window)
            velocity[step] = (positions[step] - positions[step - lag]) / (lag * dt)
        velocity[0] = velocity[1]
        augmented.append(np.concatenate([positions, velocity], axis=-1))
    return augmented
