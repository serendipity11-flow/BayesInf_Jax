"""Episode-safe loading, splitting, and normalization for ALOHA parquet data."""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid


def load_episodes(path: Path):
    # Pandas does not consistently expose Parquet columns for an empty read.
    # Read next_state optimistically for transition datasets, then fall back to
    # the observation-only ALOHA format.
    columns = ["observation.state", "action", "episode_index", "frame_index"]
    try:
        frame = pd.read_parquet(path, columns=columns + ["next_state"])
        has_next_state = True
    except (ArrowInvalid, KeyError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
        has_next_state = False
    frame = frame.sort_values(["episode_index", "frame_index"])
    states, actions, ids = [], [], []
    for episode_id, episode in frame.groupby("episode_index", sort=True):
        x = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
        u = np.stack(episode["action"].to_numpy()).astype(np.float32)
        if len(x) < 2 or len(x) != len(u):
            raise ValueError(f"invalid episode {episode_id}: x={x.shape}, u={u.shape}")
        if has_next_state:
            final_state = np.asarray(
                episode["next_state"].iloc[-1], dtype=np.float32
            )[None]
            x = np.concatenate([x, final_state], axis=0)
            episode_actions = u
        else:
            episode_actions = u[:-1]
        states.append(x); actions.append(episode_actions); ids.append(int(episode_id))
    if not states:
        raise ValueError(f"no episodes found in {path}")
    state_dim, control_dim = states[0].shape[1], actions[0].shape[1]
    if any(x.shape[1] != state_dim for x in states):
        raise ValueError("inconsistent observation.state dimensions")
    if any(u.shape[1] != control_dim for u in actions):
        raise ValueError("inconsistent action dimensions")
    return states, actions, np.asarray(ids), state_dim, control_dim


def load_transition_auxiliary(path: Path, column: str, episode_ids):
    """Load a per-transition label aligned with episodes from ``load_episodes``."""
    frame = pd.read_parquet(
        path, columns=[column, "episode_index", "frame_index"]
    ).sort_values(["episode_index", "frame_index"])
    grouped = {int(key): value for key, value in frame.groupby("episode_index")}
    auxiliary = []
    for episode_id in episode_ids:
        if int(episode_id) not in grouped:
            raise ValueError(f"missing {column} values for episode {episode_id}")
        values = np.stack(grouped[int(episode_id)][column].to_numpy()).astype(
            np.float32
        )
        auxiliary.append(values)
    return auxiliary


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
