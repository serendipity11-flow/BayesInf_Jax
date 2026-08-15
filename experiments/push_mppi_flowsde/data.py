"""Episode-safe data loading and normalization."""

import h5py
import numpy as np
import pandas as pd

from .state import STATE_DIM


def load(path):
    if str(path).endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as handle:
            histories = handle["observation.state"][:].astype(np.float32)
            actions = handle["action"][:].astype(np.float32)
            next_states = handle["next_observation.state"][:].astype(np.float32)
            episodes = handle["episode_index"][:]
        return histories, actions, next_states - histories[:, :STATE_DIM], episodes
    table = pd.read_parquet(path)
    histories = np.stack(table["observation.state"].map(np.asarray)).astype(np.float32)
    actions = np.stack(table["action"].map(np.asarray)).astype(np.float32)
    next_states = np.stack(table["next_observation.state"].map(np.asarray)).astype(np.float32)
    deltas = next_states - histories[:, :STATE_DIM]
    return histories, actions, deltas, table.episode_index.to_numpy()


def split(episodes, seed=0, validation_fraction=0.1, test_fraction=0.1):
    unique = np.random.default_rng(seed).permutation(np.unique(episodes))
    nt, nv = max(1, round(len(unique) * test_fraction)), max(1, round(len(unique) * validation_fraction))
    return unique[nt + nv:], unique[nt:nt + nv], unique[:nt]


def statistics(histories, actions, deltas):
    return {"history_mean": histories.mean(0), "history_std": histories.std(0) + 1e-6,
            "action_mean": actions.mean(0), "action_std": actions.std(0) + 1e-6,
            "delta_mean": deltas.mean(0), "delta_std": deltas.std(0) + 1e-6}


def normalize(histories, actions, deltas, stats):
    return ((histories - stats["history_mean"]) / stats["history_std"],
            (actions - stats["action_mean"]) / stats["action_std"],
            (deltas - stats["delta_mean"]) / stats["delta_std"])
