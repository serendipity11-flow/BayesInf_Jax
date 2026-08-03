from types import SimpleNamespace

import numpy as np

from experiments.cond_stoch_interpolant_dppo.check_door_compatibility import (
    check_prepared_dataset,
)


class Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class FakeDoor:
    observation_space = Box([-10, -10], [10, 10])
    action_space = Box([-1], [1])
    spec = SimpleNamespace(max_episode_steps=4)

    def reset(self, seed=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {"success": False}


def _write_dataset(root, lengths):
    np.savez(
        root / "normalization.npz",
        obs_mean=np.zeros(2), obs_std=np.ones(2),
        action_min=np.array([-1.0]), action_max=np.array([1.0]),
        action_scale=np.array([2.0]),
    )
    for split, split_lengths in zip(("train", "val"), lengths):
        count = sum(split_lengths)
        np.savez(
            root / f"{split}.npz",
            states=np.zeros((count, 2)), actions=np.zeros((count, 1)),
            rewards=np.zeros(count), terminals=np.zeros(count, dtype=bool),
            traj_lengths=np.asarray(split_lengths),
        )


def test_compatible_contract(tmp_path):
    _write_dataset(tmp_path, ([4], [3]))
    report = check_prepared_dataset(tmp_path, FakeDoor())
    assert report["compatible"]


def test_detects_episode_horizon_mismatch(tmp_path):
    _write_dataset(tmp_path, ([5], [3]))
    report = check_prepared_dataset(tmp_path, FakeDoor())
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert "episode_horizon" in failed
    assert not report["compatible"]
