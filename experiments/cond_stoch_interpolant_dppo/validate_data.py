"""Strict validation of prepared train/validation artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def validate(root: Path, obs_dim: int, action_dim: int, horizon: int):
    report = {}
    manifest = json.loads((root / "manifest.json").read_text())
    train_ids, val_ids = set(manifest["train_episode_ids"]), set(manifest["validation_episode_ids"])
    assert train_ids.isdisjoint(val_ids), "episode leakage between splits"
    norm = np.load(root / "normalization.npz", allow_pickle=False)
    required_norm = (
        ("obs_mean", "obs_std", "action_min", "action_max", "action_scale")
        if "obs_mean" in norm else ("obs_min", "obs_max", "action_min", "action_max")
    )
    for key in required_norm:
        assert key in norm and np.isfinite(norm[key]).all()
    obs_key = "obs_mean" if "obs_mean" in norm else "obs_min"
    assert norm[obs_key].shape == (obs_dim,)
    assert norm["action_min"].shape == (action_dim,)
    for split in ("train", "val"):
        data = np.load(root / f"{split}.npz", allow_pickle=False)
        assert set(("states", "actions", "rewards", "terminals", "traj_lengths")) <= set(data.files)
        n = int(data["traj_lengths"].sum())
        assert n == len(data["states"]) == len(data["actions"]) == len(data["rewards"]) == len(data["terminals"])
        assert data["states"].shape == (n, obs_dim)
        assert data["actions"].shape == (n, action_dim)
        assert np.isfinite(data["states"]).all() and np.isfinite(data["actions"]).all()
        assert np.all(data["traj_lengths"] >= horizon), "episode shorter than action horizon"
        chunks = int(np.sum(data["traj_lengths"] - horizon + 1))
        report[split] = {
            "episodes": len(data["traj_lengths"]), "transitions": n, "chunks": chunks,
            "state_dtype": str(data["states"].dtype), "action_dtype": str(data["actions"].dtype),
            "action_min": float(data["actions"].min()), "action_max": float(data["actions"].max()),
        }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--obs-dim", type=int, required=True)
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--horizon", type=int, default=4)
    a = p.parse_args()
    validate(a.root, a.obs_dim, a.action_dim, a.horizon)
