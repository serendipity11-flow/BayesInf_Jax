"""Split an existing DPPO trajectory NPZ without crossing episode boundaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from .data import split_and_write


def load_episodes(path: Path):
    data = np.load(path, allow_pickle=False)
    required = {"states", "actions", "traj_lengths"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    episodes, offset = [], 0
    for length in data["traj_lengths"].astype(int):
        end = offset + length
        episodes.append({
            "states": np.asarray(data["states"][offset:end], dtype=np.float32),
            "actions": np.asarray(data["actions"][offset:end], dtype=np.float32),
            "rewards": np.asarray(
                data["rewards"][offset:end] if "rewards" in data else np.zeros(length),
                dtype=np.float32,
            ),
            "terminals": np.asarray(
                data["terminals"][offset:end] if "terminals" in data else
                np.r_[np.zeros(max(length - 1, 0), dtype=bool), True],
                dtype=bool,
            ),
        })
        offset = end
    if offset != len(data["states"]) or offset != len(data["actions"]):
        raise ValueError("sum(traj_lengths) does not match states/actions")
    return episodes


def preserve_preprocessing_split(
    path: Path, normalization: Path, output: Path, validation_fraction: float, seed: int, source: str
):
    """Split already-normalized official data without applying normalization twice."""
    data = np.load(path, allow_pickle=False)
    lengths = data["traj_lengths"].astype(int)
    if len(lengths) < 2:
        raise ValueError("at least two trajectories are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    n_val = min(max(1, int(round(len(lengths) * validation_fraction))), len(lengths) - 1)
    val_ids, train_ids = order[:n_val], order[n_val:]
    starts = np.r_[0, np.cumsum(lengths)]

    def pack(ids):
        indices = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in ids])
        values = {}
        for key in ("states", "actions", "rewards", "terminals"):
            if key in data:
                values[key] = data[key][indices]
        values["traj_lengths"] = lengths[ids]
        return values

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train.npz", **pack(train_ids))
    np.savez_compressed(output / "val.npz", **pack(val_ids))
    shutil.copyfile(normalization, output / "normalization.npz")
    manifest = {
        "source": source,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "preprocessing": "preserved official DPPO full-dataset min/max normalization",
        "train_episode_ids": train_ids.tolist(),
        "validation_episode_ids": val_ids.tolist(),
        "train_episodes": len(train_ids),
        "validation_episodes": len(val_ids),
        "train_transitions": int(lengths[train_ids].sum()),
        "validation_transitions": int(lengths[val_ids].sum()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", default="official-dppo-preprocessed")
    p.add_argument("--normalization", type=Path)
    p.add_argument("--preserve-preprocessing", action="store_true")
    a = p.parse_args()
    if a.preserve_preprocessing:
        if a.normalization is None:
            p.error("--normalization is required with --preserve-preprocessing")
        preserve_preprocessing_split(
            a.input, a.normalization, a.output, a.validation_fraction, a.seed, a.source
        )
    else:
        split_and_write(load_episodes(a.input), a.output, a.validation_fraction, a.seed, a.source)
