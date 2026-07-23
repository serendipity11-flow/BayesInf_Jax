"""Prepare episode-disjoint DPPO NPZ train/validation datasets from Minari."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _pack(episodes: list[dict[str, np.ndarray]], stats: dict[str, np.ndarray]):
    states = np.concatenate([ep["states"] for ep in episodes])
    actions = np.concatenate([ep["actions"] for ep in episodes])
    return {
        "states": ((states - stats["obs_mean"]) / stats["obs_std"]).astype(np.float32),
        "actions": (2 * (actions - stats["action_min"]) / stats["action_scale"] - 1)
        .clip(-1, 1).astype(np.float32),
        "rewards": np.concatenate([ep["rewards"] for ep in episodes]).astype(np.float32),
        "terminals": np.concatenate([ep["terminals"] for ep in episodes]).astype(bool),
        "traj_lengths": np.asarray([len(ep["actions"]) for ep in episodes], dtype=np.int64),
    }


def split_and_write(
    episodes: list[dict[str, np.ndarray]],
    output: Path,
    validation_fraction: float,
    seed: int,
    source: str,
) -> None:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required for a disjoint split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(len(episodes) * validation_fraction)))
    n_val = min(n_val, len(episodes) - 1)
    val_ids, train_ids = order[:n_val], order[n_val:]
    train = [episodes[i] for i in train_ids]
    val = [episodes[i] for i in val_ids]
    train_states = np.concatenate([ep["states"] for ep in train])
    train_actions = np.concatenate([ep["actions"] for ep in train])
    action_min, action_max = train_actions.min(0), train_actions.max(0)
    stats = {
        "obs_mean": train_states.mean(0),
        "obs_std": train_states.std(0).clip(1e-6),
        "action_min": action_min,
        "action_max": action_max,
        "action_scale": (action_max - action_min).clip(1e-6),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train.npz", **_pack(train, stats))
    np.savez_compressed(output / "val.npz", **_pack(val, stats))
    np.savez(output / "normalization.npz", **stats)
    manifest = {
        "source": source,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "train_episode_ids": train_ids.tolist(),
        "validation_episode_ids": val_ids.tolist(),
        "train_episodes": len(train),
        "validation_episodes": len(val),
        "train_transitions": int(sum(len(ep["actions"]) for ep in train)),
        "validation_transitions": int(sum(len(ep["actions"]) for ep in val)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def convert(
    dataset_id: str,
    output: Path,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> None:
    import minari

    dataset = minari.load_dataset(dataset_id, download=True)
    episodes = []
    for ep in dataset.iterate_episodes():
        # Minari has T+1 observations and T actions.
        n = len(ep.actions)
        observations = ep.observations
        if isinstance(observations, dict):
            if "observation" not in observations:
                raise ValueError(f"unsupported observation keys: {observations.keys()}")
            observations = observations["observation"]
        episodes.append({
            "states": np.asarray(observations[:n], dtype=np.float32),
            "actions": np.asarray(ep.actions, dtype=np.float32),
            "rewards": np.asarray(ep.rewards, dtype=np.float32),
            "terminals": np.asarray(ep.terminations, dtype=bool),
        })
    split_and_write(episodes, output, validation_fraction, seed, dataset_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="D4RL/door/human-v2")
    p.add_argument("--output", type=Path, default=Path("data/dppo/door-human-v2"))
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    convert(a.dataset, a.output, a.validation_fraction, a.seed)
