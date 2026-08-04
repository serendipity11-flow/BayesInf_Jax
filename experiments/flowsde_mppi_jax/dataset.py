"""Collect 1000 scripted trajectories in a LeRobot-compatible table."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .env_wrapper import NoisyPushingEnv
from .state import HistoryBuffer, OBJECT_POSITION


def scripted_action(state, target, rng, exploration):
    ee, obj = state[:3], state[OBJECT_POSITION]
    direction = target - obj; direction[2] = 0
    direction /= max(np.linalg.norm(direction), 1e-6)
    behind = obj - 0.13 * direction
    waypoint = behind if np.linalg.norm(ee - behind) > 0.06 else obj + 0.10 * direction
    return np.clip(6.0 * (waypoint - ee) + exploration * rng.normal(size=3),
                   -1.2, 1.2).astype(np.float32)


def collect(args):
    env, rng = NoisyPushingEnv(args.model, seed=args.seed), np.random.default_rng(args.seed)
    target, rows = np.asarray(args.target, dtype=np.float32), []
    for episode in range(args.episodes):
        state = env.reset(randomize=True); history = HistoryBuffer.from_state(state)
        for frame in range(args.steps):
            action = scripted_action(state, target, rng, args.exploration)
            model_input = history.vector.copy(); next_state = env.step(action)
            rows.append({"episode_index": episode, "frame_index": frame,
                         "observation.state": model_input.tolist(), "action": action.tolist(),
                         "next_observation.state": next_state.tolist()})
            history.append(next_state, action); state = next_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if args.output.suffix in {".h5", ".hdf5"}:
        with h5py.File(args.output, "w") as handle:
            for name in ("observation.state", "action", "next_observation.state"):
                handle.create_dataset(name, data=np.stack(table[name].map(np.asarray)),
                                      compression="gzip", shuffle=True)
            handle.create_dataset("episode_index", data=table.episode_index.to_numpy())
            handle.create_dataset("frame_index", data=table.frame_index.to_numpy())
            handle.attrs["format"] = "LeRobot transition dataset"
    else:
        table.to_parquet(args.output, index=False)
    metadata = {"format": "LeRobot-compatible transition table", "episodes": args.episodes,
                "steps": args.steps, "samples": len(rows), "target": target.tolist(),
                "model": str(Path(args.model).resolve())}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/flowsde_mppi_jax/pushing.h5"))
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--target", type=float, nargs=3, default=(0.68, 0.0, 0.055))
    p.add_argument("--exploration", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    print(json.dumps(collect(parser().parse_args()), indent=2))
