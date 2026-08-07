"""Collect the seven-state continuous-yaw off-road dataset."""

import argparse
import json
from pathlib import Path

import numpy as np

from .env_psi import OffroadRoverPsiEnv


def collect(args):
    env, rng = OffroadRoverPsiEnv(seed=args.seed), np.random.default_rng(args.seed)
    states, actions, next_states, episode_ids = [], [], [], []
    for episode in range(args.episodes):
        state = env.reset(pose=(0, 0, rng.uniform(-np.pi, np.pi)))
        command = np.zeros(2)
        for frame in range(args.steps):
            if frame % args.hold_steps == 0:
                command = np.asarray((rng.uniform(0.25, 1.0), rng.uniform(-1, 1)))
            action = np.clip(command + rng.normal(0, args.exploration, 2), -1, 1).astype(np.float32)
            next_state = env.step(action)
            states.append(state); actions.append(action); next_states.append(next_state)
            episode_ids.append(episode); state = next_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{"observation.state": np.stack(states),
        "action": np.stack(actions), "next_observation.state": np.stack(next_states),
        "episode_index": np.asarray(episode_ids)})
    metadata = {"episodes": args.episodes, "steps": args.steps, "state_dim": 7,
                "state_order": ["x", "y", "psi", "v_long", "v_lat", "yaw_rate", "wheel_slip"],
                "action_order": ["throttle", "steering"], "yaw": "continuous unwrapped", "seed": args.seed}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/offroad_flowsde/transitions_psi.npz"))
    p.add_argument("--episodes", type=int, default=500); p.add_argument("--steps", type=int, default=300)
    p.add_argument("--hold-steps", type=int, default=8); p.add_argument("--exploration", type=float, default=.08)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    print(json.dumps(collect(parser().parse_args()), indent=2))
