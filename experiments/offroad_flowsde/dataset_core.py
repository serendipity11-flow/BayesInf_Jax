"""Collect off-road transitions with observable XY context and 14-D core."""

import argparse
import json
from pathlib import Path

import numpy as np

from .env_core import OffroadRoverCoreEnv

CORE_ORDER = ["psi", "v_long", "v_lat", "yaw_rate", "omega_fl", "omega_fr",
              "omega_rl", "omega_rr", "z", "roll", "pitch", "v_z",
              "roll_rate", "pitch_rate"]


def collect(args):
    env, rng = OffroadRoverCoreEnv(seed=args.seed), np.random.default_rng(args.seed)
    state_rows, action_rows, next_rows, ids = [], [], [], []
    for episode in range(args.episodes):
        state = env.reset((0, 0, rng.uniform(-np.pi, np.pi)))
        command = np.zeros(2)
        for frame in range(args.steps):
            if frame % args.hold_steps == 0:
                command = np.asarray((rng.uniform(.25, 1), rng.uniform(-1, 1)))
            action = np.clip(command + rng.normal(0, args.exploration, 2), -1, 1).astype(np.float32)
            next_state = env.step(action)
            state_rows.append(state); action_rows.append(action); next_rows.append(next_state); ids.append(episode)
            state = next_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{"observation.state": np.stack(state_rows),
        "action": np.stack(action_rows), "next_observation.state": np.stack(next_rows),
        "episode_index": np.asarray(ids)})
    meta = {"episodes": args.episodes, "steps": args.steps, "state_dim": 16,
            "conditioning_context": ["x", "y"] + CORE_ORDER,
            "learned_core": CORE_ORDER, "core_dim": 14, "seed": args.seed}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    return meta


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("data/offroad_flowsde/transitions_core.npz"))
    p.add_argument("--episodes",type=int,default=500);p.add_argument("--steps",type=int,default=300)
    p.add_argument("--hold-steps",type=int,default=8);p.add_argument("--exploration",type=float,default=.08)
    p.add_argument("--seed",type=int,default=0);return p
if __name__=="__main__":print(json.dumps(collect(parser().parse_args()),indent=2))
