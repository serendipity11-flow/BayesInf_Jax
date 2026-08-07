"""Preflight compatibility check for offline Door data and the online environment.

This intentionally checks contracts, not just array dimensions.  A dataset and
an environment with equal observation/action shapes can still be incompatible
because of different episode limits, action bounds, or reward implementations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _check(name: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def check_prepared_dataset(
    root: Path, env, reset_seeds=(0, 1, 2, 3), robotics_version: str | None = None
) -> dict:
    """Compare a prepared NPZ dataset with an instantiated Gymnasium env."""
    normalization = np.load(root / "normalization.npz", allow_pickle=False)
    splits = [
        np.load(root / f"{split}.npz", allow_pickle=False)
        for split in ("train", "val")
    ]
    lengths = np.concatenate([split["traj_lengths"] for split in splits])
    rewards = np.concatenate([split["rewards"] for split in splits])
    observations = np.concatenate([split["states"] for split in splits])
    actions = np.concatenate([split["actions"] for split in splits])

    env_obs_shape = tuple(env.observation_space.shape)
    env_action_shape = tuple(env.action_space.shape)
    episode_limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    checks = [
        _check("observation_shape", observations.shape[1:] == env_obs_shape,
               list(env_obs_shape), list(observations.shape[1:])),
        _check("action_shape", actions.shape[1:] == env_action_shape,
               list(env_action_shape), list(actions.shape[1:])),
        _check("finite_dataset_rewards", bool(np.isfinite(rewards).all()),
               bool(np.isfinite(rewards).all()), True),
        _check("action_space_low",
               np.all(normalization["action_min"] >= env.action_space.low - 1e-6),
               normalization["action_min"].tolist(), env.action_space.low.tolist()),
        _check("action_space_high",
               np.all(normalization["action_max"] <= env.action_space.high + 1e-6),
               normalization["action_max"].tolist(), env.action_space.high.tolist()),
    ]
    if robotics_version is not None:
        environment_id = getattr(getattr(env, "spec", None), "id", None)
        # Upstream v2 currently precedes the 1.4.3 wheel and still reports
        # 1.4.2 internally. The registered v2 ID is the authoritative check.
        checks.insert(0, _check(
            "gymnasium_robotics_version",
            robotics_version in {"1.4.2", "1.4.3"},
            robotics_version,
            "pinned upstream v2 revision or released 1.4.3",
        ))
        checks.insert(1, _check(
            "environment_version",
            environment_id == "AdroitHandDoor-v2",
            environment_id,
            "AdroitHandDoor-v2",
        ))
    if episode_limit is not None:
        checks.append(_check(
            "episode_horizon",
            bool(np.all(lengths <= episode_limit)),
            {"minimum": int(lengths.min()), "maximum": int(lengths.max())},
            {"maximum": int(episode_limit)},
        ))

    # Initial online states should lie in the support used to normalize the
    # offline observations. Large z scores are a useful version-mismatch alarm.
    if "obs_mean" in normalization:
        reset_observations = np.stack([
            env.reset(seed=int(seed))[0] for seed in reset_seeds
        ])
        reset_z = np.abs(
            (reset_observations - normalization["obs_mean"])
            / normalization["obs_std"]
        )
        checks.append(_check(
            "reset_observation_support",
            bool(np.isfinite(reset_z).all() and np.max(reset_z) <= 10.0),
            {"maximum_abs_z": float(np.max(reset_z)),
             "dimensions_above_10": np.flatnonzero(np.max(reset_z, axis=0) > 10).tolist()},
            {"maximum_abs_z": 10.0},
        ))

    # Verify the runtime success/reward API with an in-bounds action.
    env.reset(seed=int(reset_seeds[0]))
    midpoint = (env.action_space.low + env.action_space.high) / 2
    _, reward, _, _, info = env.step(midpoint)
    checks.extend([
        _check("finite_environment_reward", bool(np.isfinite(reward)),
               float(reward), "finite"),
        _check("success_info_key", "success" in info, sorted(info), ["success"]),
    ])

    return {
        "compatible": all(item["passed"] for item in checks),
        "dataset": str(root),
        "dataset_episode_lengths": {
            "minimum": int(lengths.min()), "maximum": int(lengths.max()),
        },
        "dataset_reward": {
            "minimum": float(rewards.min()), "maximum": float(rewards.max()),
            "mean": float(rewards.mean()),
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/door-expert-v2-100"))
    parser.add_argument("--env", default="AdroitHandDoor-v2")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--allow-incompatible", action="store_true",
                        help="Print failures without returning a failing exit status.")
    args = parser.parse_args()

    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env, max_episode_steps=args.max_episode_steps)
    try:
        report = check_prepared_dataset(
            args.data, env, robotics_version=gymnasium_robotics.__version__
        )
    finally:
        env.close()
    print(json.dumps(report, indent=2))
    if not report["compatible"] and not args.allow_incompatible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
