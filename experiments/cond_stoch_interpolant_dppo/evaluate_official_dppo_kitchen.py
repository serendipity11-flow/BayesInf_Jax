"""Evaluate the official DPPO Kitchen BC checkpoint on both environment backends."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets import (
    info_value,
    make_environment,
)


class OfficialPolicy:
    def __init__(self, python: Path, checkpoint: Path, device: str):
        worker = Path(__file__).with_name("official_dppo_policy_worker.py")
        environment = os.environ.copy()
        root = Path(__file__).resolve().parents[2]
        environment["PYTHONPATH"] = str(root / "third_party" / "dppo")
        # The pinned PyTorch runtime is installed in the system Python's user
        # site; the JAX parent intentionally disables its own user site.
        environment.pop("PYTHONNOUSERSITE", None)
        self.process = subprocess.Popen(
            [str(python), str(worker), "--checkpoint", str(checkpoint),
             "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment,
        )

    def request(self, command, payload=None):
        if self.process.poll() is not None:
            raise RuntimeError(f"official policy worker exited with {self.process.returncode}")
        message = pickle.dumps((command, payload), protocol=pickle.HIGHEST_PROTOCOL)
        self.process.stdin.write(struct.pack("!Q", len(message)))
        self.process.stdin.write(message)
        self.process.stdin.flush()
        size_bytes = self.process.stdout.read(8)
        if len(size_bytes) != 8:
            raise RuntimeError("official policy worker closed its output pipe")
        size = struct.unpack("!Q", size_bytes)[0]
        response = self.process.stdout.read(size)
        if len(response) != size:
            raise RuntimeError("truncated policy response")
        return pickle.loads(response)

    def sample(self, observation, seed):
        return self.request("sample", (np.asarray(observation).tolist(), seed))

    def close(self):
        if self.process.poll() is None:
            try:
                self.request("close")
            finally:
                self.process.wait(timeout=10)


class Normalizer:
    def __init__(self, path: Path):
        values = np.load(path)
        self.obs_min, self.obs_max = values["obs_min"], values["obs_max"]
        self.action_min, self.action_max = values["action_min"], values["action_max"]

    def observation(self, value):
        return 2 * ((value - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - .5)

    def action(self, value):
        return (value + 1) * .5 * (self.action_max - self.action_min) + self.action_min


def task():
    return {
        "env_id": "FrankaKitchen-v1",
        "registration_modules": ["gymnasium_robotics"],
        "env_kwargs": {"max_episode_steps": 280, "tasks_to_complete": [
            "microwave", "kettle", "light switch", "slide cabinet"]},
        "observation_adapter": "d4rl_kitchen_v0",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    normalizer = Normalizer(args.normalization)
    policy = OfficialPolicy(args.policy_python, args.checkpoint.resolve(), args.device)
    rows = []
    try:
        for backend in ("gymnasium",):
            env = make_environment(task(), False)
            try:
                for episode in range(args.episodes):
                    reset_seed = args.seed_start + episode
                    observation, _ = env.reset(seed=reset_seed)
                    total, completed, length, done = 0.0, set(), 0, False
                    calls = 0
                    start = time.perf_counter()
                    while not done:
                        normalized = normalizer.observation(observation)
                        actions = policy.sample(normalized, reset_seed * 1000 + calls)
                        calls += 1
                        for action in actions:
                            observation, reward, terminated, truncated, info = env.step(
                                normalizer.action(action)
                            )
                            total += float(reward)
                            length += 1
                            for name in ("microwave", "kettle", "light_switch", "slide_cabinet"):
                                if info_value(info, f"completed_tasks.{name}", False):
                                    completed.add(name)
                            done = bool(terminated or truncated)
                            if done:
                                break
                    rows.append({
                        "backend": backend, "episode": episode,
                        "reset_seed": reset_seed, "return": total,
                        "success": int(total >= 4 or len(completed) == 4),
                        "completed_subtasks": len(completed), "length": length,
                        "wall_seconds": time.perf_counter() - start,
                    })
            finally:
                env.close()
    finally:
        policy.close()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for backend in ("gymnasium",):
        selected = [row for row in rows if row["backend"] == backend]
        summary[backend] = {
            "episodes": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "return_mean": float(np.mean([row["return"] for row in selected])),
            "return_std": float(np.std([row["return"] for row in selected])),
            "completed_subtasks_mean": float(np.mean([
                row["completed_subtasks"] for row in selected
            ])),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
