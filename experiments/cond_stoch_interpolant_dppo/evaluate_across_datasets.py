"""Manifest-driven evaluation across tasks, methods, seeds, and checkpoints.

The same rollout function is used for every policy.  A JSON manifest supplies
environment registration, normalization, success semantics, checkpoint globs,
and evaluation reset seeds, making new datasets configuration-only additions.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import os
import pickle
import struct
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .train_jax import Normalizer, create_policy


@dataclass(frozen=True)
class Checkpoint:
    task: str
    method: str
    path: Path
    seed: int
    stage: str
    iteration: int
    environment_steps: int


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values) -> tuple[float, float]:
    values = np.asarray([value for value in values if value != ""], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return (
        (float(np.mean(values)), float(np.std(values, ddof=1)))
        if len(values) > 1
        else (float(values[0]), 0.0)
    )


def discover_checkpoints(
    root: Path, task_name: str, task: dict, method_name: str, method: dict
) -> list[Checkpoint]:
    found = []
    for pattern in method["checkpoint_globs"]:
        for path in sorted(root.glob(pattern.format(task=task_name, method=method_name))):
            if not (path / "policy.eqx").exists() or not (path / "config.json").exists():
                continue
            text = str(path)
            seed_match = re.search(method.get("seed_regex", r"seed(\d+)"), text)
            if seed_match is None:
                raise ValueError(f"cannot infer seed from {path}")
            config = json.loads((path / "config.json").read_text())
            stage = method.get("stage")
            if stage is None:
                stage = "pretrain" if "pretrain" in path.parts else "finetune"
            iteration_matches = re.findall(
                method.get("iteration_regex", r"(?:itr|iteration|checkpoint)[_-]?(\d+)"),
                text,
            )
            iteration = (
                int(iteration_matches[-1])
                if iteration_matches else int(config.get("iteration", 0 if stage == "pretrain"
                                                           else config.get("iterations", 0)))
            )
            environment_steps = int(
                config.get(
                    "environment_steps",
                    iteration * int(config.get("n_envs", 0))
                    * int(config.get("rollout_steps", 0))
                    * int(config.get("execution_horizon", config.get("horizon", 0))),
                )
            )
            found.append(Checkpoint(
                task_name, method_name, path, int(seed_match.group(1)), stage,
                iteration, environment_steps,
            ))
    return found


def load_checkpoint(checkpoint: Checkpoint, method: dict | None = None):
    config = json.loads((checkpoint.path / "config.json").read_text())
    config.setdefault("policy", "interpolant")
    config.setdefault("denoising_steps", 20)
    config.setdefault("integration_steps", 50)
    if method and config.get("policy") != "dppo" and "inference_steps" in method:
        config["integration_steps"] = int(method["inference_steps"])
    if method and config.get("policy") != "dppo" and "sde_eta" in method:
        config["sde_eta"] = float(method["sde_eta"])
    args = SimpleNamespace(**config)
    policy = create_policy(args, jr.PRNGKey(checkpoint.seed))
    policy = eqx.tree_deserialise_leaves(checkpoint.path / "policy.eqx", policy)
    base_policy = None
    if (checkpoint.path / "base_policy.eqx").exists():
        base_policy = create_policy(args, jr.PRNGKey(checkpoint.seed))
        base_policy = eqx.tree_deserialise_leaves(
            checkpoint.path / "base_policy.eqx", base_policy
        )
    return policy, base_policy, config


def make_environment(task: dict, record: bool):
    if task.get("environment_backend") == "legacy_d4rl_kitchen":
        if record:
            raise ValueError("legacy D4RL worker does not support video recording")

        class LegacyD4RLKitchenEnvironment:
            """Proxy the exact Python 3.10 D4RL simulator to this JAX process."""

            def __init__(self):
                root = Path(task.get("legacy_root", ".")).resolve()
                python = Path(task["legacy_python"])
                if not python.is_absolute():
                    python = root / python
                worker = Path(__file__).with_name("legacy_d4rl_kitchen_worker.py")
                environment = os.environ.copy()
                environment.update({
                    "D4RL_DATASET_DIR": str(root / "data" / "d4rl_cache"),
                    "D4RL_SUPPRESS_IMPORT_ERROR": "1",
                    "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
                    "PYTHONPATH": str(root / "third_party" / "d4rl_legacy"),
                })
                self.process = subprocess.Popen(
                    [str(python), str(worker)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, env=environment,
                )

            def _request(self, command, payload=None):
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"legacy D4RL worker exited with {self.process.returncode}"
                    )
                message = pickle.dumps((command, payload), protocol=pickle.HIGHEST_PROTOCOL)
                self.process.stdin.write(struct.pack("!Q", len(message)))
                self.process.stdin.write(message)
                self.process.stdin.flush()
                size_bytes = self.process.stdout.read(8)
                if len(size_bytes) != 8:
                    raise RuntimeError("legacy D4RL worker closed its output pipe")
                size = struct.unpack("!Q", size_bytes)[0]
                response = self.process.stdout.read(size)
                if len(response) != size:
                    raise RuntimeError("truncated legacy D4RL worker response")
                return pickle.loads(response)

            def reset(self, seed=None, **_kwargs):
                return self._request("reset", 0 if seed is None else seed)

            def step(self, action):
                # A plain list avoids NumPy 1.x/2.x pickle module-name skew
                # between the legacy simulator and current inference process.
                return self._request(
                    "step", np.asarray(action, dtype=np.float32).tolist()
                )

            def render(self):
                return None

            def close(self):
                if self.process.poll() is None:
                    try:
                        self._request("close")
                    finally:
                        self.process.wait(timeout=10)

        return LegacyD4RLKitchenEnvironment()

    import gymnasium as gym
    for module_name in task.get("registration_modules", []):
        module = importlib.import_module(module_name)
        if hasattr(gym, "register_envs"):
            gym.register_envs(module)
    kwargs = dict(task.get("env_kwargs", {}))
    if record:
        kwargs["render_mode"] = "rgb_array"
    env = gym.make(task["env_id"], **kwargs)
    if task.get("observation_adapter") == "d4rl_kitchen_v0":
        class D4RLKitchenObservation(gym.Wrapper):
            """Expose the legacy D4RL 60-D qpos+goal Kitchen observation."""

            indices = {
                "bottom burner": np.array([11, 12]),
                "top burner": np.array([15, 16]),
                "light switch": np.array([17, 18]),
                "slide cabinet": np.array([19]),
                "hinge cabinet": np.array([20, 21]),
                "microwave": np.array([22]),
                "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
            }

            def __init__(self, wrapped):
                super().__init__(wrapped)
                self.observation_space = gym.spaces.Box(
                    -np.inf, np.inf, shape=(60,), dtype=np.float64
                )

            def _legacy_observation(self):
                base = self.env.unwrapped
                goal = np.zeros(30, dtype=np.float64)
                for name, value in base.goal.items():
                    goal[self.indices[name]] = value
                return np.concatenate((base.data.qpos[:30].copy(), goal))

            @staticmethod
            def _legacy_info(info):
                completed = set(info.get("episode_task_completions", ()))
                info = dict(info)
                info["completed_tasks"] = {
                    "microwave": "microwave" in completed,
                    "kettle": "kettle" in completed,
                    "light_switch": "light switch" in completed,
                    "slide_cabinet": "slide cabinet" in completed,
                }
                info["success"] = len(completed) == 4
                return info

            def reset(self, **reset_kwargs):
                _, info = self.env.reset(**reset_kwargs)
                return self._legacy_observation(), self._legacy_info(info)

            def step(self, action):
                _, reward, terminated, truncated, info = self.env.step(action)
                return (
                    self._legacy_observation(), reward, terminated, truncated,
                    self._legacy_info(info),
                )

        env = D4RLKitchenObservation(env)
    return env


def info_value(info: dict, dotted_key: str, default=False):
    value = info
    for component in dotted_key.split("."):
        if not isinstance(value, dict) or component not in value:
            return default
        value = value[component]
    return value


def evaluate_checkpoint(
    checkpoint: Checkpoint,
    task: dict,
    method: dict,
    output: Path,
    record_episodes: int,
) -> tuple[list[dict], dict]:
    policy, base_policy, config = load_checkpoint(checkpoint, method)
    normalizer = Normalizer(Path(task["normalization"]))
    deterministic = bool(
        method.get("deterministic", task.get("deterministic", True))
    )
    reset_seeds = task["reset_seeds"]
    rows, call_latencies, action_latencies = [], [], []
    execution_horizon = int(
        method.get(
            "execution_horizon",
            task.get("execution_horizon", config.get("execution_horizon", config["horizon"])),
        )
    )
    if not 1 <= execution_horizon <= int(config["horizon"]):
        raise ValueError(
            f"execution_horizon must be in [1, {config['horizon']}], got "
            f"{execution_horizon}"
        )

    # Compile/warm up outside all timing measurements.
    env = make_environment(task, record_episodes > 0)
    inference_steps = int(method.get("inference_steps", policy.nfe))
    sampler = method.get("sampler")
    if sampler == "sde":
        deterministic = False
    if sampler == "ddpm_bc":
        if config.get("policy") != "dppo":
            raise ValueError("DDPM BC sampler requires a diffusion checkpoint")
        # Sampler-controlled comparison: use the same ancestral DDPM kernel
        # for pretrained and fine-tuned parameters. In particular, the final
        # action-producing transition is noise-free, as in BC evaluation.
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_bc(states_, key_)
        )
    elif sampler == "ddim":
        if config.get("policy") != "dppo":
            raise ValueError("DDIM sampler requires a diffusion checkpoint")
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_ddim(
                states_, key_, inference_steps=inference_steps
            )
        )
    elif sampler == "ode":
        if config.get("policy") == "dppo":
            raise ValueError("ODE sampler requires an interpolant checkpoint")
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_ode(states_, key_)
        )
    elif sampler == "ode_heun":
        if config.get("policy") == "dppo":
            raise ValueError("ODE sampler requires an interpolant checkpoint")
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_ode_heun(states_, key_)
        )
    elif sampler == "sde":
        if config.get("policy") == "dppo":
            raise ValueError("SDE sampler requires an interpolant checkpoint")
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_sde(states_, key_)
        )
    elif sampler == "sde_heun":
        if config.get("policy") == "dppo":
            raise ValueError("SDE sampler requires an interpolant checkpoint")
        deterministic = False
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_sde_heun(states_, key_)
        )
    elif checkpoint.stage == "pretrain" and config.get("policy") == "dppo":
        # Match the official pretrained DDPM sampler. DPPO's 0.1 transition
        # floor belongs only to online fine-tuning and corrupts BC evaluation.
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_bc(
                states_, key_
            )
        )
    elif base_policy is not None and config.get("policy") == "dppo":
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample_hybrid(
                states_, key_, base_policy,
                int(config.get("ft_denoising_steps", 10)),
                deterministic=deterministic,
            )
        )
    else:
        sample_policy = eqx.filter_jit(
            lambda policy_, states_, key_: policy_.sample(
                states_, key_, deterministic=deterministic
            )
        )
    warm_observation, _ = env.reset(seed=int(reset_seeds[0]))
    warm_history = np.repeat(
        normalizer.observation(warm_observation)[None], config["cond_steps"], axis=0
    )
    warm = sample_policy(
        policy, jnp.asarray(warm_history[None]), jr.PRNGKey(0)
    )
    jax.block_until_ready(warm.actions)
    evaluation_start = time.perf_counter()

    for episode_index, reset_seed in enumerate(reset_seeds):
        observation, _ = env.reset(seed=int(reset_seed))
        history = [normalizer.observation(observation)] * config["cond_steps"]
        key = jr.PRNGKey(int(reset_seed) + 1_000_000 * checkpoint.seed)
        episode_return, success, length = 0.0, False, 0
        squared_jerk_sum = 0.0
        jerk_count = 0
        previous_action = None
        previous_delta = None
        subtask_success = {name: False for name in task.get("subtasks", {})}
        frames = []
        if episode_index < record_episodes:
            frame = env.render()
            if frame is not None: frames.append(frame)
        done = False
        while not done:
            state = jnp.asarray(np.stack(history)[None])
            key, sample_key = jr.split(key)
            call_start = time.perf_counter()
            sample = sample_policy(policy, state, sample_key)
            jax.block_until_ready(sample.actions)
            call_ms = 1e3 * (time.perf_counter() - call_start)
            call_latencies.append(call_ms)
            executed = 0
            for action_index in range(execution_horizon):
                action = normalizer.action(np.asarray(sample.actions[0, action_index]))
                observation, reward, terminated, truncated, info = env.step(action)
                executed += 1
                history = history[1:] + [normalizer.observation(observation)]
                episode_return += float(reward)
                length += 1
                if previous_action is not None:
                    delta = action - previous_action
                    if previous_delta is not None:
                        squared_jerk_sum += float(np.mean((delta - previous_delta) ** 2))
                        jerk_count += 1
                    previous_delta = delta
                previous_action = action.copy()
                success_value = info_value(
                    info, task.get("success_key", "success"), None
                )
                if success_value is not None:
                    success |= bool(success_value)
                for name, key_name in task.get("subtasks", {}).items():
                    subtask_success[name] |= bool(info_value(info, key_name, False))
                if episode_index < record_episodes:
                    frame = env.render()
                    if frame is not None: frames.append(frame)
                done = bool(terminated or truncated)
                if done: break
            action_latencies.extend([call_ms / executed] * executed)
        if not success and "success_return_threshold" in task:
            success = episode_return >= float(task["success_return_threshold"])
        row = {
            "task": checkpoint.task, "method": checkpoint.method,
            "seed": checkpoint.seed, "stage": checkpoint.stage,
            "iteration": checkpoint.iteration,
            "environment_steps": checkpoint.environment_steps,
            "checkpoint": str(checkpoint.path), "episode": episode_index,
            "reset_seed": reset_seed, "return": episode_return,
            "success": int(success), "length": length,
            "action_jerk": (
                squared_jerk_sum / jerk_count if jerk_count else 0.0
            ),
            # Compatibility alias for existing downstream CSV consumers.
            "action_smoothness": (
                squared_jerk_sum / jerk_count if jerk_count else 0.0
            ),
            "execution_horizon": execution_horizon, "nfe": inference_steps,
            "sampler": sampler or "default", "deterministic": deterministic,
        }
        row.update({f"subtask_{name}": int(value)
                    for name, value in subtask_success.items()})
        rows.append(row)
        if frames:
            import imageio.v2 as imageio

            video_dir = output / "recordings" / checkpoint.task / checkpoint.method
            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                video_dir / (
                    f"seed{checkpoint.seed}_{checkpoint.stage}_"
                    f"itr{checkpoint.iteration}_episode{episode_index}.gif"
                ),
                frames,
                fps=int(task.get("video_fps", 20)),
            )
    env.close()
    wall_seconds = time.perf_counter() - evaluation_start
    summary = summarize_checkpoint(rows)
    summary.update({
        "latency_call_ms_mean": mean_std(call_latencies)[0],
        "latency_call_ms_std": mean_std(call_latencies)[1],
        "latency_action_ms_mean": mean_std(action_latencies)[0],
        "latency_action_ms_std": mean_std(action_latencies)[1],
        "evaluation_wall_seconds": wall_seconds,
    })
    return rows, summary


def summarize_checkpoint(rows: list[dict]) -> dict:
    first = rows[0]
    summary = {
        key: first[key] for key in (
            "task", "method", "seed", "stage", "iteration",
            "environment_steps", "checkpoint", "nfe",
        )
    }
    for source, target in (
        ("return", "return"), ("success", "success_rate"),
        ("action_jerk", "action_jerk"),
    ):
        summary[f"{target}_mean"], summary[f"{target}_std"] = mean_std(
            [row[source] for row in rows]
        )
    for key in rows[0]:
        if key.startswith("subtask_"):
            summary[f"{key}_rate"] = float(np.mean([row[key] for row in rows]))
    summary["episodes"] = len(rows)
    successful_lengths = [row["length"] for row in rows if row["success"]]
    if successful_lengths:
        summary["successful_length_mean"], summary["successful_length_std"] = (
            mean_std(successful_lengths)
        )
    else:
        summary["successful_length_mean"] = float("nan")
        summary["successful_length_std"] = float("nan")
    return summary


def aggregate_final(
    summaries: list[dict], expected_seeds: int, final_window: int
) -> tuple[list[dict], list[dict]]:
    final_rows, flags = [], []
    groups = sorted(set((row["task"], row["method"], row["stage"]) for row in summaries))
    for task, method, stage in groups:
        selected = [row for row in summaries if (
            row["task"], row["method"], row["stage"]) == (task, method, stage)]
        seeds = sorted(set(row["seed"] for row in selected))
        if len(seeds) < expected_seeds:
            flags.append({
                "task": task, "method": method, "stage": stage,
                "issue": "insufficient_seeds", "observed": len(seeds),
                "required": expected_seeds,
            })
        per_seed = []
        for seed in seeds:
            candidates = sorted(
                [row for row in selected if row["seed"] == seed],
                key=lambda row: row["iteration"],
            )
            candidates = candidates[-final_window:] if stage == "finetune" else candidates[-1:]
            per_seed.append({
                metric: float(np.mean([row[metric] for row in candidates]))
                for metric in (
                    "success_rate_mean", "return_mean", "action_jerk_mean",
                    "successful_length_mean", "latency_call_ms_mean",
                    "latency_action_ms_mean", "evaluation_wall_seconds", "nfe",
                )
            })
        result = {"task": task, "method": method, "stage": stage, "seeds": len(seeds)}
        for metric in per_seed[0]:
            result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(
                [row[metric] for row in per_seed]
            )
        final_rows.append(result)
    return final_rows, flags


def plot_learning_curves(summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    for task in sorted(set(row["task"] for row in summaries)):
        task_rows = [row for row in summaries if row["task"] == task]
        if not any(row["stage"] == "finetune" for row in task_rows):
            continue
        for metric, label, suffix in (
            ("success_rate_mean", "Success rate", "success"),
            ("return_mean", "Episode return", "return"),
        ):
            figure, axis = plt.subplots(figsize=(6, 4))
            for method in sorted(set(row["method"] for row in task_rows)):
                method_rows = [row for row in task_rows if row["method"] == method]
                finetune_rows = [row for row in method_rows
                                 if row["stage"] == "finetune"]
                if not finetune_rows:
                    continue
                x_key = (
                    "environment_steps"
                    if any(row["environment_steps"] for row in finetune_rows)
                    else "iteration"
                )
                x_values = sorted(set(row[x_key] for row in finetune_rows))
                means, stds = [], []
                for x_value in x_values:
                    values = [row[metric] for row in finetune_rows
                              if row[x_key] == x_value]
                    mean, std = mean_std(values)
                    means.append(mean); stds.append(std)
                pretrain_values = [row[metric] for row in method_rows
                                   if row["stage"] == "pretrain"]
                if pretrain_values:
                    pretrain_mean, pretrain_std = mean_std(pretrain_values)
                    x_values.insert(0, 0)
                    means.insert(0, pretrain_mean)
                    stds.insert(0, pretrain_std)
                x, y, band = map(np.asarray, (x_values, means, stds))
                axis.plot(x, y, marker="o", label=method)
                axis.fill_between(x, y - band, y + band, alpha=.2)
            axis.set(
                xlabel=("Environment steps" if x_key == "environment_steps"
                        else "Fine-tuning iteration"),
                ylabel=label, title=task,
            )
            if metric == "success_rate_mean":
                axis.set_ylim(-.02, 1.02)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{task}_{suffix}_curve.png", dpi=200)
            plt.close(figure)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Task | Method | Stage | Seeds | Success | Return | NFE | "
        "Steps to success | Latency/chunk ms | Action jerk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def formatted(row, metric, digits=3):
        return f"{row[metric + '_mean']:.{digits}f} ± {row[metric + '_std']:.{digits}f}"
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['stage']} | {row['seeds']} | "
            f"{formatted(row, 'success_rate_mean')} | {formatted(row, 'return_mean')} | "
            f"{formatted(row, 'nfe', 1)} | "
            f"{formatted(row, 'successful_length_mean')} | "
            f"{formatted(row, 'latency_call_ms_mean')} | "
            f"{formatted(row, 'action_jerk_mean')} |"
        )
    return "\n".join(lines) + "\n"


def main(
    manifest_path: Path,
    output_override: Path | None,
    record_episodes: int,
    episode_limit: int | None = None,
):
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest.get("root", "."))
    output = output_override or Path(manifest["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    discovery_flags = []
    for task_name, task in manifest["tasks"].items():
        for method_name, method in manifest["methods"].items():
            discovered = discover_checkpoints(
                root, task_name, task, method_name, method
            )
            checkpoints.extend(discovered)
            if not discovered:
                discovery_flags.append({
                    "task": task_name, "method": method_name, "stage": "all",
                    "issue": "no_checkpoints", "observed": 0, "required": 1,
                })
                continue
            stages = {checkpoint.stage for checkpoint in discovered}
            for required_stage in ("pretrain", "finetune"):
                if required_stage not in stages:
                    discovery_flags.append({
                        "task": task_name, "method": method_name,
                        "stage": required_stage,
                        "issue": "missing_stage", "observed": 0, "required": 1,
                    })
    if not checkpoints:
        raise RuntimeError("manifest did not discover any checkpoints")
    episode_rows, checkpoint_rows = [], []
    for checkpoint in checkpoints:
        task = manifest["tasks"][checkpoint.task]
        if episode_limit is not None:
            if episode_limit < 1:
                raise ValueError("--episodes must be positive")
            task = dict(task)
            task["reset_seeds"] = task["reset_seeds"][:episode_limit]
        method = manifest["methods"][checkpoint.method]
        print(f"evaluating {checkpoint.task}/{checkpoint.method}/seed{checkpoint.seed}/"
              f"{checkpoint.stage}/itr{checkpoint.iteration}", flush=True)
        episodes, summary = evaluate_checkpoint(
            checkpoint, task, method, output, record_episodes
        )
        episode_rows.extend(episodes); checkpoint_rows.append(summary)
        write_csv(output / "episodes.csv", episode_rows)
        write_csv(output / "checkpoints.csv", checkpoint_rows)
    final_rows, flags = aggregate_final(
        checkpoint_rows, int(manifest.get("expected_seeds", 3)),
        int(manifest.get("final_checkpoint_window", 5)),
    )
    flags = discovery_flags + flags
    write_csv(output / "final_results.csv", final_rows)
    write_csv(output / "flags.csv", flags)
    (output / "results.md").write_text(markdown_table(final_rows))
    plot_learning_curves(checkpoint_rows, output)
    print(markdown_table(final_rows))
    if flags:
        print("FLAGS:", json.dumps(flags, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-episodes", type=int, default=0)
    parser.add_argument(
        "--episodes", type=int,
        help="Evaluate only the first N configured reset seeds per checkpoint.",
    )
    arguments = parser.parse_args()
    main(
        arguments.manifest, arguments.output, arguments.record_episodes,
        arguments.episodes,
    )
