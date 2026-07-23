#!/usr/bin/env python3
"""Evaluate and plot every saved ALOHA MPPI episode.

The script consumes the ``h*_task*_{exact_mujoco,flow_sde}.npz`` files written
by ``experiments.aloha_mujoco_flow.compare_mppi``.  It does not rerun MPPI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("exact_mujoco", "flow_sde")
COLORS = {"exact_mujoco": "tab:blue", "flow_sde": "tab:orange"}


def load_runs(run_dir: Path, horizon: int, expected_episodes: int):
    runs = {}
    for task in range(expected_episodes):
        runs[task] = {}
        for model in MODELS:
            path = run_dir / f"h{horizon}_task{task}_{model}.npz"
            if not path.exists():
                raise FileNotFoundError(f"missing episode rollout: {path}")
            with np.load(path) as data:
                runs[task][model] = {name: np.asarray(data[name]) for name in data.files}
        exact, flow = runs[task]["exact_mujoco"], runs[task]["flow_sde"]
        if exact["states"].shape != flow["states"].shape:
            raise ValueError(f"task {task}: model state shapes differ")
        if exact["actions"].shape != flow["actions"].shape:
            raise ValueError(f"task {task}: model action shapes differ")
    return runs


def calculate_metrics(runs, x_std: np.ndarray) -> pd.DataFrame:
    rows = []
    for task, model_runs in runs.items():
        for model, run in model_runs.items():
            states, actions, goal = run["states"], run["actions"], run["goal"]
            error = (states - goal) / x_std
            rows.append({
                "task": task,
                "episode_id": int(run["episode_id"]),
                "model": model,
                "final_normalized_rmse": np.sqrt(np.mean(error[-1] ** 2)),
                "trajectory_normalized_rmse": np.sqrt(np.mean(error ** 2)),
                "cumulative_normalized_error": np.sum(np.mean(error ** 2, axis=1)),
                "control_rms": np.sqrt(np.mean(actions ** 2)),
                "control_smoothness": np.sqrt(np.mean(np.diff(actions, axis=0) ** 2)),
            })
    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(index=["task", "episode_id"], columns="model")
    for metric in ("final_normalized_rmse", "trajectory_normalized_rmse",
                   "cumulative_normalized_error", "control_rms", "control_smoothness"):
        wide[(metric, "flow_minus_exact")] = (
            wide[(metric, "flow_sde")] - wide[(metric, "exact_mujoco")]
        )
    return wide.sort_index(axis=1).reset_index()


def plot_performance(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    specs = [
        ("final_normalized_rmse", "Final normalized RMSE"),
        ("cumulative_normalized_error", "Cumulative normalized squared error"),
        ("control_smoothness", "Control smoothness (RMS Δu)"),
    ]
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, specs):
        values = [metrics[(metric, model)].to_numpy() for model in MODELS]
        violins = ax.violinplot(values, positions=[0, 1], widths=.75,
                                showmeans=False, showmedians=False, showextrema=False)
        for body, model in zip(violins["bodies"], MODELS):
            body.set_facecolor(COLORS[model])
            body.set_edgecolor(COLORS[model])
            body.set_alpha(.22)
        boxes = ax.boxplot(values, positions=[0, 1], widths=.25, patch_artist=True,
                           showfliers=False, medianprops={"color": "black", "lw": 1.5})
        for box, model in zip(boxes["boxes"], MODELS):
            box.set_facecolor(COLORS[model])
            box.set_alpha(.7)
        for position, model, sample in zip([0, 1], MODELS, values):
            jitter = rng.uniform(-.09, .09, len(sample))
            ax.scatter(position + jitter, sample, s=9, alpha=.35,
                       color=COLORS[model], edgecolors="none")
        ax.set_xticks([0, 1], MODELS)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("MPPI performance distributions across 80 episodes")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_episode(task: int, model_runs: dict, output: Path, control_dt: float) -> None:
    exact = model_runs["exact_mujoco"]
    n_state, n_action = exact["states"].shape[1], exact["actions"].shape[1]
    n_cols = 4
    n_rows = int(np.ceil((n_state + n_action) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.35 * n_rows),
                             constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    t_state = np.arange(exact["states"].shape[0]) * control_dt
    t_action = np.arange(exact["actions"].shape[0]) * control_dt
    for dim in range(n_state):
        ax = axes[dim]
        for model in MODELS:
            ax.plot(t_state, model_runs[model]["states"][:, dim], lw=1.25,
                    color=COLORS[model], label=model)
        ax.axhline(exact["goal"][dim], color="black", ls="--", lw=.8, label="goal")
        ax.set_title(f"state x[{dim}]")
        ax.grid(alpha=.2)
    for dim in range(n_action):
        ax = axes[n_state + dim]
        for model in MODELS:
            ax.step(t_action, model_runs[model]["actions"][:, dim], where="post",
                    lw=1.25, color=COLORS[model], label=model)
        ax.set_title(f"input u[{dim}]")
        ax.grid(alpha=.2)
    for ax in axes[n_state + n_action:]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    for ax in axes[-n_cols:]:
        if ax.get_visible():
            ax.set_xlabel("time [s]")
    episode_id = int(exact["episode_id"])
    fig.suptitle(f"MPPI episode {episode_id}: all {n_state} states and {n_action} inputs")
    fig.savefig(output / f"episode_{episode_id:03d}_states_inputs.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--control-dt", type=float, default=.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.run_dir / "all_episode_evaluation"
    episode_dir = output / "states_inputs"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.normalization) as stats:
        x_std = np.maximum(np.asarray(stats["x_std"]), 1e-8)
    runs = load_runs(args.run_dir, args.horizon, args.episodes)
    metrics = calculate_metrics(runs, x_std)
    metrics.to_csv(output / "all_episode_performance.csv", index=False)
    plot_performance(metrics, output / "all_episode_performance.png")
    for task, model_runs in runs.items():
        plot_episode(task, model_runs, episode_dir, args.control_dt)

    print(f"evaluated {len(runs)} episodes")
    print(f"metrics: {output / 'all_episode_performance.csv'}")
    print(f"performance plot: {output / 'all_episode_performance.png'}")
    print(f"state/input plots: {episode_dir}")


if __name__ == "__main__":
    main()
