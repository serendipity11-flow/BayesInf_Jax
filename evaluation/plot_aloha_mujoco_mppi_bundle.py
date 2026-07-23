#!/usr/bin/env python3
"""Visualize ALOHA MuJoCo MPPI comparison results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _maybe_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def build_bundle(run_dir: Path, metrics_only: bool = False) -> Path:
    comp_dir = run_dir / "mppi_comparison"
    if not comp_dir.exists():
        comp_dir = run_dir / "all_episode_mppi"
    summary_path = comp_dir / "summary.csv"
    case_path = comp_dir / "case_metrics.csv"
    metrics_out = run_dir / "mppi_comparison_metrics.png"
    trajectories_out = run_dir / "mppi_comparison_trajectories.png"

    recovered_summary = run_dir / "mppi_comparison_summary.csv"
    if not summary_path.exists() and recovered_summary.exists() and metrics_only:
        summary_path = recovered_summary
    if not summary_path.exists() or (not case_path.exists() and not metrics_only):
        raise FileNotFoundError(f"Missing summary or case metrics in {comp_dir}")

    summary = pd.read_csv(summary_path)
    case = pd.read_csv(case_path) if case_path.exists() else None
    tasks = sorted(case["task"].unique()) if case is not None else []

    # Metrics figure.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    summary_models = summary["model"].tolist()
    metrics = [
        ("final_normalized_rmse", "Final normalized RMSE"),
        ("cumulative_normalized_error", "Cumulative normalized error"),
        ("control_smoothness", "Control smoothness (RMS Δu)"),
        ("mean_runtime_ms", "Mean inference time [ms]"),
    ]
    colors = ["tab:blue" if model == "exact_mujoco" else "tab:orange"
              for model in summary_models]
    x = np.arange(len(summary_models))
    for ax, (metric, title) in zip(axes.flat, metrics):
        std_column = f"{metric}_std"
        error = summary[std_column].to_numpy() if std_column in summary else None
        ax.bar(x, summary[metric].to_numpy(), yerr=error, capsize=4,
               color=colors, alpha=.85)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_models)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Aggregate MPPI comparison (independent metric scales)")
    fig.savefig(metrics_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if metrics_only:
        return metrics_out

    # Trajectory figure.
    fig = plt.figure(figsize=(18, 4 + 3.0 * len(tasks)), constrained_layout=True)
    gs = fig.add_gridspec(len(tasks), 2)
    for row, task in enumerate(tasks, start=1):
        ax = fig.add_subplot(gs[row - 1, 0])
        ax2 = fig.add_subplot(gs[row - 1, 1])

        for model, color in [("exact_mujoco", "tab:blue"), ("flow_sde", "tab:orange")]:
            npz_path = comp_dir / f"task{task}_{model}.npz"
            if not npz_path.exists():
                continue
            data = _load_npz(npz_path)
            states = np.asarray(data["states"])
            actions = np.asarray(data["actions"])
            goal = _maybe_1d(data["goal"])
            t = np.arange(states.shape[0])
            # Plot two representative state dims and action norm.
            ax.plot(t, states[:, 0], color=color, lw=1.6, label=f"{model} q0")
            ax.plot(t, states[:, 1], color=color, lw=1.0, ls="--", label=f"{model} q1")
            ax.plot([t[0], t[-1]], [goal[0], goal[0]], color=color, alpha=0.25, lw=1.0)
            ax.plot([t[0], t[-1]], [goal[1], goal[1]], color=color, alpha=0.25, lw=1.0, ls="--")
            ax2.plot(np.arange(actions.shape[0]), np.linalg.norm(actions, axis=1), color=color, lw=1.6, label=f"{model} ||u||")

        ax.set_title(f"Task {task}: representative state coordinates")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        if row == 1:
            ax.legend(frameon=False, fontsize=8, ncol=2)

        ax2.set_title(f"Task {task}: action magnitude")
        ax2.set_xlabel("step")
        ax2.grid(True, alpha=0.25)
        if row == 1:
            ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{run_dir.name} MPPI trajectories", fontsize=14)
    fig.savefig(trajectories_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Run directory containing mppi_comparison/")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only regenerate the aggregate metrics figure.")
    args = parser.parse_args()
    print(build_bundle(args.run_dir, metrics_only=args.metrics_only))


if __name__ == "__main__":
    main()
