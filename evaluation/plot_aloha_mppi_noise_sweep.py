#!/usr/bin/env python3
"""Aggregate FlowSDE and exact-MuJoCo MPPI plant-noise sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROLLERS = (
    ("exact_mujoco", "Exact MuJoCo MPPI", "tab:blue"),
    ("flow_sde", "FlowSDE MPPI (risk=1, particles=8)", "tab:orange"),
)
METRICS = (
    ("final_normalized_rmse", "Final normalized RMSE"),
    ("cumulative_normalized_error", "Cumulative normalized error"),
    ("control_smoothness", "Control smoothness (RMS Δu)"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("noise*/case_metrics.csv")):
        data = pd.read_csv(path)
        if len(data) != 160 or set(data.model) != {"exact_mujoco", "flow_sde"}:
            raise ValueError(f"incomplete noise case {path}: rows={len(data)}")
        rows.append(data)
    if not rows:
        raise FileNotFoundError(f"no completed noise cases under {args.root}")
    cases = pd.concat(rows, ignore_index=True).sort_values(
        ["plant_noise_std", "task", "model"]
    )
    cases.to_csv(args.root / "noise_sweep_per_episode.csv", index=False)
    summary = cases.groupby(["plant_noise_std", "model"], as_index=False).agg(
        episodes=("episode_id", "size"),
        final_normalized_rmse=("final_normalized_rmse", "mean"),
        final_normalized_rmse_std=("final_normalized_rmse", lambda x: np.std(x, ddof=0)),
        cumulative_normalized_error=("cumulative_normalized_error", "mean"),
        cumulative_normalized_error_std=("cumulative_normalized_error", lambda x: np.std(x, ddof=0)),
        control_smoothness=("control_smoothness", "mean"),
        control_smoothness_std=("control_smoothness", lambda x: np.std(x, ddof=0)),
        mean_runtime_ms=("mean_runtime_ms", "mean"),
        mean_runtime_ms_std=("mean_runtime_ms", lambda x: np.std(x, ddof=0)),
    )
    summary.to_csv(args.root / "noise_sweep_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for model, label, color in CONTROLLERS:
            selected = summary[summary.model == model].sort_values("plant_noise_std")
            ci = 1.96 * selected[f"{metric}_std"] / np.sqrt(selected.episodes)
            ax.errorbar(selected.plant_noise_std, selected[metric], yerr=ci,
                        marker="o", capsize=3, color=color, lw=1.8, label=label)
        ax.set(xlabel="Plant noise standard deviation", ylabel=ylabel)
        ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("MPPI plant-noise sweep across 80 episodes (mean ± 95% CI)")
    fig.savefig(args.root / "noise_sweep_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model, label, color in CONTROLLERS:
        selected = summary[summary.model == model].sort_values("plant_noise_std")
        ci = 1.96 * selected.mean_runtime_ms_std / np.sqrt(selected.episodes)
        ax.errorbar(selected.plant_noise_std, selected.mean_runtime_ms, yerr=ci,
                    marker="o", capsize=3, color=color, lw=1.8, label=label)
    ax.set(xlabel="Plant noise standard deviation", ylabel="Mean inference time [ms]",
           title="MPPI runtime during plant-noise sweep")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(args.root / "noise_sweep_runtime.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
