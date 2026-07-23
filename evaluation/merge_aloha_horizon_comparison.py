#!/usr/bin/env python3
"""Merge FlowSDE and exact-MuJoCo MPPI horizon sweeps into canonical plots."""

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
    exact_path = args.root / "exact_mujoco_horizon_sweep" / "case_metrics.csv"
    flow_path = args.root / "risk1_particles8_horizon_sweep" / "case_metrics.csv"
    exact, flow = pd.read_csv(exact_path), pd.read_csv(flow_path)
    expected = 6 * 80
    if len(exact) != expected or len(flow) != expected:
        raise ValueError(f"expected {expected} cases/controller, got exact={len(exact)}, flow={len(flow)}")
    keys = ["horizon", "task", "episode_id"]
    if not exact[keys].sort_values(keys).reset_index(drop=True).equals(
            flow[keys].sort_values(keys).reset_index(drop=True)):
        raise ValueError("controller sweeps do not contain matching horizon/episode cases")
    cases = pd.concat([exact, flow], ignore_index=True).sort_values(keys + ["model"])
    cases.to_csv(args.root / "sweep_horizon_per_episode.csv", index=False)

    summary = cases.groupby(["horizon", "model"], as_index=False).agg(
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
    summary.insert(1, "flow_risk_weight", np.where(summary.model == "flow_sde", 1.0, np.nan))
    summary.insert(2, "flow_particles", np.where(summary.model == "flow_sde", 8, np.nan))
    summary.to_csv(args.root / "sweep_horizon_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for model, label, color in CONTROLLERS:
            selected = summary[summary.model == model].sort_values("horizon")
            ci = 1.96 * selected[f"{metric}_std"] / np.sqrt(selected.episodes)
            ax.errorbar(selected.horizon, selected[metric], yerr=ci, marker="o",
                        capsize=3, lw=1.8, color=color, label=label)
        ax.set(xlabel="MPPI horizon", ylabel=ylabel, xticks=[2, 4, 6, 8, 10, 12])
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("MPPI horizon comparison across 80 episodes (mean ± 95% CI)")
    fig.savefig(args.root / "horizon_sweep_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model, label, color in CONTROLLERS:
        selected = summary[summary.model == model].sort_values("horizon")
        ci = 1.96 * selected.mean_runtime_ms_std / np.sqrt(selected.episodes)
        ax.errorbar(selected.horizon, selected.mean_runtime_ms, yerr=ci, marker="o",
                    capsize=3, lw=1.8, color=color, label=label)
    ax.set(xlabel="MPPI horizon", ylabel="Mean inference time [ms]",
           xticks=[2, 4, 6, 8, 10, 12], title="MPPI inference time across 80 episodes")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(args.root / "horizon_sweep_runtime.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
