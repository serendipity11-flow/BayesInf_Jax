#!/usr/bin/env python3
"""Aggregate a FlowSDE-MPPI smoothness-weight sweep with exact-MuJoCo baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    ("final_normalized_rmse", "Final normalized RMSE"),
    ("cumulative_normalized_error", "Cumulative normalized error"),
    ("control_smoothness", "Control smoothness (RMS Δu)"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    exact = pd.read_csv(args.root / "exact_baseline" / "case_metrics.csv")
    if len(exact) != 80 or set(exact.model) != {"exact_mujoco"}:
        raise ValueError("exact-MuJoCo baseline is incomplete")

    flow_rows = []
    for path in sorted(args.root.glob("weight*/case_metrics.csv")):
        tag = path.parent.name.removeprefix("weight").replace("p", ".")
        weight = float(tag)
        data = pd.read_csv(path)
        if len(data) != 80 or set(data.model) != {"flow_sde"}:
            raise ValueError(f"incomplete FlowSDE case {path}")
        data.insert(0, "smoothness_weight", weight)
        flow_rows.append(data)
    if not flow_rows:
        raise FileNotFoundError("no FlowSDE smoothness cases found")
    flow = pd.concat(flow_rows, ignore_index=True).sort_values(["smoothness_weight", "task"])
    exact.insert(0, "smoothness_weight", 0.0)
    pd.concat([exact, flow], ignore_index=True).to_csv(
        args.root / "smoothness_sweep_per_episode.csv", index=False
    )

    summary = flow.groupby("smoothness_weight", as_index=False).agg(
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
    exact_summary = {
        "episodes": len(exact),
        **{metric: exact[metric].mean() for metric, _ in METRICS},
        **{f"{metric}_std": exact[metric].std(ddof=0) for metric, _ in METRICS},
        "mean_runtime_ms": exact.mean_runtime_ms.mean(),
        "mean_runtime_ms_std": exact.mean_runtime_ms.std(ddof=0),
    }
    summary.to_csv(args.root / "smoothness_sweep_summary.csv", index=False)
    pd.DataFrame([exact_summary]).to_csv(args.root / "exact_baseline_summary.csv", index=False)

    labels = [f"{value:g}" for value in summary.smoothness_weight]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        ci = 1.96 * summary[f"{metric}_std"] / np.sqrt(summary.episodes)
        ax.errorbar(x, summary[metric], yerr=ci, marker="o", capsize=3,
                    color="tab:orange", lw=1.8, label="FlowSDE MPPI")
        exact_ci = 1.96 * exact_summary[f"{metric}_std"] / np.sqrt(exact_summary["episodes"])
        ax.axhline(exact_summary[metric], color="tab:blue", ls="--", lw=1.8,
                   label="Exact MuJoCo MPPI")
        ax.fill_between([-0.3, len(x) - .7],
                        exact_summary[metric] - exact_ci, exact_summary[metric] + exact_ci,
                        color="tab:blue", alpha=.12)
        ax.set(xticks=x, xticklabels=labels, xlabel="Smoothness penalty weight", ylabel=ylabel)
        ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("FlowSDE-MPPI smoothness sweep across 80 episodes (mean ± 95% CI)")
    fig.savefig(args.root / "smoothness_sweep_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ci = 1.96 * summary.mean_runtime_ms_std / np.sqrt(summary.episodes)
    ax.errorbar(x, summary.mean_runtime_ms, yerr=ci, marker="o", capsize=3,
                color="tab:orange", lw=1.8, label="FlowSDE MPPI")
    ax.axhline(exact_summary["mean_runtime_ms"], color="tab:blue", ls="--",
               lw=1.8, label="Exact MuJoCo MPPI")
    ax.set(xticks=x, xticklabels=labels, xlabel="Smoothness penalty weight",
           ylabel="Mean inference time [ms]", title="MPPI runtime")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(args.root / "smoothness_sweep_runtime.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
