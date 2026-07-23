#!/usr/bin/env python3
"""Aggregate the ALOHA FlowSDE-MPPI particle/risk sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = (
    ("final_normalized_rmse", "Final normalized RMSE"),
    ("cumulative_normalized_error", "Cumulative normalized error"),
    ("control_smoothness", "Control smoothness"),
    ("mean_runtime_ms", "Inference time [ms]"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("risk*_particles*/case_metrics.csv")):
        parts = path.parent.name.split("_particles")
        risk = float(parts[0].removeprefix("risk").replace("p", "."))
        particles = int(parts[1])
        data = pd.read_csv(path)
        if len(data) != 80 or set(data["model"]) != {"flow_sde"}:
            raise ValueError(f"incomplete sweep case {path}: rows={len(data)}")
        data.insert(0, "risk_weight", risk)
        data.insert(1, "particles", particles)
        rows.append(data)
    if not rows:
        raise FileNotFoundError(f"no completed cases under {args.root}")

    cases = pd.concat(rows, ignore_index=True)
    cases.to_csv(args.root / "particle_risk_per_episode.csv", index=False)
    summary = cases.groupby(["risk_weight", "particles"], as_index=False).agg(
        episodes=("episode_id", "size"),
        final_normalized_rmse=("final_normalized_rmse", "mean"),
        final_normalized_rmse_std=("final_normalized_rmse", "std"),
        cumulative_normalized_error=("cumulative_normalized_error", "mean"),
        cumulative_normalized_error_std=("cumulative_normalized_error", "std"),
        control_smoothness=("control_smoothness", "mean"),
        control_smoothness_std=("control_smoothness", "std"),
        mean_runtime_ms=("mean_runtime_ms", "mean"),
        mean_runtime_ms_std=("mean_runtime_ms", "std"),
    )
    summary.to_csv(args.root / "particle_risk_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for risk, group in summary.groupby("risk_weight"):
            group = group.sort_values("particles")
            ax.plot(group["particles"], group[metric], marker="o", label=f"risk={risk:g}")
        ax.set(xlabel="FlowSDE particles", ylabel=label, xticks=[4, 8, 16, 32])
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("FlowSDE-MPPI particle count and risk-weight sweep (80 episodes)")
    fig.savefig(args.root / "particle_risk_tradeoffs.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
