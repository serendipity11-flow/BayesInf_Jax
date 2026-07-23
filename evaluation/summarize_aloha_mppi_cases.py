#!/usr/bin/env python3
"""Summarize a completed per-episode ALOHA MPPI evaluation."""

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
    ("mean_runtime_ms", "Mean inference time [ms]"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="MPPI performance across all episodes")
    parser.add_argument("--expected-episodes", type=int, default=80)
    args = parser.parse_args()

    cases = pd.read_csv(args.case_metrics)
    if len(cases) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} rows, found {len(cases)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for metric, _ in METRICS:
        values = cases[metric].to_numpy()
        summary_rows.append({
            "metric": metric, "mean": np.mean(values), "std": np.std(values),
            "median": np.median(values), "q25": np.quantile(values, .25),
            "q75": np.quantile(values, .75), "min": np.min(values),
            "max": np.max(values),
        })
    pd.DataFrame(summary_rows).to_csv(
        args.output.with_name(args.output.stem + "_summary.csv"), index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        values = cases[metric].to_numpy()
        violin = ax.violinplot([values], positions=[0], widths=.8,
                               showmeans=False, showmedians=False, showextrema=False)
        violin["bodies"][0].set_facecolor("tab:orange")
        violin["bodies"][0].set_edgecolor("tab:orange")
        violin["bodies"][0].set_alpha(.25)
        box = ax.boxplot([values], positions=[0], widths=.25, patch_artist=True,
                         showfliers=False, medianprops={"color": "black", "lw": 1.5})
        box["boxes"][0].set_facecolor("tab:orange")
        box["boxes"][0].set_alpha(.75)
        ax.scatter(rng.uniform(-.09, .09, len(values)), values, s=10,
                   color="tab:orange", alpha=.4, edgecolors="none")
        ax.set(xticks=[0], xticklabels=["FlowSDE MPPI"], ylabel=label, title=label)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle(args.title)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
