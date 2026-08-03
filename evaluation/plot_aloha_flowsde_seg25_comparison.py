#!/usr/bin/env python3
"""Plot test metrics from the segment-25, continuity-weight-1 FlowSDE sweep."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/results_state_estimation/Aloha_sim_insertion")
RUNS = (
    ("Dataset noisy", ROOT / "logs/flowsde_s25_cw1_13833_0.out"),
    ("Pseudo-process noise", ROOT / "logs/flowsde_s25_cw1_13833_1.out"),
    ("MuJoCo random", ROOT / "logs/flowsde_s25_cw1_13833_2.out"),
)


def read_metrics(path: Path) -> tuple[float, float, float]:
    text = path.read_text()
    patterns = (
        r"RMSE posterior mean:\s+([0-9.]+)",
        r"MAE\s+posterior mean:\s+([0-9.]+)",
        r"Coverage @ 90% CI:\s+([0-9.]+)%",
    )
    values = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if not matches:
            raise ValueError(f"Metric {pattern!r} missing from {path}")
        values.append(float(matches[-1]))
    return values[0], values[1], values[2] / 100.0


def main() -> None:
    rows = [(label, *read_metrics(path)) for label, path in RUNS]
    summary = ROOT / "flow_sde_seg25_cw1_noenergy_metrics.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "rmse", "mae", "coverage_90"))
        writer.writerows(rows)

    labels = [row[0] for row in rows]
    rmse = np.asarray([row[1] for row in rows])
    mae = np.asarray([row[2] for row in rows])
    coverage = np.asarray([row[3] for row in rows])
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(x - width / 2, rmse, width, label="RMSE")
    axes[0].bar(x + width / 2, mae, width, label="MAE")
    axes[0].set(
        xticks=x, xticklabels=labels, ylabel="Error",
        title="Posterior-mean accuracy",
    )
    axes[0].legend(frameon=False)
    bars = axes[1].bar(x, 100 * coverage, color="#59A14F")
    axes[1].axhline(90, color="black", linestyle="--", label="90% target")
    axes[1].bar_label(
        bars, labels=[f"{100 * value:.1f}%" for value in coverage], padding=3
    )
    axes[1].set(
        xticks=x, xticklabels=labels, ylabel="Coverage [%]",
        title="90% credible-interval coverage", ylim=(0, 105),
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=12)
    output = ROOT / "flow_sde_seg25_cw1_noenergy_metrics_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(summary)
    print(output)


if __name__ == "__main__":
    main()
