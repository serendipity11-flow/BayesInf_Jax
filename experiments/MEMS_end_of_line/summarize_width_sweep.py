"""Summarize the matched LSTM-BN encoder/flow width sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


WIDTHS = (8, 16, 32, 64)
METRICS = ("rmse", "r2", "coverage_90", "interval_width_90", "energy_score")


def main():
    root = Path(
        "results/results_parameter_identification/MEMS_end_of_line/encoder_width_sweep")
    rows = []
    for width in WIDTHS:
        result_dir = root / f"width_{width}"
        posterior = json.loads((result_dir / "lstm_bn_flow_metrics.json").read_text())
        training = json.loads(
            (result_dir / "lstm_bn_flow_training_metrics.json").read_text())
        rows.append({"width": width, **posterior, **training})

    columns = ("width", "epochs_trained", "best_validation_loss", *METRICS)
    with (root / "width_sweep_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 3.4))
    for axis, metric in zip(axes, METRICS):
        axis.plot(WIDTHS, [row[metric] for row in rows], marker="o")
        axis.set_xlabel("encoder and flow width")
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(WIDTHS)
        axis.grid(alpha=0.25)
    fig.suptitle("Matched LSTM + BatchNorm encoder/flow width sweep")
    fig.tight_layout()
    fig.savefig(root / "width_sweep_metrics.png", dpi=200)
    fig.savefig(root / "width_sweep_metrics.pdf")


if __name__ == "__main__":
    main()
