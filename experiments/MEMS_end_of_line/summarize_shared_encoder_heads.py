"""Collect posterior metrics for heads using the frozen LSTM-BN64 encoder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "RealNVP": ("../lstm_bn_encoder_width_sweep/width_64", "lstm_bn_flow"),
    "BNN": ("bnn", "bnn"),
    "VAE": ("vae", "vae"),
    "NSF": ("nsf", "nsf"),
    "FM": ("fm", "fm"),
    "DDPM": ("ddpm", "ddpm"),
    "SBM": ("sbm", "sbm"),
}
METRICS = ("rmse", "r2", "coverage_90", "interval_width_90", "energy_score")


def main():
    root = Path(
        "results/results_parameter_identification/MEMS_end_of_line/"
        "lstm_bn64_inference_heads")
    rows = []
    for display_name, (relative_dir, checkpoint_name) in RUNS.items():
        run_dir = (root / relative_dir).resolve()
        metrics = json.loads(
            (run_dir / f"{checkpoint_name}_metrics.json").read_text())
        training = json.loads(
            (run_dir / f"{checkpoint_name}_training_metrics.json").read_text())
        rows.append({**metrics, **training, "model": display_name})
    fields = ("model", "epochs_trained", "best_validation_loss", *METRICS)
    with (root / "head_comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 3.5))
    for axis, metric in zip(axes, METRICS):
        axis.bar(list(RUNS), [row[metric] for row in rows])
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "MEMS inference architectures with the width-64 LSTM-BN encoder")
    fig.tight_layout()
    fig.savefig(root / "head_comparison_metrics.png", dpi=200)
    fig.savefig(root / "head_comparison_metrics.pdf")


if __name__ == "__main__":
    main()
