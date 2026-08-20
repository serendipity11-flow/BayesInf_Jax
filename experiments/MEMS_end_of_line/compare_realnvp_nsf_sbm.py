"""Compare the width-64 RealNVP posterior with tuned NSF and SBM heads."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_mems
from experiments.MEMS_end_of_line.plots import PARAM_NAMES


ROOT = Path("results/results_parameter_identification/MEMS_end_of_line")
OUTPUT_DIR = ROOT / "lstm_bn64_inference_heads"
RUNS = {
    "RealNVP": (ROOT / "lstm_bn_encoder_width_sweep/width_64", "lstm_bn_flow"),
    "NSF": (OUTPUT_DIR / "nsf", "nsf"),
    "SBM": (OUTPUT_DIR / "sbm", "sbm"),
}
METRICS = ("rmse", "mae", "r2", "coverage_90", "interval_width_90", "energy_score")
COLORS = {"RealNVP": "#4c78a8", "NSF": "#f58518", "SBM": "#54a24b"}


def main() -> None:
    rows, samples = [], {}
    for display_name, (run_dir, checkpoint_name) in RUNS.items():
        metrics = json.loads((run_dir / f"{checkpoint_name}_metrics.json").read_text())
        training = json.loads(
            (run_dir / f"{checkpoint_name}_training_metrics.json").read_text())
        rows.append({**metrics, **training, "model": display_name})
        samples[display_name] = np.load(run_dir / f"{checkpoint_name}_test_samples.npy")

    overall_fields = ("model", *METRICS, "epochs_trained", "best_validation_loss", "train_seconds")
    with (OUTPUT_DIR / "realnvp_nsf_sbm_comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    data = load_mems(DEFAULT_DATA_DIR, seed=0, standardize_signals=False)
    truth = np.asarray(data.test_theta)
    parameter_rows = []
    for model_name, draws in samples.items():
        mean = draws.mean(axis=0)
        low, high = np.quantile(draws, (0.05, 0.95), axis=0)
        for index, parameter in enumerate(PARAM_NAMES):
            error = mean[:, index] - truth[:, index]
            parameter_rows.append({
                "model": model_name,
                "parameter": parameter,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "coverage_90": float(np.mean(
                    (truth[:, index] >= low[:, index]) &
                    (truth[:, index] <= high[:, index]))),
                "interval_width_90": float(np.mean(high[:, index] - low[:, index])),
            })
    parameter_fields = ("model", "parameter", "rmse", "mae", "coverage_90", "interval_width_90")
    with (OUTPUT_DIR / "realnvp_nsf_sbm_parameter_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=parameter_fields)
        writer.writeheader()
        writer.writerows(parameter_rows)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    names = list(RUNS)
    for axis, metric in zip(axes.flat, METRICS):
        values = [next(row[metric] for row in rows if row["model"] == name) for name in names]
        axis.bar(names, values, color=[COLORS[name] for name in names])
        axis.set_title(metric.replace("_", " ").upper())
        axis.grid(axis="y", alpha=0.25)
        if metric == "coverage_90":
            axis.axhline(0.9, color="black", linestyle="--", linewidth=1, label="nominal")
            axis.legend(fontsize=8)
    fig.suptitle("MEMS end-of-line: width-64 LSTM-BN posterior comparison", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"realnvp_nsf_sbm_comparison.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    x = np.arange(len(PARAM_NAMES))
    width = 0.24
    for axis, metric in zip(axes.flat, ("rmse", "mae", "coverage_90",
                                        "interval_width_90")):
        for offset, name in enumerate(names):
            values = [row[metric] for row in parameter_rows if row["model"] == name]
            axis.bar(x + (offset - 1) * width, values, width, label=name,
                     color=COLORS[name])
        axis.set_xticks(x, PARAM_NAMES)
        axis.set_title(f"Per-parameter {metric.replace('_', ' ')}")
        axis.grid(axis="y", alpha=0.25)
        if metric == "coverage_90":
            axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("MEMS end-of-line: parameter-wise posterior quality", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"realnvp_nsf_sbm_parameter_comparison.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
