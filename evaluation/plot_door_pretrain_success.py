"""Plot seed-level success rates for pretrained Adroit Door policies only."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "StochasticInterpolant-SDE-DPPO": "CSI-SDE\n(analytic score)",
    "StochasticInterpolant-ODE-DPPO": "CSI-ODE",
    "DiffusionPolicy-DPPO": "Diffusion",
}
ORDER = list(LABELS)
COLORS = ["#4C78A8", "#72B7B2", "#F58518"]


def load_pretrain_success(path: Path) -> dict[str, list[tuple[int, float]]]:
    grouped = {method: [] for method in ORDER}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if row["stage"] == "pretrain" and method in grouped:
                grouped[method].append(
                    (int(row["seed"]), float(row["success_rate_mean"]))
                )
    missing = [method for method, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"missing pretrained evaluations for: {missing}")
    return grouped


def plot(input_path: Path, output_path: Path) -> None:
    grouped = load_pretrain_success(input_path)
    values = [np.asarray([value for _, value in grouped[m]]) for m in ORDER]
    means = np.asarray([array.mean() for array in values])
    stds = np.asarray([
        array.std(ddof=1) if len(array) > 1 else 0.0 for array in values
    ])
    x = np.arange(len(ORDER))

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.bar(
        x, means, yerr=stds, capsize=5, width=0.65, color=COLORS,
        edgecolor="black", linewidth=0.8, alpha=0.88,
    )
    offsets = np.linspace(-0.12, 0.12, max(len(array) for array in values))
    for index, array in enumerate(values):
        axis.scatter(
            index + offsets[:len(array)], array, s=42, color="white",
            edgecolor="black", linewidth=0.8, zorder=3,
        )
        axis.text(
            index, max(0.025, means[index] - 0.08), f"{100*means[index]:.1f}%",
            ha="center", va="center", fontsize=10, fontweight="bold",
        )

    axis.set_xticks(x, [LABELS[method] for method in ORDER])
    axis.set_ylabel("Door-opening success rate")
    axis.set_ylim(0.0, 1.08)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_yticklabels([f"{int(100*tick)}%" for tick in np.linspace(0, 1, 6)])
    axis.set_title("Adroit Door: pretrained policy success")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(
            "results/results_optimal_control/door/"
            "generative_policy_evaluation/checkpoints.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/results_optimal_control/door/"
            "generative_policy_evaluation/pretrain_success_rate.png"
        ),
    )
    args = parser.parse_args()
    plot(args.input, args.output)
