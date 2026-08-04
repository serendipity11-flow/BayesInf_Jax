"""Plot the recovery/NMPC flow and MPC seed-0 results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PUBLISHED = Path("results/results_optimal_control")


def read(path):
    with path.open() as file:
        return [{key: value if key == "method" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(file)]


def main():
    new = read(
        PUBLISHED
        / "amortized_seq_flow_recovery_nmpc_seed0_closed_loop_metrics.csv"
    )
    series = {
        "Recovery/NMPC flow": [row for row in new if row["method"] == "amortized_seq_flow"],
        "Linear MPC": [row for row in new if row["method"] == "linear_mpc"],
        "Nonlinear MPC": [row for row in new if row["method"] == "nonlinear_mpc"],
    }
    colors = {
        "Recovery/NMPC flow": "#d95f02",
        "Linear MPC": "#377eb8", "Nonlinear MPC": "#1b9e77",
    }
    specs = [
        ("lateral_rmse", "Lateral RMSE [m]", False),
        ("max_abs_lateral", "Maximum |lateral error| [m]", False),
        ("steering_rate_rms", "Steering-rate RMS [rad/step]", False),
        ("mean_runtime_ms", "Runtime [ms/command]", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (field, ylabel, log) in zip(axes.flat, specs):
        for label, rows in series.items():
            axis.plot([row["noise_std"] for row in rows],
                      [row[field] for row in rows], marker="o", lw=2,
                      label=label, color=colors[label])
        if log:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0.0)
        axis.set(xlabel="Measurement noise std [m]", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Seed 0 closed-loop comparison on nonlinear bicycle plant")
    output = (
        PUBLISHED
        / "amortized_seq_flow_before_after_recovery_nmpc_seed0.png"
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
