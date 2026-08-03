#!/usr/bin/env python3
"""Compare FlowSDE metrics across the three ALOHA dataset variants."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("results/results_state_estimation/Aloha_sim_insertion"),
    )
    parser.add_argument("--variant", default="uflow_obs1_ms5")
    args = parser.parse_args()
    variant = args.variant.strip("_")
    specs = [
        ("Dataset noisy", "dataset_noisy", f"realnvp_flow_sde_{variant}_eval_metrics.npz"),
        (
            "Pseudo-process noise", "pseudo_process_uncertainty",
            f"realnvp_flow_sde_{variant}_eval_metrics.npz",
        ),
        (
            "MuJoCo random", "mujoco_random",
            f"realnvp_flow_sde_gmm_{variant}_eval_metrics.npz",
        ),
    ]
    values = []
    for label, directory, filename in specs:
        with np.load(args.root / directory / filename) as data:
            values.append((label, float(data["rmse"]), float(data["mae"]), float(data["coverage"])))

    labels = [row[0] for row in values]
    rmse = np.asarray([row[1] for row in values])
    mae = np.asarray([row[2] for row in values])
    coverage = np.asarray([row[3] for row in values])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, rmse, width, label="RMSE")
    axes[0].bar(x + width / 2, mae, width, label="MAE")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Error", title="Posterior-mean accuracy")
    axes[0].legend(frameon=False)
    bars = axes[1].bar(x, 100 * coverage, color="#59A14F")
    axes[1].axhline(90, color="black", linestyle="--", label="90% target")
    axes[1].bar_label(bars, labels=[f"{100*v:.1f}%" for v in coverage], padding=3)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Coverage [%]",
                title="90% credible-interval coverage", ylim=(0, 105))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=12)
    output = args.root / f"flow_sde_{variant}_metrics_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
