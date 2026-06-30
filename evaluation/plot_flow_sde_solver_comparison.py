"""Plot RealNVPFlowSDE solver metrics produced by eval_flow_sde_lorenz96."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in args.metrics:
        with np.load(path) as data:
            records.append({
                "solver": str(data["solver"]),
                "rmse": float(data["rmse_all"]),
                "rmse_obs": float(data["rmse_obs"]),
                "rmse_unobs": float(data["rmse_unobs"]),
                "mae": float(data["mae_all"]),
                "coverage": 100.0 * float(data["coverage"]),
            })
    records.sort(key=lambda record: record["rmse"])

    labels = [record["solver"].replace("_", " ").title() for record in records]
    x = np.arange(len(records))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(x - width, [r["rmse"] for r in records], width, label="All")
    axes[0].bar(x, [r["rmse_obs"] for r in records], width, label="Observed")
    axes[0].bar(x + width, [r["rmse_unobs"] for r in records], width, label="Unobserved")
    axes[0].set_title("Posterior mean RMSE")
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, [r["mae"] for r in records], color="tab:orange")
    axes[1].set_title("Posterior mean MAE")
    axes[1].set_ylabel("MAE (lower is better)")

    axes[2].bar(x, [r["coverage"] for r in records], color="tab:green")
    axes[2].axhline(90.0, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("90% interval coverage")
    axes[2].set_ylabel("Coverage (%)")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle("RealNVPFlowSDE solver comparison: observation-anchored segmented inference")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
