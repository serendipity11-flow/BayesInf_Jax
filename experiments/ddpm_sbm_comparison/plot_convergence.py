"""Aggregate and visualize same-network convergence over cases and seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/ddpm_sbm_comparison/results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results / "theoretical_convergence_summary.pdf"
    files = sorted(args.results.glob("*_seed*/convergence_metrics.csv"))
    if not files:
        raise SystemExit(f"No convergence_metrics.csv files found below {args.results}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    # Distance is duplicated once for each sampler; retain one copy.
    distance = frame[frame.sampler == "ddpm"]
    cases = ("basic", "linear", "multimodal")
    colors = {"ddpm": "#2878B5", "sbm": "#E07B39"}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex="col")

    for column, case in enumerate(cases):
        case_distance = distance[distance.case == case]
        for trained_as in ("ddpm", "sbm"):
            subset = case_distance[case_distance.trained_as == trained_as]
            grouped = subset.groupby("steps").same_network_sliced_wasserstein
            steps = np.asarray(sorted(subset.steps.unique()))
            mean = grouped.mean().reindex(steps).to_numpy()
            std = grouped.std().reindex(steps).fillna(0).to_numpy()
            axes[0, column].plot(steps, mean, "o-", color=colors[trained_as],
                                 label=f"{trained_as.upper()}-trained score")
            axes[0, column].fill_between(steps, np.maximum(mean - std, 1e-8), mean + std,
                                         color=colors[trained_as], alpha=0.18)

        case_metrics = frame[frame.case == case]
        for trained_as in ("ddpm", "sbm"):
            subset = case_metrics[case_metrics.trained_as == trained_as]
            pivot = subset.pivot_table(index=["seed", "steps"], columns="sampler", values="energy_score")
            relative = ((pivot.ddpm - pivot.sbm) / pivot.sbm.abs().clip(lower=1e-8)).reset_index(name="difference")
            grouped = relative.groupby("steps").difference
            steps = np.asarray(sorted(relative.steps.unique()))
            mean = grouped.mean().reindex(steps).to_numpy()
            std = grouped.std().reindex(steps).fillna(0).to_numpy()
            axes[1, column].plot(steps, mean, "o-", color=colors[trained_as],
                                 label=f"{trained_as.upper()}-trained score")
            axes[1, column].fill_between(steps, mean - std, mean + std,
                                         color=colors[trained_as], alpha=0.18)

        axes[0, column].set_title(case.capitalize())
        axes[0, column].set_xscale("log")
        axes[0, column].set_yscale("log")
        axes[0, column].grid(True, which="both", alpha=0.25)
        axes[1, column].set_xscale("log")
        axes[1, column].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1, column].grid(True, which="both", alpha=0.25)
        axes[1, column].set_xlabel("Number of reverse steps")

    axes[0, 0].set_ylabel("Same-network sliced Wasserstein ↓")
    axes[1, 0].set_ylabel("Relative energy-score difference\n(DDPM sampler − SBM sampler)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("DDPM ancestral sampler converging to the reverse VP-SDE sampler", y=1.01)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    png = output.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {output} and {png} from {len(files)} result files")


if __name__ == "__main__":
    main()
