#!/usr/bin/env python3
"""Plot one FlowSDE-MPPI episode before and after an action smoothness penalty."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--penalized", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, penalized = load(args.baseline), load(args.penalized)
    with np.load(args.normalization) as stats:
        x_std = np.maximum(np.asarray(stats["x_std"]), 1e-8)

    runs = (("No smoothness penalty", baseline), ("Smoothness weight = 0.1", penalized))
    errors = [(run["states"] - run["goal"]) / x_std for _, run in runs]
    action_limit = max(np.abs(run["actions"]).max() for _, run in runs)
    error_limit = max(np.abs(error).max() for error in errors)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, ((label, run), error) in enumerate(zip(runs, errors)):
        image = axes[row, 0].imshow(
            error.T, aspect="auto", origin="lower", cmap="coolwarm",
            vmin=-error_limit, vmax=error_limit,
        )
        axes[row, 0].set(title=f"{label}: normalized state error", xlabel="control step",
                         ylabel="state dimension")
        fig.colorbar(image, ax=axes[row, 0], label="normalized error")

        image = axes[row, 1].imshow(
            run["actions"].T, aspect="auto", origin="lower", cmap="coolwarm",
            vmin=-action_limit, vmax=action_limit,
        )
        axes[row, 1].set(title=f"{label}: selected inputs", xlabel="control step",
                         ylabel="input dimension")
        fig.colorbar(image, ax=axes[row, 1], label="command")

    time = np.arange(baseline["states"].shape[0])
    for (label, run), error, color in zip(runs, errors, ("tab:blue", "tab:orange")):
        axes[0, 2].plot(time, np.sqrt(np.mean(error ** 2, axis=1)),
                        marker="o", color=color, label=label)
        delta = np.sqrt(np.mean(np.diff(run["actions"], axis=0) ** 2, axis=1))
        axes[1, 2].plot(time[1:], delta, marker="o", color=color, label=label)
    axes[0, 2].set(title="Tracking error by control step", xlabel="control step",
                   ylabel="normalized RMSE")
    axes[1, 2].set(title="Input change by control step", xlabel="control step",
                   ylabel="RMS Δu")
    for ax in axes[:, 2]:
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("FlowSDE MPPI episode 0: effect of action smoothness penalty")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
