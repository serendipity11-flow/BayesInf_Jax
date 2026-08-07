"""Clean matched-reset plot of the decision-relevant pretrained Door models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("DDPM", "base", "DiffusionPolicy-DPPO", "#F58518"),
    ("CSI-ODE\nHeun + path energy", "path", "CSI-ODE-Heun-PathEnergy", "#72B7B2"),
    ("CSI-SDE\nEuler, η=0.01", "path", "CSI-SDE-Euler-eta0.01-PathEnergy", "#4C78A8"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot(base_path: Path, path_energy_path: Path, output_dir: Path) -> None:
    sources = {"base": read_rows(base_path), "path": read_rows(path_energy_path)}
    # The path-energy pilot used reset seeds 1000--1019. Apply that exact set
    # to DDPM as well; all reported bars therefore contain 3 x 20 rollouts.
    reset_seeds = {str(seed) for seed in range(1000, 1020)}
    summaries = []
    seed_values = []
    for label, source, method, _ in MODELS:
        selected = [
            row for row in sources[source]
            if row["method"] == method
            and row["stage"] == "pretrain"
            and row["reset_seed"] in reset_seeds
        ]
        if len(selected) != 60:
            raise ValueError(f"{method}: expected 60 matched episodes, got {len(selected)}")
        per_seed = []
        for seed in (0, 1, 2):
            episodes = [row for row in selected if int(row["seed"]) == seed]
            per_seed.append(np.mean([int(row["success"]) for row in episodes]))
        values = np.asarray(per_seed)
        seed_values.append(values)
        summaries.append({
            "model": label.replace("\n", " "),
            "episodes": len(selected),
            "success_rate": float(values.mean()),
            "seed_std": float(values.std(ddof=1)),
            "seed0": float(values[0]),
            "seed1": float(values[1]),
            "seed2": float(values[2]),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)

    x = np.arange(len(MODELS))
    means = np.asarray([row["success_rate"] for row in summaries])
    stds = np.asarray([row["seed_std"] for row in summaries])
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(
        x, means, yerr=stds, capsize=5, width=0.64,
        color=[model[3] for model in MODELS], edgecolor="black",
        linewidth=0.8, alpha=0.9,
    )
    offsets = np.asarray([-0.11, 0.0, 0.11])
    for index, values in enumerate(seed_values):
        axis.scatter(
            index + offsets, values, s=44, color="white", edgecolor="black",
            linewidth=0.8, zorder=3,
        )
        axis.text(
            index, means[index] - 0.07, f"{100*means[index]:.1f}%",
            ha="center", va="center", fontsize=11, fontweight="bold",
        )
    axis.set_xticks(x, [model[0] for model in MODELS])
    axis.set_ylabel("Door-opening success rate")
    axis.set_ylim(0, 1.08)
    ticks = np.linspace(0, 1, 6)
    axis.set_yticks(ticks, [f"{100*tick:.0f}%" for tick in ticks])
    axis.set_title("Adroit Door pretrained policies (matched resets)")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "success_rate.png", dpi=220)
    figure.savefig(output_dir / "success_rate.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path,
        default=Path("results/results_optimal_control/door/generative_policy_evaluation/episodes.csv"),
    )
    parser.add_argument(
        "--path-energy", type=Path,
        default=Path("results/results_optimal_control/door/si_analytic_path_bc_evaluation/episodes.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/results_optimal_control/door/pretrain_clean_comparison"),
    )
    args = parser.parse_args()
    plot(args.base, args.path_energy, args.output)
