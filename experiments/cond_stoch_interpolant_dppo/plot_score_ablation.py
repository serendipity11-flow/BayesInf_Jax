"""Plot pretraining-only CSI score ablations from an evaluation CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(results: Path, output: Path, title: str) -> None:
    with results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {results}")
    method_order = {
        "CSI-ODE": 0,
        "CSI-SDE-analytic": 1,
        "CSI-SDE-learned": 2,
        "CSI-analytic-SDE": 0,
        "CSI-analytic-ODE": 1,
        "CSI-hybrid-SDE": 2,
        "CSI-learned-SDE": 3,
        "DDPM": 4,
        "DDPM-BC": 4,
    }
    rows.sort(key=lambda row: method_order.get(row["method"], len(method_order)))

    with plt.rc_context({
        "font.size": 19, "axes.titlesize": 20, "axes.labelsize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 18,
        "legend.fontsize": 17, "figure.titlesize": 22,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        labels = [row["method"].replace("CSI-", "CSI\n").replace("-", "\n") for row in rows]
        x = np.arange(len(rows))
        figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
        specifications = (
            ("success_rate_mean_mean", "success_rate_mean_std", "Success rate", (0, 1.05)),
            ("return_mean_mean", "return_mean_std", "Episode return", None),
        )
        colors = ["#4c78a8" if not row["method"].startswith("DDPM") else "#f58518"
                  for row in rows]
        for axis, (mean_key, std_key, ylabel, limits) in zip(axes, specifications):
            means = np.asarray([float(row[mean_key]) for row in rows])
            stds = np.asarray([float(row[std_key]) for row in rows])
            axis.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=.9,
                     edgecolor="black", linewidth=.7)
            axis.set_xticks(x, labels)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.25)
            axis.spines[["top", "right"]].set_visible(False)
            if limits is not None:
                axis.set_ylim(*limits)
        figure.suptitle(f"{title} behavior-cloning score ablation (mean ± seed SD)")
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=240, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Door")
    arguments = parser.parse_args()
    main(arguments.results, arguments.output, arguments.title)
