"""Create publication figures for the Door CSI-versus-DDPM experiment."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


COLORS = {
    "CSI-DPPO": "#4C78A8", "DDPM-DPPO": "#F58518",
    "FM": "#54A24B", "SBM": "#B279A2",
}
LABELS = {
    "CSI-DPPO": "CSI", "DDPM-DPPO": "DDPM", "FM": "FM", "SBM": "SBM",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "seed", "iteration", "environment_steps", "return_mean",
            "success_rate_mean",
        ):
            row[key] = float(row[key])
    return rows


def seed_mean_sd(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def plot_curves(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True)
    specifications = (
        ("success_rate_mean", "Success rate", axes[0]),
        ("return_mean", "Episode return", axes[1]),
    )
    for metric, ylabel, axis in specifications:
        for method in ("CSI-DPPO", "DDPM-DPPO"):
            selected = [row for row in rows if row["method"] == method]
            grouped = defaultdict(list)
            for row in selected:
                grouped[int(row["environment_steps"])].append(row)
            x = sorted(grouped)
            stats = [seed_mean_sd(grouped[value], metric) for value in x]
            mean = np.asarray([value[0] for value in stats])
            sd = np.asarray([value[1] for value in stats])
            axis.plot(
                x, mean, marker="o", linewidth=2.4, markersize=6,
                color=COLORS[method], label=LABELS[method],
            )
            axis.fill_between(x, mean - sd, mean + sd,
                              color=COLORS[method], alpha=.2, linewidth=0)
        axis.set_xlabel("Environment steps", fontsize=16)
        axis.set_ylabel(ylabel, fontsize=16)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="y", alpha=.25, linewidth=.6)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylim(.78, 1.01)
    axes[0].legend(frameon=False, loc="lower right", fontsize=14)
    figure.suptitle("Door DPPO fine-tuning (mean ± seed SD)", fontsize=19)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_pretraining(rows: list[dict], baseline_root: Path, output: Path) -> None:
    pretrain = [row for row in rows if row["stage"] == "pretrain"]
    for method, directory in (("FM", "fm"), ("SBM", "sbm")):
        for seed in range(3):
            summary = json.loads(
                (baseline_root / directory / f"seed{seed}"
                 / "evaluation_summary.json").read_text()
            )
            pretrain.append({
                "method": method,
                "success_rate_mean": float(summary["success_rate"]),
                "return_mean": float(summary["return_mean"]),
            })
    methods = ("CSI-DPPO", "DDPM-DPPO", "FM", "SBM")
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.1))
    for metric, ylabel, axis in (
        ("success_rate_mean", "Success rate", axes[0]),
        ("return_mean", "Episode return", axes[1]),
    ):
        stats = [seed_mean_sd(
            [row for row in pretrain if row["method"] == method], metric
        ) for method in methods]
        means = [value[0] for value in stats]
        errors = [value[1] for value in stats]
        axis.bar(
            np.arange(len(methods)), means, yerr=errors, capsize=5,
            color=[COLORS[method] for method in methods],
            edgecolor="black", linewidth=.6,
        )
        axis.set_xticks(np.arange(len(methods)), [LABELS[method] for method in methods])
        axis.set_ylabel(ylabel, fontsize=16)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="y", alpha=.25, linewidth=.6)
    axes[0].set_ylim(0, 1.05)
    figure.suptitle("Door behavior-cloning pretraining (mean ± seed SD)", fontsize=19)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def merge_pngs(left: Path, right: Path, output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in (left, right)]
    height = max(image.height for image in images)
    resized = [
        image.resize((round(image.width * height / image.height), height), Image.Resampling.LANCZOS)
        if image.height != height else image
        for image in images
    ]
    canvas = Image.new("RGB", (sum(image.width for image in resized), height), "white")
    offset = 0
    for image in resized:
        canvas.paste(image, (offset, 0))
        offset += image.width
    canvas.save(output, "PDF", resolution=150.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--return-png", type=Path, required=True)
    parser.add_argument("--success-png", type=Path, required=True)
    parser.add_argument("--fm-sbm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.checkpoints)
    plot_curves(rows, args.output_dir / "door_csi_ddpm_finetune_curves.pdf")
    plot_pretraining(
        rows, args.fm_sbm_root,
        args.output_dir / "door_csi_ddpm_pretraining_comparison.pdf",
    )
    merge_pngs(
        args.return_png, args.success_png,
        args.output_dir / "door_return_success_curves_merged.pdf",
    )


if __name__ == "__main__":
    main()
