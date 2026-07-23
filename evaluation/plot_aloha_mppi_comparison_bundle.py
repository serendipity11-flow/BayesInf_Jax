#!/usr/bin/env python3
"""Visualize ALOHA MPPI comparison results.

This bundles the saved aggregate CSV/JSON metrics with the existing trajectory
plot into one figure that compares the FlowSDE rollout model against the exact
MuJoCo simulation model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.3f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def build_bundle(run_dir: Path) -> Path:
    csv_path = run_dir / "mppi_comparison.csv"
    json_path = run_dir / "mppi_comparison.json"
    traj_path = run_dir / "mppi_trajectories.png"
    out_path = run_dir / "mppi_comparison_bundle.png"

    for path in [csv_path, json_path, traj_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        meta = json.load(f)

    summary = df.groupby("controller", as_index=False).agg(
        tracking_rmse=("tracking_rmse", "mean"),
        terminal_rmse=("terminal_rmse", "mean"),
        action_rms_normalized=("action_rms_normalized", "mean"),
        median_solve_ms=("median_solve_ms", "mean"),
        mean_effective_sample_size=("mean_effective_sample_size", "mean"),
    )
    controllers = list(summary["controller"])

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

    metrics = [
        ("tracking_rmse", "Tracking RMSE"),
        ("terminal_rmse", "Terminal RMSE"),
        ("action_rms_normalized", "Action RMS"),
        ("median_solve_ms", "Median solve ms"),
        ("mean_effective_sample_size", "Mean ESS"),
    ]

    x = np.arange(len(controllers))
    width = 0.14
    for idx, (col, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, 0] if idx < 3 else gs[0, 1])
        offset = (idx % 3 - 1) * width if idx < 3 else (idx - 3) * width - 0.5 * width
        ax.bar(x + offset, summary[col].to_numpy(), width=width, label=title)
        ax.set_xticks(x)
        ax.set_xticklabels(controllers, rotation=15)
        ax.set_title("Aggregate MPPI metrics" if idx == 0 else "")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    traj_img = _load_image(traj_path)
    ax.imshow(traj_img)
    ax.axis("off")
    ax.set_title("Closed-loop trajectories")

    config = meta.get("configuration", {})
    title_lines = [
        f"flow_sde_mppi: tracking_rmse={_fmt(meta['flow_sde_mppi']['tracking_rmse'])}, terminal_rmse={_fmt(meta['flow_sde_mppi']['terminal_rmse'])}, solve_ms={_fmt(meta['flow_sde_mppi']['median_solve_ms'])}",
        f"physical_mppi: tracking_rmse={_fmt(meta['physical_mppi']['tracking_rmse'])}, terminal_rmse={_fmt(meta['physical_mppi']['terminal_rmse'])}, solve_ms={_fmt(meta['physical_mppi']['median_solve_ms'])}",
        f"comparison={config.get('comparison', 'n/a')}, horizon_steps={config.get('horizon_steps', 'n/a')}, tasks={config.get('tasks', 'n/a')}, candidates={config.get('candidates', 'n/a')}",
    ]
    fig.suptitle("\n".join(title_lines), fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Directory containing mppi_comparison.csv/json/png.")
    args = parser.parse_args()
    out_path = build_bundle(args.run_dir)
    print(out_path)


if __name__ == "__main__":
    main()
