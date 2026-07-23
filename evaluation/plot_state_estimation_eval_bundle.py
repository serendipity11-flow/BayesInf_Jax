#!/usr/bin/env python3
"""Create a compact visualization bundle for a state-estimation evaluation run.

This script is intentionally lightweight: it reads the normalization statistics
and evaluation metrics from `.npz` files, and combines them with the existing
diagnostic/trajectory PNGs into a single summary figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _to_rgb_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _scale_image_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    target_w = max(1, int(round(img.shape[1] * scale)))
    return np.asarray(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


def _fmt_scalar(x) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            x = x.item()
        else:
            return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def build_bundle(checkpoint_prefix: Path, eval_prefix: Path | None = None) -> Path:
    if eval_prefix is None:
        eval_prefix = checkpoint_prefix

    norm_path = checkpoint_prefix.with_name(checkpoint_prefix.name + "_norm_stats.npz")
    metrics_path = eval_prefix.with_name(eval_prefix.name + "_eval_explicit_eval_continuous_obs1_metrics.npz")
    diag_path = eval_prefix.with_name(eval_prefix.name + "_eval_explicit_eval_continuous_obs1_diagnostics.png")
    traj_path = eval_prefix.with_name(eval_prefix.name + "_eval_explicit_eval_continuous_obs1_trajectories.png")
    out_path = eval_prefix.with_name(eval_prefix.name + "_eval_explicit_eval_continuous_obs1_bundle.png")

    for path in [norm_path, metrics_path, diag_path, traj_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    norm = _load_npz(norm_path)
    metrics = _load_npz(metrics_path)

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 0.45])

    # Normalization stats.
    ax = fig.add_subplot(gs[0, 0])
    x_mean = np.asarray(norm["x_mean"])
    x_std = np.asarray(norm["x_std"])
    x = np.arange(len(x_mean))
    ax.plot(x, x_mean, marker="o", label="x_mean")
    ax.plot(x, x_std, marker="s", label="x_std")
    ax.set_title("Normalization stats: state")
    ax.set_xlabel("dimension")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    y_mean = np.asarray(norm["y_mean"])
    y_std = np.asarray(norm["y_std"])
    y = np.arange(len(y_mean))
    ax.plot(y, y_mean, marker="o", label="y_mean")
    ax.plot(y, y_std, marker="s", label="y_std")
    ax.set_title("Normalization stats: observation")
    ax.set_xlabel("dimension")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    # Existing PNG diagnostics and trajectories.
    diag_img = _to_rgb_image(diag_path)
    traj_img = _to_rgb_image(traj_path)
    target_h = max(diag_img.shape[0], traj_img.shape[0])
    diag_img = _scale_image_to_height(diag_img, target_h)
    traj_img = _scale_image_to_height(traj_img, target_h)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(diag_img)
    ax.axis("off")
    ax.set_title("Diagnostics")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(traj_img)
    ax.axis("off")
    ax.set_title("Trajectories")

    # Metrics summary text.
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ordered_keys = [
        "solver",
        "sampler",
        "seg_length",
        "observation_stride",
        "diffusion_temp",
        "n_eval",
        "n_particles",
        "rmse_all",
        "rmse_obs",
        "rmse_unobs",
        "mae_all",
        "coverage",
    ]
    present = [k for k in ordered_keys if k in metrics]
    lines = [f"{k}: {_fmt_scalar(metrics[k])}" for k in present]
    ax.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    fig.suptitle(eval_prefix.name, fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_prefix",
        type=Path,
        help="Checkpoint artifact prefix, used for *_norm_stats.npz.",
    )
    parser.add_argument(
        "--eval-prefix",
        type=Path,
        default=None,
        help="Evaluation artifact prefix, used for metrics/diagnostic/trajectory PNGs.",
    )
    args = parser.parse_args()
    out_path = build_bundle(args.checkpoint_prefix, args.eval_prefix)
    print(out_path)


if __name__ == "__main__":
    main()
