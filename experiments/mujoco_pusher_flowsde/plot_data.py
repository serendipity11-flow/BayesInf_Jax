#!/usr/bin/env python3
"""Plot trajectory coverage and pseudo-process noise in a generated dataset."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=24)
    args = parser.parse_args()
    output = args.output or args.data.with_name(args.data.stem + "_overview.png")
    frame = pd.read_parquet(args.data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for episode, group in list(frame.groupby("episode_index"))[:args.episodes]:
        pose = np.stack(group["object_pose"])
        axes[0].plot(pose[:, 0], pose[:, 1], alpha=0.55, linewidth=1)
    axes[0].add_patch(Rectangle((0.225, 0.15), 0.11, 0.62, color="#555555"))
    axes[0].add_patch(Rectangle((0.225, -0.77), 0.11, 0.62, color="#555555"))
    axes[0].scatter([0.68], [0], s=180, facecolors="none", edgecolors="green", linewidth=2)
    axes[0].set(title="Cuboid trajectory coverage", xlabel="x [m]", ylabel="y [m]",
                xlim=(-0.65, 0.9), ylim=(-0.85, 0.85), aspect="equal")

    first = frame[frame.episode_index == frame.episode_index.iloc[0]]
    noise = np.stack(first["pseudo_process_noise"])
    axes[1].plot(first.frame_index, noise)
    axes[1].set(title="Pseudo-process disturbance", xlabel="step", ylabel="force / torque")
    axes[1].legend(("force x", "force y", "yaw torque"), frameon=False)

    for name in ("goal_cost", "obstacle_cost", "control_cost", "smoothness_cost"):
        values = frame.groupby("frame_index")[name].mean()
        axes[2].plot(values.index, values, label=name.replace("_", " "))
    axes[2].set(title="Mean cost decomposition", xlabel="step", ylabel="cost")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("MuJoCo narrow-passage pusher dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
