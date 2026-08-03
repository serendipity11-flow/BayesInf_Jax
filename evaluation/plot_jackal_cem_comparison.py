"""Plot the completed learned-vs-exact Jackal CEM benchmark."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "learned_latent_ssm_flow_cem": "LatentSSMFlow CEM",
    "exact_simulator_cem": "Exact-model CEM",
}
COLORS = {
    "learned_latent_ssm_flow_cem": "#2878B5",
    "exact_simulator_cem": "#F28E2B",
}


def run(result_dir: Path, output: Path):
    summary = json.loads(
        (result_dir / "jackal_cem_comparison.json").read_text()
    )
    trajectories = np.load(result_dir / "jackal_cem_trajectories.npz")
    names = list(summary["closed_loop"])
    figure = plt.figure(figsize=(13.5, 8))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0),
                               hspace=0.34, wspace=0.27)
    trajectory_axis = figure.add_subplot(grid[:, 0])
    performance_axis = figure.add_subplot(grid[0, 1])
    runtime_axis = figure.add_subplot(grid[1, 1])

    for name in names:
        prefix = f"{name}_episode_"
        keys = sorted(
            (key for key in trajectories.files if key.startswith(prefix)),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        for index, key in enumerate(keys):
            value = trajectories[key]
            trajectory_axis.plot(
                value[:, 0], value[:, 1], color=COLORS[name], alpha=0.44,
                linewidth=1.7, label=DISPLAY[name] if index == 0 else None,
            )
    settings = summary["settings"]
    robot_radius = 0.27
    safety_margin = 0.18
    for index, (x, y, radius) in enumerate(settings["obstacles"]):
        trajectory_axis.add_patch(plt.Circle(
            (x, y), radius, color="#C44E52", alpha=0.60,
            label="Obstacle" if index == 0 else None,
        ))
        trajectory_axis.add_patch(plt.Circle(
            (x, y), radius + robot_radius + safety_margin,
            facecolor="#C44E52", edgecolor="#C44E52", alpha=0.13,
            linestyle="--", linewidth=1.2,
            label="Safety envelope" if index == 0 else None,
        ))
    trajectory_axis.scatter(
        *settings["goal"], marker="*", s=220, color="#2CA02C",
        edgecolor="white", linewidth=0.8, label="Goal", zorder=5,
    )
    trajectory_axis.set(
        xlabel="x [m]", ylabel="y [m]",
        title="Closed-loop trajectories on the exact noisy plant",
    )
    trajectory_axis.set_aspect("equal")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend(loc="upper left", fontsize=9)

    closed = summary["closed_loop"]
    metric_names = ["Success", "Collision-free", "Safety-compliant"]
    x_positions = np.arange(len(metric_names))
    width = 0.34
    for index, name in enumerate(names):
        values = [
            closed[name]["success_rate"],
            1.0 - closed[name]["collision_rate"],
            1.0 - closed[name]["safety_violation_rate"],
        ]
        bars = performance_axis.bar(
            x_positions + (index - 0.5) * width, values, width,
            color=COLORS[name], label=DISPLAY[name],
        )
        performance_axis.bar_label(
            bars, labels=[f"{100 * value:.0f}%" for value in values],
            padding=2, fontsize=9,
        )
    performance_axis.set_xticks(x_positions, metric_names)
    performance_axis.set_ylim(0, 1.15)
    performance_axis.set_ylabel("Episode fraction")
    performance_axis.set_title("Task and safety outcomes (8 episodes)")
    performance_axis.grid(axis="y", alpha=0.25)
    performance_axis.legend(fontsize=9, loc="lower center")

    runtimes = [closed[name]["mean_command_time_s"] for name in names]
    bars = runtime_axis.bar(
        [DISPLAY[name] for name in names], runtimes,
        color=[COLORS[name] for name in names], width=0.6,
    )
    runtime_axis.bar_label(
        bars, labels=[f"{value:.2f} s" for value in runtimes], padding=3
    )
    runtime_axis.set_ylabel("Mean time per MPC command [s]")
    runtime_axis.set_title("CPU computation cost (lower is better)")
    runtime_axis.grid(axis="y", alpha=0.25)
    speed_ratio = runtimes[0] / runtimes[1]
    runtime_axis.text(
        0.5, 0.78, f"Learned rollout is {speed_ratio:.1f}× slower",
        transform=runtime_axis.transAxes, ha="center", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    learned_open = summary["open_loop"]["learned_latent_ssm_flow"]
    figure.suptitle(
        "Jackal obstacle avoidance: LatentSSMFlow vs exact-model stochastic CEM\n"
        f"Learned open-loop position RMSE {learned_open['position_rmse_m']:.3f} m, "
        f"yaw RMSE {learned_open['yaw_rmse_rad']:.3f} rad, "
        f"90% coverage {100 * summary['open_loop']['learned_position_90pct_coverage']:.1f}%",
        fontsize=13,
    )
    figure.subplots_adjust(top=0.87)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("result_dir", type=Path)
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    destination = arguments.output or (
        arguments.result_dir / "jackal_cem_comparison_dashboard.png"
    )
    run(arguments.result_dir, destination)
