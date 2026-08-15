"""Plot the completed learned-vs-exact Jackal CEM benchmark."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


DISPLAY = {
    "learned_latent_ssm_flow_cem": "LatentSSMFlow CEM",
    "exact_simulator_cem": "Nominal-model CEM",
}
COLORS = {
    "learned_latent_ssm_flow_cem": "#2878B5",
    "exact_simulator_cem": "#F28E2B",
}


def run(result_dir: Path, output: Path, cpu_result_dir: Path | None = None):
    summary = json.loads(
        (result_dir / "jackal_cem_comparison.json").read_text()
    )
    trajectories = np.load(result_dir / "jackal_cem_trajectories.npz")
    names = list(summary["closed_loop"])
    figure = plt.figure(figsize=(13.5, 8))
    grid = figure.add_gridspec(
        8, 2, width_ratios=(1.45, 1.0), hspace=0.2, wspace=0.2,
    )
    trajectory_axis = figure.add_subplot(grid[0:7, 0])
    performance_axis = figure.add_subplot(grid[0:3, 1])
    runtime_axis = figure.add_subplot(grid[4:7, 1])

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
    trajectory_axis.legend(loc="upper left", fontsize=14)

    closed = summary["closed_loop"]
    metric_names = ["Success", "Collision-\nfree", "Safety-\ncompliant"]
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
            padding=0, fontsize=11, label_type="center", color="white",
            fontweight="bold",
        )
    performance_axis.set_xticks(x_positions, metric_names)
    performance_axis.set_ylim(0, 1.15)
    performance_axis.set_ylabel("Episode fraction")
    episodes = closed[names[0]]["episodes"]
    performance_axis.set_title(f"Task and safety outcomes ({episodes} episodes)")
    performance_axis.grid(axis="y", alpha=0.25)
    performance_axis.legend(fontsize=12, loc="lower center")

    if cpu_result_dir is None:
        cpu_result_dir = result_dir
    cpu_summary = json.loads(
        (cpu_result_dir / "jackal_cem_comparison.json").read_text()
    )
    runtimes = [
        cpu_summary["closed_loop"]["learned_latent_ssm_flow_cem"]["mean_command_time_s"],
        closed["learned_latent_ssm_flow_cem"]["mean_command_time_s"],
        cpu_summary["closed_loop"]["exact_simulator_cem"]["mean_command_time_s"],
    ]
    runtime_labels = ["LSSMFlow\nCPU", "LSSMFlow\nGPU", "Nominal\nCPU"]
    bars = runtime_axis.bar(
        runtime_labels, runtimes,
        color=["#6BAED6", "#2878B5", "#F28E2B"], width=0.68,
    )
    runtime_axis.bar_label(
        bars, labels=[f"{value:.3g} s" for value in runtimes], padding=3,
        fontsize=14,
    )
    runtime_axis.set_ylabel("Time per MPC command [s]")
    runtime_axis.set_title("CEM computation time (lower is better)")
    runtime_axis.set_yscale("log")
    runtime_axis.tick_params(axis="x", labelsize=12)
    runtime_axis.set_ylim(min(runtimes) * 0.45, max(runtimes) * 2.3)
    runtime_axis.grid(axis="y", which="both", alpha=0.25)

    figure.subplots_adjust(top=0.96)
    # The equal-aspect trajectory axis is shorter than its nominal grid cell.
    # Align the two right-hand axes to its final rendered top and bottom edges.
    figure.canvas.draw()
    left_box = trajectory_axis.get_position()
    performance_box = performance_axis.get_position()
    gap = 0.24 * left_box.height
    panel_height = (left_box.height - gap) / 2
    runtime_axis.set_position([
        performance_box.x0, left_box.y0, performance_box.width, panel_height,
    ])
    performance_axis.set_position([
        performance_box.x0, left_box.y0 + panel_height + gap,
        performance_box.width, panel_height,
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("result_dir", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--cpu-result-dir", type=Path)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    destination = arguments.output or (
        arguments.result_dir / "jackal_cem_comparison_dashboard.png"
    )
    run(arguments.result_dir, destination, arguments.cpu_result_dir)
