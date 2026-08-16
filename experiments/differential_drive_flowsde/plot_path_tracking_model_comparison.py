"""Plot FlowSDE and physical-model closed-loop navigation side by side."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .env import DifferentialDriveEnv


def run(flowsde: Path, physical: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    flow_metrics = json.loads((flowsde / "closed_loop_metrics.json").read_text())
    physical_metrics = json.loads(
        (physical / "nonlinear_physical_mppi_metrics.json").read_text())

    figure = plt.figure(figsize=(18.5, 7.4))
    grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, .055), wspace=.08)
    axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    colorbar_ax = figure.add_subplot(grid[0, 2])

    terrain = DifferentialDriveEnv(seed=0)
    gx, gy = np.meshgrid(np.linspace(-4.5, 4.5, 160), np.linspace(-3.5, 3.5, 130))
    mu = np.vectorize(terrain.friction_at)(gx, gy)
    panels = (("NormFlowSDE", flowsde, flow_metrics, "flowsde_mppi_task", "controls"),
              ("Nonlinear physical model", physical, physical_metrics,
               "nonlinear_physical_mppi_task", "actions"))

    image = None
    for ax, (label, directory, metrics, prefix, _) in zip(axes, panels):
        image = ax.contourf(gx, gy, mu, levels=14, cmap="YlGn", alpha=.55)
        for episode in metrics["episodes"]:
            record = np.load(directory / f"{prefix}{episode['task']}.npz")
            xy = record["states"][:, :2]
            waypoints = np.asarray(episode["waypoints"])
            line, = ax.plot(*xy.T, lw=3.0, label=f"Task {episode['task']}")
            ax.plot(*xy[0], "o", ms=9, color=line.get_color())
            ax.plot(waypoints[:, 0], waypoints[:, 1], "*--", ms=11, lw=1.7,
                    color=line.get_color())
        ax.set_title(label, fontsize=24, pad=12)
        ax.set_xlabel("x [m]", fontsize=24)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=19)
    axes[0].set_ylabel("y [m]", fontsize=24)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(fontsize=17, loc="upper right")

    figure.subplots_adjust(left=.055, right=.94, bottom=.12, top=.91, wspace=.08)
    figure.canvas.draw()
    plot_box = axes[1].get_position()
    colorbar_box = colorbar_ax.get_position()
    colorbar_ax.set_position((colorbar_box.x0, plot_box.y0,
                              colorbar_box.width, plot_box.height))
    colorbar = figure.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Local friction coefficient", fontsize=23)
    colorbar.ax.tick_params(labelsize=19)

    figure.savefig(output / "closed_loop_navigation_model_comparison.pdf",
                   bbox_inches="tight")
    figure.savefig(output / "closed_loop_navigation_model_comparison.png",
                   dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowsde", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.flowsde, args.physical, args.output)
