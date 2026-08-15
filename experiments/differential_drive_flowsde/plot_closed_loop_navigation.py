"""Re-render saved FlowSDE closed-loop navigation trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .env import DifferentialDriveEnv


def run(output: Path):
    metrics = json.loads((output / "closed_loop_metrics.json").read_text())
    figure, ax = plt.subplots(figsize=(10, 7.5))
    terrain = DifferentialDriveEnv(seed=0)
    gx, gy = np.meshgrid(np.linspace(-4.5, 4.5, 160), np.linspace(-3.5, 3.5, 130))
    mu = np.vectorize(terrain.friction_at)(gx, gy)
    image = ax.contourf(gx, gy, mu, levels=14, cmap="YlGn", alpha=.55)
    optimizer = "mppi" if "MPPI" in metrics["controller"] else "cem"

    for episode in metrics["episodes"]:
        record = np.load(output / f"flowsde_{optimizer}_task{episode['task']}.npz")
        xy, waypoints = record["states"][:, :2], record["waypoints"]
        line, = ax.plot(*xy.T, lw=3.2, label=f"Task {episode['task']}")
        ax.plot(*xy[0], "o", ms=11, color=line.get_color())
        ax.plot(waypoints[:, 0], waypoints[:, 1], "*--", ms=18, lw=2.1,
                color=line.get_color())

    # Derive the colorbar axes from the navigation axes so that the bar retains
    # exactly the same height when the equal data aspect shrinks the main plot.
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="4.5%", pad=0.28)
    colorbar = figure.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Local friction coefficient", fontsize=21)
    colorbar.ax.tick_params(labelsize=18)
    ax.set_xlabel("x [m]", fontsize=22); ax.set_ylabel("y [m]", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.legend(fontsize=18, loc="best")
    figure.tight_layout()
    figure.savefig(output / "closed_loop_navigation.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "closed_loop_navigation.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
