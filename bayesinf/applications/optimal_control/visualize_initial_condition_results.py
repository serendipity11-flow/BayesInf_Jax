"""Create performance and trajectory figures for the initial-state sweep."""

import csv
from pathlib import Path

import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from .amortized_seq_flow import AmortizedSeqFlow
from .bicycle_dataset import BicycleConfig
from .evaluate_lane_keeping import (
    make_scenario,
    run_flow,
    run_linear_mpc_kf,
    run_nonlinear_mpc_kf,
)


OUTPUT = Path("results/results_optimal_control")
CHECKPOINT = OUTPUT / "amortized_seq_flow_recovery_nmpc_seed0_checkpoint.eqx"
CSV_PATH = OUTPUT / "initial_condition_sweep_seed0.csv"
METHODS = {
    "amortized_seq_flow": ("Amortized flow", "#d95f02"),
    "linear_mpc": ("Linear MPC", "#377eb8"),
    "nonlinear_mpc": ("Nonlinear MPC", "#1b9e77"),
}


def load_rows():
    with CSV_PATH.open() as file:
        return [
            {key: value if key == "method" else float(value)
             for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def performance_figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    names = list(METHODS)
    positions = np.arange(len(names))
    rmse_groups = [
        [row["lateral_rmse"] for row in rows if row["method"] == name]
        for name in names
    ]
    boxes = axes[0].boxplot(rmse_groups, positions=positions, widths=0.55,
                            patch_artist=True, medianprops={"color": "black"})
    for patch, name in zip(boxes["boxes"], names):
        patch.set_facecolor(METHODS[name][1]); patch.set_alpha(0.8)
    axes[0].set_xticks(positions, [METHODS[name][0] for name in names], rotation=12)
    axes[0].set(ylabel="Lateral RMSE [m]", title="Accuracy over 15 initial states")
    axes[0].set_ylim(bottom=0); axes[0].grid(alpha=0.25, axis="y")

    runtime = [np.mean([row["mean_runtime_ms"] for row in rows
                        if row["method"] == name]) for name in names]
    bars = axes[1].bar(positions, runtime, color=[METHODS[name][1] for name in names])
    axes[1].bar_label(bars, labels=[f"{value:.1f}" for value in runtime], padding=3)
    axes[1].set_xticks(positions, [METHODS[name][0] for name in names], rotation=12)
    axes[1].set(ylabel="Runtime [ms/command]", title="Mean computation time")
    axes[1].set_ylim(0, max(runtime) * 1.18); axes[1].grid(alpha=0.25, axis="y")

    lateral = sorted({row["initial_lateral_error"] for row in rows})
    heading = sorted({row["initial_heading_error"] for row in rows})
    difference = np.array([
        [next(row["lateral_rmse"] for row in rows
              if row["method"] == "amortized_seq_flow"
              and row["initial_lateral_error"] == ey
              and row["initial_heading_error"] == ep)
         - next(row["lateral_rmse"] for row in rows
                if row["method"] == "nonlinear_mpc"
                and row["initial_lateral_error"] == ey
                and row["initial_heading_error"] == ep)
         for ey in lateral] for ep in heading
    ])
    bound = np.max(np.abs(difference))
    image = axes[2].imshow(difference, origin="lower", aspect="auto",
                           cmap="RdBu_r", vmin=-bound, vmax=bound)
    axes[2].set_xticks(range(len(lateral)), [f"{value:+.1f}" for value in lateral])
    axes[2].set_yticks(range(len(heading)), [f"{value:+.2f}" for value in heading])
    axes[2].set(xlabel="Initial lateral error [m]",
                ylabel="Initial heading error [rad]",
                title="Flow RMSE − nonlinear MPC RMSE")
    for i in range(len(heading)):
        for j in range(len(lateral)):
            axes[2].text(j, i, f"{difference[i, j]:+.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[2], label="RMSE difference [m]")
    fig.suptitle("Initial-condition robustness on the nonlinear plant")
    path = OUTPUT / "initial_condition_performance_seed0.png"
    fig.savefig(path, dpi=220); plt.close(fig)
    return path


def trajectory_figure():
    cfg = BicycleConfig()
    model = AmortizedSeqFlow(cfg.horizon, cfg.delta_max, jr.PRNGKey(0))
    with CHECKPOINT.open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)
    key = jr.PRNGKey(0)
    scenario = make_scenario(key, cfg, 300, 0.12)
    cases = [
        ((0.0, 0.0), "Nominal: $e_y=0$, $e_\\psi=0$"),
        ((0.5, -0.2), "Flow-favorable: $e_y=0.5$, $e_\\psi=-0.2$"),
        ((-1.0, -0.2), "Difficult: $e_y=-1.0$, $e_\\psi=-0.2$"),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 8), sharex=True,
                             constrained_layout=True)
    time = np.arange(300) * cfg.dt
    for axis, (initial, title) in zip(axes, cases):
        results = {
            "amortized_seq_flow": run_flow(model, scenario, key, cfg, 300, initial),
            "linear_mpc": run_linear_mpc_kf(scenario, cfg, 300, 0.12, initial),
            "nonlinear_mpc": run_nonlinear_mpc_kf(scenario, cfg, 300, 0.12, initial),
        }
        for name, result in results.items():
            axis.plot(time, result[0], color=METHODS[name][1], lw=1.7,
                      label=METHODS[name][0])
        axis.axhline(0, color="black", lw=0.8)
        axis.set(ylabel="Lateral error [m]", title=title)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("Closed-loop trajectories for representative initial conditions")
    path = OUTPUT / "initial_condition_trajectories_seed0.png"
    fig.savefig(path, dpi=220); plt.close(fig)
    return path


def main():
    print(f"saved {performance_figure(load_rows())}")
    print(f"saved {trajectory_figure()}")


if __name__ == "__main__":
    main()
