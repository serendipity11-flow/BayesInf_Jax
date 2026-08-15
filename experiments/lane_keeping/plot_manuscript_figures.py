"""Generate manuscript-ready PDF figures for the bicycle lane experiment."""

import csv
from pathlib import Path

import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control.amortized_seq_flow import AmortizedSeqFlow
from .dataset import BicycleConfig
from .evaluate import (
    make_scenario,
    run_flow,
    run_linear_mpc_kf,
    run_nonlinear_mpc_kf,
)


OUTPUT = Path("results/results_optimal_control/bycicle_lane")
CHECKPOINT = OUTPUT / "amortized_seq_flow_recovery_nmpc_checkpoint.eqx"
NOISE_CSV = OUTPUT / "all_flow_variants_noise_sweep.csv"
METHODS = {
    "flow_recovery_nmpc": ("AmortizedSeqFlow", "#d95f02"),
    "linear_mpc": ("Linear MPC", "#377eb8"),
    "nonlinear_mpc": ("Nonlinear MPC", "#1b9e77"),
}


def configure_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "figure.titlesize": 19,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def trajectory_figure():
    cfg = BicycleConfig()
    model = AmortizedSeqFlow(cfg.horizon, cfg.delta_max, jr.PRNGKey(0))
    with CHECKPOINT.open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)

    key = jr.PRNGKey(0)
    scenario = make_scenario(key, cfg, 300, cfg.obs_std)
    cases = [
        ((0.0, 0.0), r"Nominal: $e_y=0$ m, $e_\psi=0$ rad"),
        ((0.5, -0.2), r"Moderate recovery: $e_y=+0.5$ m, $e_\psi=-0.2$ rad"),
        ((-1.0, -0.2), r"Challenging recovery: $e_y=-1.0$ m, $e_\psi=-0.2$ rad"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True,
                             constrained_layout=True)
    time = np.arange(300) * cfg.dt
    for axis, (initial, title) in zip(axes, cases):
        results = {
            "flow_recovery_nmpc": run_flow(model, scenario, key, cfg, 300, initial),
            "linear_mpc": run_linear_mpc_kf(scenario, cfg, 300, cfg.obs_std, initial),
            "nonlinear_mpc": run_nonlinear_mpc_kf(scenario, cfg, 300, cfg.obs_std, initial),
        }
        for name, result in results.items():
            label, color = METHODS[name]
            axis.plot(time, result[0], color=color, label=label)
        axis.axhline(0, color="black", lw=1.0)
        axis.set_ylabel("Lateral error [m]")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("Closed-loop trajectories for representative initial conditions")
    path = OUTPUT / "amortized_seq_flow_vs_mpc_initial_condition_trajectories.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def noise_figure():
    with NOISE_CSV.open() as file:
        rows = list(csv.DictReader(file))
    specs = [
        ("lateral_rmse", "Lateral RMSE [m]", False),
        ("max_abs_lateral", "Maximum |lateral error| [m]", False),
        ("steering_rate_rms", "Steering-rate RMS [rad/step]", False),
        ("mean_runtime_ms", "Runtime [ms/command]", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2), constrained_layout=True)
    for axis, (field, ylabel, use_log_scale) in zip(axes, specs):
        for name, (label, color) in METHODS.items():
            group = [row for row in rows if row["method"] == name]
            axis.plot(
                np.arange(len(group)),
                [float(row[field]) for row in group],
                "o-", markersize=8, label=label, color=color,
            )
        noise_levels = [
            float(row["noise_std"])
            for row in rows if row["method"] == next(iter(METHODS))
        ]
        axis.set_xticks(
            np.arange(len(noise_levels)),
            [f"{value:.2f}" for value in noise_levels],
        )
        axis.set_ylabel(ylabel)
        if use_log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Closed-loop performance under measurement noise")
    fig.supxlabel(r"Noise $\sigma$ [m]")
    path = OUTPUT / "amortized_seq_flow_vs_mpc_noise_sweep.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    configure_style()
    print(f"saved {trajectory_figure()}")
    print(f"saved {noise_figure()}")


if __name__ == "__main__":
    main()
