"""Visualize one flow checkpoint against linear and nonlinear MPC."""

import argparse
import csv
from pathlib import Path

import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control import AmortizedSeqFlow, BicycleConfig
from bayesinf.applications.optimal_control.evaluate_lane_keeping import (
    make_scenario,
    metrics,
    run_flow,
    run_linear_mpc_kf,
    run_nonlinear_mpc_kf,
)


LABELS = {
    "amortized_seq_flow": "Amortized flow",
    "linear_mpc": "Linear MPC",
    "nonlinear_mpc": "Nonlinear MPC",
}
COLORS = {
    "amortized_seq_flow": "#d95f02",
    "linear_mpc": "#377eb8",
    "nonlinear_mpc": "#1b9e77",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--root", type=Path,
                        default=Path("results/results_optimal_control/runs"))
    parser.add_argument("--directory", type=str, default=None,
                        help="Result directory name; defaults to seed<seed>")
    args = parser.parse_args()

    cfg = BicycleConfig()
    output = args.root / (args.directory or f"seed{args.seed}")
    checkpoint = output / "lane_flow.eqx"
    model = AmortizedSeqFlow(cfg.horizon, cfg.delta_max, jr.PRNGKey(0))
    with checkpoint.open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)

    key = jr.fold_in(jr.PRNGKey(args.seed), 1)
    scenario = make_scenario(key, cfg, args.steps, args.noise)
    results = {
        "amortized_seq_flow": run_flow(model, scenario, key, cfg, args.steps),
        "linear_mpc": run_linear_mpc_kf(scenario, cfg, args.steps, args.noise),
        "nonlinear_mpc": run_nonlinear_mpc_kf(scenario, cfg, args.steps, args.noise),
    }
    rows = [metrics(name, args.noise, *result, cfg)
            for name, result in results.items()]
    with (output / "visualization_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    t = np.arange(args.steps) * cfg.dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, result in results.items():
        axes[0, 0].plot(t, result[0], color=COLORS[name], label=LABELS[name])
        axes[1, 0].plot(t, np.rad2deg(result[1]), color=COLORS[name], label=LABELS[name])
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set(xlabel="Time [s]", ylabel="Lateral error [m]", title="Tracking")
    limit = np.rad2deg(cfg.delta_max)
    axes[1, 0].axhline(limit, color="black", ls="--", lw=0.8)
    axes[1, 0].axhline(-limit, color="black", ls="--", lw=0.8)
    axes[1, 0].set(xlabel="Time [s]", ylabel="Steering [deg]", title="Control")

    names = list(results)
    x = np.arange(len(names))
    rmse_values = [r["lateral_rmse"] for r in rows]
    rmse_bars = axes[0, 1].bar(
        x, rmse_values, color=[COLORS[n] for n in names]
    )
    # A zero-based linear scale represents the small absolute RMSE difference
    # honestly; a truncated/log scale makes 0.128 versus 0.102 look dramatic.
    axes[0, 1].set_ylim(0.0, max(rmse_values) * 1.22)
    axes[0, 1].bar_label(rmse_bars, labels=[f"{value:.3f}" for value in rmse_values],
                         padding=3)
    axes[0, 1].set(ylabel="Lateral RMSE [m]", title="Closed-loop accuracy")
    axes[1, 1].bar(x, [r["mean_runtime_ms"] for r in rows],
                   color=[COLORS[n] for n in names])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(ylabel="Runtime [ms/command]", title="Computation")
    for ax in axes[:, 1]:
        ax.set_xticks(x, [LABELS[n] for n in names], rotation=12)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Seed {args.seed}: nonlinear plant, noise std={args.noise:.2f} m")
    path = output / "flow_linear_nonlinear_mpc.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"saved {path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
