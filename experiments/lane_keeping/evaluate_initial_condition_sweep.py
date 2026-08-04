"""Compare flow and MPC controllers across initial tracking errors."""

import argparse
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
    metrics,
    run_flow,
    run_linear_mpc_kf,
    run_nonlinear_mpc_kf,
)


METHODS = {
    "amortized_seq_flow": ("Amortized flow", "#d95f02"),
    "linear_mpc": ("Linear MPC", "#377eb8"),
    "nonlinear_mpc": ("Nonlinear MPC", "#1b9e77"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("results/results_optimal_control/"
                     "amortized_seq_flow_recovery_nmpc_seed0_checkpoint.eqx"),
    )
    parser.add_argument("--output", type=Path,
                        default=Path("results/results_optimal_control"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--saturation-factor", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="initial_condition_sweep_seed0")
    args = parser.parse_args()

    cfg = BicycleConfig()
    model = AmortizedSeqFlow(
        cfg.horizon, cfg.delta_max, jr.PRNGKey(0),
        saturation_factor=args.saturation_factor,
    )
    with args.checkpoint.open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)

    lateral_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    heading_values = np.array([-0.20, 0.0, 0.20])
    key = jr.PRNGKey(args.seed)
    scenario = make_scenario(key, cfg, args.steps, args.noise)
    rows = []
    trajectories = {}

    for heading in heading_values:
        for lateral in lateral_values:
            initial = (float(lateral), float(heading))
            results = {
                "amortized_seq_flow": run_flow(
                    model, scenario, key, cfg, args.steps, initial
                ),
                "linear_mpc": run_linear_mpc_kf(
                    scenario, cfg, args.steps, args.noise, initial
                ),
                "nonlinear_mpc": run_nonlinear_mpc_kf(
                    scenario, cfg, args.steps, args.noise, initial
                ),
            }
            for name, result in results.items():
                row = metrics(name, args.noise, *result, cfg)
                row["initial_lateral_error"] = lateral
                row["initial_heading_error"] = heading
                rows.append(row)
                trajectories[(name, lateral, heading)] = result
            print(
                f"initial=({lateral:+.2f}, {heading:+.2f}) "
                + " ".join(
                    f"{name}={results[name][0].dot(results[name][0])**0.5 / args.steps**0.5:.3f}"
                    for name in METHODS
                )
            )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f"{args.name}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    all_rmse = [row["lateral_rmse"] for row in rows]
    vmax = max(all_rmse)
    for axis, method in zip(axes.flat[:3], METHODS):
        matrix = np.array([
            [next(row["lateral_rmse"] for row in rows
                  if row["method"] == method
                  and row["initial_lateral_error"] == lateral
                  and row["initial_heading_error"] == heading)
             for lateral in lateral_values]
            for heading in heading_values
        ])
        image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=vmax,
                            cmap="viridis")
        axis.set_xticks(range(len(lateral_values)), [f"{v:+.1f}" for v in lateral_values])
        axis.set_yticks(range(len(heading_values)), [f"{v:+.2f}" for v in heading_values])
        axis.set(xlabel="Initial lateral error [m]",
                 ylabel="Initial heading error [rad]", title=METHODS[method][0])
        for row_index in range(len(heading_values)):
            for column_index in range(len(lateral_values)):
                value = matrix[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.3f}", ha="center",
                          va="center", color="white" if value > 0.55 * vmax else "black")
    fig.colorbar(image, ax=axes.flat[:3].tolist(), label="Lateral RMSE [m]", shrink=0.8)

    axis = axes[1, 1]
    for method, (label, color) in METHODS.items():
        values = [row["lateral_rmse"] for row in rows if row["method"] == method]
        axis.boxplot(values, positions=[list(METHODS).index(method)], widths=0.55,
                     patch_artist=True, boxprops={"facecolor": color, "alpha": 0.75},
                     medianprops={"color": "black"})
    axis.set_xticks(range(len(METHODS)), [value[0] for value in METHODS.values()], rotation=12)
    axis.set(ylabel="Lateral RMSE [m]", title="Distribution over 15 initial conditions")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25, axis="y")
    fig.suptitle(
        f"Initial-condition robustness: nonlinear plant, noise std={args.noise:.2f} m"
    )
    figure_path = args.output / f"{args.name}.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    print(f"saved {csv_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
