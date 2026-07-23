"""Unified reevaluation of available flow variants and MPC baselines."""

import csv
from pathlib import Path

import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from .amortized_seq_flow import AmortizedSeqFlow
from .bicycle_dataset import BicycleConfig
from .evaluate_lane_keeping import (
    make_flow_chunk_evaluator, make_scenario, metrics, run_flow,
    run_linear_mpc_kf, run_nonlinear_mpc_kf,
)


OUTPUT = Path("results/results_optimal_control")
VARIANTS = {
    "flow_recovery_nmpc": (
        "Recovery/NMPC flow", "#d95f02",
        OUTPUT / "amortized_seq_flow_recovery_nmpc_checkpoint.eqx", 1.0,
    ),
    "flow_saturation_105": (
        "Saturation-aware flow", "#984ea3",
        OUTPUT / "runs/seed0_saturation_factor_105/lane_flow.eqx", 1.05,
    ),
}
BASELINES = {
    "linear_mpc": ("Linear MPC", "#377eb8"),
    "nonlinear_mpc": ("Nonlinear MPC", "#1b9e77"),
}


def load_models(cfg):
    models = {}
    for name, (_, _, checkpoint, factor) in VARIANTS.items():
        model = AmortizedSeqFlow(
            cfg.horizon, cfg.delta_max, jr.PRNGKey(0), saturation_factor=factor
        )
        with checkpoint.open("rb") as file:
            models[name] = eqx.tree_deserialise_leaves(file, model)
    return models


def save_rows(rows, path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def evaluate_noise(models, cfg):
    rows = []
    evaluators = {name: make_flow_chunk_evaluator() for name in models}
    initial = (0.55, 0.12)
    for index, noise in enumerate((0.0, cfg.obs_std, 2 * cfg.obs_std)):
        key = jr.fold_in(jr.PRNGKey(0), index)
        scenario = make_scenario(key, cfg, 300, noise)
        results = {
            name: run_flow(model, scenario, key, cfg, 300, initial,
                           evaluators[name])
            for name, model in models.items()
        }
        results["linear_mpc"] = run_linear_mpc_kf(scenario, cfg, 300, noise, initial)
        results["nonlinear_mpc"] = run_nonlinear_mpc_kf(scenario, cfg, 300, noise, initial)
        for name, result in results.items():
            rows.append(metrics(name, noise, *result, cfg))
    return rows


def evaluate_initial_conditions(models, cfg):
    rows, selected = [], {}
    evaluators = {name: make_flow_chunk_evaluator() for name in models}
    lateral = (-1.0, -0.5, 0.0, 0.5, 1.0)
    heading = (-0.2, 0.0, 0.2)
    key = jr.PRNGKey(0); scenario = make_scenario(key, cfg, 300, cfg.obs_std)
    selected_states = {(-1.0, -0.2), (0.0, 0.0), (1.0, 0.2)}
    for ep in heading:
        for ey in lateral:
            initial = (ey, ep)
            results = {
                name: run_flow(model, scenario, key, cfg, 300, initial,
                               evaluators[name])
                for name, model in models.items()
            }
            results["linear_mpc"] = run_linear_mpc_kf(
                scenario, cfg, 300, cfg.obs_std, initial
            )
            results["nonlinear_mpc"] = run_nonlinear_mpc_kf(
                scenario, cfg, 300, cfg.obs_std, initial
            )
            for name, result in results.items():
                row = metrics(name, cfg.obs_std, *result, cfg)
                row.update(initial_lateral_error=ey, initial_heading_error=ep)
                rows.append(row)
                if initial in selected_states:
                    selected[(name, initial)] = result
            print(f"completed initial state ({ey:+.1f}, {ep:+.1f})")
    return rows, selected


def style(name):
    return VARIANTS[name][:2] if name in VARIANTS else BASELINES[name]


def plot_noise(rows):
    specs = [
        ("lateral_rmse", "Lateral RMSE [m]", False),
        ("max_abs_lateral", "Maximum |lateral error| [m]", False),
        ("steering_rate_rms", "Steering-rate RMS [rad/step]", False),
        ("mean_runtime_ms", "Runtime [ms/command]", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (field, ylabel, log) in zip(axes.flat, specs):
        for name in (*VARIANTS, *BASELINES):
            group = [row for row in rows if row["method"] == name]
            label, color = style(name)
            axis.plot([row["noise_std"] for row in group],
                      [row[field] for row in group], "o-", lw=2,
                      label=label, color=color)
        axis.set(xlabel="Measurement noise std [m]", ylabel=ylabel)
        if log: axis.set_yscale("log")
        else: axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Flow variants and MPC: corrected closed-loop noise sweep")
    path = OUTPUT / "all_flow_variants_noise_sweep.png"
    fig.savefig(path, dpi=220); plt.close(fig)


def plot_initial(rows):
    names = [*VARIANTS, *BASELINES]
    lateral = sorted({row["initial_lateral_error"] for row in rows})
    heading = sorted({row["initial_heading_error"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    groups = [[row["lateral_rmse"] for row in rows if row["method"] == name]
              for name in names]
    boxes = axes[0].boxplot(groups, patch_artist=True,
                            medianprops={"color": "black"})
    for box, name in zip(boxes["boxes"], names): box.set_facecolor(style(name)[1])
    axes[0].set_xticks(range(1, len(names) + 1), [style(n)[0] for n in names], rotation=15)
    axes[0].set(ylabel="Lateral RMSE [m]", title="15-state RMSE distribution")
    axes[0].set_ylim(bottom=0); axes[0].grid(alpha=0.25, axis="y")
    for axis, variant in zip(axes[1:], VARIANTS):
        matrix = np.array([[next(row["lateral_rmse"] for row in rows
                                 if row["method"] == variant
                                 and row["initial_lateral_error"] == ey
                                 and row["initial_heading_error"] == ep)
                            - next(row["lateral_rmse"] for row in rows
                                   if row["method"] == "nonlinear_mpc"
                                   and row["initial_lateral_error"] == ey
                                   and row["initial_heading_error"] == ep)
                            for ey in lateral] for ep in heading])
        bound = max(abs(matrix.min()), abs(matrix.max()))
        image = axis.imshow(matrix, origin="lower", cmap="RdBu_r", aspect="auto",
                            vmin=-bound, vmax=bound)
        axis.set_xticks(range(5), [f"{v:+.1f}" for v in lateral])
        axis.set_yticks(range(3), [f"{v:+.1f}" for v in heading])
        axis.set(xlabel="Initial lateral error [m]", ylabel="Initial heading error [rad]",
                 title=f"{style(variant)[0]} − NMPC")
        for i in range(3):
            for j in range(5): axis.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center")
        fig.colorbar(image, ax=axis, label="RMSE difference [m]", shrink=0.8)
    fig.suptitle("Initial-condition robustness of both flow variants")
    path = OUTPUT / "all_flow_variants_initial_condition_performance.png"
    fig.savefig(path, dpi=220); plt.close(fig)


def plot_trajectories(selected, cfg):
    states = [(-1.0, -0.2), (0.0, 0.0), (1.0, 0.2)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    time = np.arange(300) * cfg.dt
    for axis, state in zip(axes, states):
        for name in (*VARIANTS, *BASELINES):
            label, color = style(name)
            axis.plot(time, selected[(name, state)][0], color=color, lw=1.6, label=label)
        axis.axhline(0, color="black", lw=0.8); axis.grid(alpha=0.25)
        axis.set(ylabel="Lateral error [m]",
                 title=rf"Initial $e_y={state[0]:+.1f}$ m, $e_\psi={state[1]:+.1f}$ rad")
    axes[-1].set_xlabel("Time [s]"); axes[0].legend(ncol=4, frameon=False)
    fig.suptitle("Representative trajectories for all controllers")
    path = OUTPUT / "all_flow_variants_initial_condition_trajectories.png"
    fig.savefig(path, dpi=220); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = BicycleConfig(); models = load_models(cfg)
    noise_rows = evaluate_noise(models, cfg)
    initial_rows, selected = evaluate_initial_conditions(models, cfg)
    save_rows(noise_rows, OUTPUT / "all_flow_variants_noise_sweep.csv")
    save_rows(initial_rows, OUTPUT / "all_flow_variants_initial_condition_sweep.csv")
    plot_noise(noise_rows); plot_initial(initial_rows); plot_trajectories(selected, cfg)
    print("saved unified CSVs and figures")


if __name__ == "__main__":
    main()
