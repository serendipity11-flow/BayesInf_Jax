"""Aggregate and visualize the five-seed lane-control experiment."""
import argparse
import csv
import re
from pathlib import Path

import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control.amortized_seq_flow import AmortizedSeqFlow
from bayesinf.applications.optimal_control.bicycle_dataset import BicycleConfig
from bayesinf.applications.optimal_control.evaluate_lane_keeping import (
    make_scenario, run_flow, run_mpc_kf,
)


METHOD_LABELS = {"amortized_seq_flow": "AmortizedSeqFlow", "mpc_kalman": "MPC + Kalman filter"}
COLORS = {"amortized_seq_flow": "#d95f02", "mpc_kalman": "#1b9e77"}


def load_rows(root):
    rows = []
    for path in sorted(root.glob("seed*/comparison.csv")):
        seed = int(path.parent.name.removeprefix("seed"))
        with path.open() as f:
            for row in csv.DictReader(f):
                row = {k: (v if k == "method" else float(v)) for k, v in row.items()}
                row["seed"] = seed
                rows.append(row)
    return rows


def aggregate(rows):
    fields = ["lateral_rmse", "max_abs_lateral", "steering_rms", "steering_rate_rms",
              "max_abs_delta_deg", "constraint_violations", "mean_runtime_ms", "solver_failures"]
    summary = []
    for method in METHOD_LABELS:
        for noise in sorted({r["noise_std"] for r in rows}):
            group = [r for r in rows if r["method"] == method and r["noise_std"] == noise]
            item = {"method": method, "noise_std": noise, "n_seeds": len(group)}
            for field in fields:
                values = np.array([r[field] for r in group])
                item[field + "_mean"] = values.mean()
                item[field + "_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            summary.append(item)
    return summary


def save_summary(summary, path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)


def metric_figure(summary, path):
    specs = [("lateral_rmse", "Lateral RMSE [m]", True),
             ("steering_rate_rms", "Steering-rate RMS [rad/step]", False),
             ("mean_runtime_ms", "Runtime per command [ms]", True),
             ("max_abs_delta_deg", "Maximum |steering| [deg]", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (field, ylabel, log_scale) in zip(axes.flat, specs):
        for method in METHOD_LABELS:
            group = [r for r in summary if r["method"] == method]
            x = np.array([r["noise_std"] for r in group])
            y = np.array([r[field + "_mean"] for r in group])
            e = np.array([r[field + "_std"] for r in group])
            ax.errorbar(x, y, yerr=e, marker="o", capsize=4, linewidth=2,
                        color=COLORS[method], label=METHOD_LABELS[method])
        if log_scale: ax.set_yscale("log")
        ax.set_xlabel("Measurement noise std [m]"); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Lane keeping: five-seed closed-loop comparison")
    fig.savefig(path, dpi=220); plt.close(fig)


def training_figure(log_dir, path):
    pattern = re.compile(r"epoch=(\d+) train_nll=([-+\d.]+) val_nll=([-+\d.]+)")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for logfile in sorted(log_dir.glob("lane_flow_mpc_13473_*.out")):
        values = [tuple(map(float, m.groups())) for m in pattern.finditer(logfile.read_text())]
        if not values: continue
        data = np.asarray(values)
        seed = logfile.stem.rsplit("_", 1)[-1]
        ax.plot(data[:, 0], data[:, 2], label=f"seed {seed}", alpha=.85)
    ax.set(xlabel="Epoch", ylabel="Validation NLL [nat/action]",
           title="AmortizedSeqFlow validation objective")
    ax.grid(alpha=.25); ax.legend(ncol=5, frameon=False)
    fig.savefig(path, dpi=220); plt.close(fig)


def rollout_figure(root, path, seed=2, noise=0.12, steps=300):
    cfg = BicycleConfig(); key = jr.fold_in(jr.PRNGKey(seed), 1)
    skeleton = AmortizedSeqFlow(cfg.horizon, cfg.delta_max, jr.PRNGKey(0))
    with (root / f"seed{seed}" / "lane_flow.eqx").open("rb") as f:
        model = eqx.tree_deserialise_leaves(f, skeleton)
    scenario = make_scenario(key, cfg, steps, noise)
    flow = run_flow(model, scenario, key, cfg, steps)
    mpc = run_mpc_kf(scenario, cfg, steps, noise)
    t = np.arange(steps) * cfg.dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    for name, result in (("amortized_seq_flow", flow), ("mpc_kalman", mpc)):
        axes[0].plot(t, result[0], color=COLORS[name], label=METHOD_LABELS[name])
        axes[1].plot(t, np.rad2deg(result[1]), color=COLORS[name], label=METHOD_LABELS[name])
    axes[0].axhline(0, color="black", lw=.8); axes[0].set_ylabel("Lateral error [m]")
    axes[1].axhline(np.rad2deg(cfg.delta_max), color="black", ls="--", lw=.8)
    axes[1].axhline(-np.rad2deg(cfg.delta_max), color="black", ls="--", lw=.8)
    axes[1].set(xlabel="Time [s]", ylabel="Steering [deg]")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(frameon=False); fig.suptitle(f"Representative rollout: seed {seed}, noise std={noise:.2f} m")
    fig.savefig(path, dpi=220); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path("results/results_optimal_control/runs"))
    p.add_argument("--logs", type=Path,
                   default=Path("results/results_optimal_control"))
    args = p.parse_args()
    rows = load_rows(args.root)
    if not rows: raise FileNotFoundError(f"No seed*/comparison.csv files below {args.root}")
    summary = aggregate(rows)
    save_summary(summary, args.root / "aggregate_summary.csv")
    metric_figure(summary, args.root / "comparison_metrics.png")
    training_figure(args.logs, args.root / "training_curves.png")
    rollout_figure(args.root, args.root / "rollout_seed2.png")
    print(f"Wrote aggregate CSV and three figures to {args.root}")


if __name__ == "__main__":
    main()
