#!/usr/bin/env python3
"""Merge a risk-aware FlowSDE run into the canonical all-episode comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    ("final_normalized_rmse", "Final normalized RMSE"),
    ("cumulative_normalized_error", "Cumulative normalized error"),
    ("control_smoothness", "Control smoothness (RMS Δu)"),
)
COLORS = {"exact_mujoco": "tab:blue", "flow_sde": "tab:orange"}


def exact_rows(canonical: Path) -> pd.DataFrame:
    # Support both the original two-row header and this script's long format.
    flat = pd.read_csv(canonical)
    if {"row_type", "controller"}.issubset(flat.columns):
        return flat[(flat.row_type == "episode") & (flat.controller == "exact_mujoco")].copy()
    wide = pd.read_csv(canonical, header=[0, 1])
    rows = pd.DataFrame({
        "row_type": "episode",
        "task": wide.iloc[:, 0].astype(int),
        "episode_id": wide.iloc[:, 1].astype(int),
        "controller": "exact_mujoco",
        "risk_weight": np.nan,
        "particles": np.nan,
    })
    for metric in ("control_rms", "control_smoothness", "cumulative_normalized_error",
                   "final_normalized_rmse", "trajectory_normalized_rmse"):
        rows[metric] = wide[(metric, "exact_mujoco")].to_numpy()
    rows["mean_runtime_ms"] = np.nan
    return rows


def add_summary_rows(episodes: pd.DataFrame, exact_summary: Path) -> pd.DataFrame:
    statistics = {
        "mean": np.mean, "std": lambda x: np.std(x, ddof=0), "median": np.median,
        "q25": lambda x: np.quantile(x, .25), "q75": lambda x: np.quantile(x, .75),
        "min": np.min, "max": np.max,
    }
    rows = []
    numeric = ["control_rms", "control_smoothness", "cumulative_normalized_error",
               "final_normalized_rmse", "trajectory_normalized_rmse", "mean_runtime_ms"]
    runtime = pd.read_csv(exact_summary)
    exact_runtime = runtime[runtime.model == "exact_mujoco"].iloc[0]
    for controller, group in episodes.groupby("controller"):
        for name, fn in statistics.items():
            row = {"row_type": f"summary_{name}", "task": np.nan, "episode_id": np.nan,
                   "controller": controller, "risk_weight": group.risk_weight.iloc[0],
                   "particles": group.particles.iloc[0]}
            for metric in numeric:
                values = group[metric].dropna().to_numpy()
                row[metric] = fn(values) if len(values) else np.nan
            if controller == "exact_mujoco" and name in ("mean", "std"):
                row["mean_runtime_ms"] = exact_runtime[
                    "mean_runtime_ms" if name == "mean" else "mean_runtime_ms_std"
                ]
            rows.append(row)
    return pd.concat([episodes, pd.DataFrame(rows)], ignore_index=True)


def distribution(ax, exact, flow, metric, label, rng):
    values = [exact[metric].to_numpy(), flow[metric].to_numpy()]
    violins = ax.violinplot(values, positions=[0, 1], widths=.75,
                            showmeans=False, showmedians=False, showextrema=False)
    for body, controller in zip(violins["bodies"], COLORS):
        body.set_facecolor(COLORS[controller]); body.set_edgecolor(COLORS[controller]); body.set_alpha(.22)
    boxes = ax.boxplot(values, positions=[0, 1], widths=.25, patch_artist=True,
                       showfliers=False, medianprops={"color": "black", "lw": 1.5})
    for box, controller in zip(boxes["boxes"], COLORS):
        box.set_facecolor(COLORS[controller]); box.set_alpha(.7)
    for position, controller, sample in zip([0, 1], COLORS, values):
        ax.scatter(position + rng.uniform(-.09, .09, len(sample)), sample, s=9,
                   alpha=.35, color=COLORS[controller], edgecolors="none")
    ax.set(xticks=[0, 1], xticklabels=["exact_mujoco", "flow_sde"], ylabel=label, title=label)
    ax.grid(axis="y", alpha=.25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    canonical = args.root / "all_episode_performance.csv"
    risk_dir = args.root / "particle_risk_sweep" / "risk1p0_particles8"
    flow_case = pd.read_csv(risk_dir / "case_metrics.csv")
    if len(flow_case) != 80:
        raise ValueError(f"risk-aware FlowSDE run has {len(flow_case)} rows, expected 80")

    exact = exact_rows(canonical)
    flow = flow_case.rename(columns={"model": "controller"}).copy()
    flow.insert(0, "row_type", "episode")
    flow["risk_weight"], flow["particles"] = 1.0, 8
    flow["trajectory_normalized_rmse"] = np.sqrt(flow.cumulative_normalized_error / 10.0)
    actions = []
    for task in flow.task.astype(int):
        with np.load(risk_dir / f"h8_task{task}_flow_sde.npz") as run:
            actions.append(np.sqrt(np.mean(run["actions"] ** 2)))
    flow["control_rms"] = actions
    columns = ["row_type", "task", "episode_id", "controller", "risk_weight", "particles",
               "control_rms", "control_smoothness", "cumulative_normalized_error",
               "final_normalized_rmse", "trajectory_normalized_rmse", "mean_runtime_ms"]
    episodes = pd.concat([exact[columns], flow[columns]], ignore_index=True)
    merged = add_summary_rows(episodes, args.root / "mppi_comparison_summary.csv")
    merged.to_csv(canonical, index=False)

    exact_ep = episodes[episodes.controller == "exact_mujoco"]
    flow_ep = episodes[episodes.controller == "flow_sde"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (metric, label) in zip(axes.flat[:3], METRICS):
        distribution(ax, exact_ep, flow_ep, metric, label, rng)
    ax = axes.flat[3]
    summary = merged[merged.row_type.isin(["summary_mean", "summary_std"])]
    means = summary[summary.row_type == "summary_mean"].set_index("controller")
    stds = summary[summary.row_type == "summary_std"].set_index("controller")
    controllers = ["exact_mujoco", "flow_sde"]
    ax.bar([0, 1], means.loc[controllers, "mean_runtime_ms"],
           yerr=stds.loc[controllers, "mean_runtime_ms"], capsize=4,
           color=[COLORS[c] for c in controllers], alpha=.8)
    ax.set(xticks=[0, 1], xticklabels=controllers, ylabel="Mean inference time [ms]",
           title="Inference time (independent scale)")
    ax.grid(axis="y", alpha=.25)
    fig.suptitle("MPPI across 80 episodes: exact MuJoCo vs FlowSDE (risk=1, particles=8)")
    fig.savefig(args.root / "all_episode_performance.png", dpi=180)
    plt.close(fig)
    print(canonical)


if __name__ == "__main__":
    main()
