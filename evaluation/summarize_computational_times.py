#!/usr/bin/env python3
"""Create a denominator-explicit timing summary across application categories."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows: list[dict] = []

    parameter = pd.read_csv(ROOT / "parameter_identification_linear_vs_multimodal_metrics.csv")
    parameter = parameter[parameter.parameter == "mean"]
    for record in parameter.itertuples():
        rows.append({
            "category": "parameter_identification",
            "dataset": record.dataset,
            "method": record.model,
            "latency_ms": record.inference_time_ms,
            "timing_unit": "ms/test_case",
            "cases_per_timed_call": 200,
            "particles_per_case": 100,
            "horizon": "",
            "notes": "JIT-warmed median full-batch sampling time / 200 cases; each case returns 100 posterior samples",
        })

    state = pd.read_csv(ROOT / "state_estimation_inference_times.csv")
    for record in state.itertuples():
        rows.append({
            "category": "state_estimation",
            "dataset": "lorenz96_N12",
            "method": record.method,
            "latency_ms": record.latency_ms_per_trajectory,
            "timing_unit": "ms/input_trajectory",
            "cases_per_timed_call": 8,
            "particles_per_case": record.posterior_samples,
            "horizon": record.horizon,
            "notes": "JIT-warmed synchronized median batch time / 8 input trajectories",
        })

    open_loop = pd.read_csv(ROOT / "open_loop_flowsde_inference_times.csv").iloc[0]
    rows.append({
        "category": "open_loop_dynamics",
        "dataset": "aloha_mujoco",
        "method": open_loop.method,
        "latency_ms": open_loop.median_ms_per_case,
        "timing_unit": "ms/control_sequence",
        "cases_per_timed_call": int(open_loop.batch_size),
        "particles_per_case": int(open_loop.particles_per_case),
        "horizon": int(open_loop.horizon),
        "notes": (
            "JIT-warmed synchronized rollout; per particle trajectory = "
            f"{open_loop.median_ms_per_particle_trajectory:.6f} ms"
        ),
    })

    closed_loop_path = (
        ROOT.parent / "results/results_optimal_control/aloha_mujoco/flow_sde_small/"
        "sweep_horizon_summary.csv"
    )
    closed = pd.read_csv(closed_loop_path)
    for record in closed.itertuples():
        rows.append({
            "category": "closed_loop_mppi",
            "dataset": "aloha_mujoco",
            "method": record.model,
            "latency_ms": record.mean_runtime_ms,
            "timing_unit": "ms/MPC_command",
            "cases_per_timed_call": 1,
            "particles_per_case": (
                int(record.flow_particles) if pd.notna(record.flow_particles) else ""
            ),
            "horizon": int(record.horizon),
            "notes": (
                f"mean over {int(record.episodes)} episodes; first JIT command excluded; "
                f"between-episode std={record.mean_runtime_ms_std:.6f} ms"
            ),
        })

    output = ROOT / "computational_time_summary.csv"
    fields = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    categories = (
        ("parameter_identification", "Parameter identification"),
        ("state_estimation", "State estimation"),
        ("open_loop_dynamics", "Open-loop dynamics"),
        ("closed_loop_mppi", "Closed-loop MPPI"),
    )
    for axis, (category, title) in zip(axes.flat, categories):
        selected = frame[frame.category == category].copy()
        if category == "parameter_identification":
            selected["label"] = selected["dataset"] + " · " + selected["method"]
        elif category == "closed_loop_mppi":
            selected["label"] = selected["method"] + " · H" + selected["horizon"].astype(str)
        else:
            selected["label"] = selected["method"]
        selected = selected.sort_values("latency_ms")
        axis.barh(selected.label, selected.latency_ms, color="#4C78A8")
        axis.set_xscale("log")
        axis.set(title=title, xlabel=selected.timing_unit.iloc[0] + " (log scale)")
        axis.grid(axis="x", alpha=0.25)
    figure_path = ROOT / "computational_time_summary.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(figure_path)


if __name__ == "__main__":
    main()
