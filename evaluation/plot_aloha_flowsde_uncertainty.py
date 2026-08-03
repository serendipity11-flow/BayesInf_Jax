#!/usr/bin/env python3
"""Plot FlowSDE uncertainty diagnostics for the three ALOHA datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = (
    ("Dataset noisy", "dataset_noisy", "realnvp_flow_sde_eval_metrics.npz", "gaussian"),
    (
        "Pseudo-process uncertainty",
        "pseudo_process_uncertainty",
        "realnvp_flow_sde_eval_metrics.npz",
        "gaussian",
    ),
    ("MuJoCo random", "mujoco_random", "realnvp_flow_sde_gmm_eval_metrics.npz", "gmm"),
)


def _subsample(values: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= size:
        return flat
    return flat[rng.choice(flat.size, size=size, replace=False)]


def _predictive_observations(
    samples: np.ndarray,
    obs_std: float,
    noise: str,
    rng: np.random.Generator,
    size: int = 300_000,
) -> np.ndarray:
    state = _subsample(samples, size, rng)
    if noise == "gmm":
        component = rng.choice(3, size=state.size, p=(0.70, 0.15, 0.15))
        offsets = obs_std * np.asarray((0.0, 2.5, -2.5))
        scales = obs_std * np.asarray((0.50, 1.50, 1.50))
        error = offsets[component] + scales[component] * rng.standard_normal(state.size)
    else:
        error = obs_std * rng.standard_normal(state.size)
    return state + error


def _draw_run(
    axes: np.ndarray,
    metrics_path: Path,
    label: str,
    noise: str,
    obs_std: float,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    with np.load(metrics_path) as data:
        truth = np.asarray(data["x_true"])
        observations = np.asarray(data["y_obs"])
        samples = np.asarray(data["samples"])
        mean = np.asarray(data["posterior_mean"])
        rmse = float(data["rmse"])
        coverage = float(data["coverage"])

    midpoint = truth.shape[1] // 2
    joint = 0
    particle_mid = samples[:, :, midpoint, joint].reshape(-1)
    truth_mid = truth[:, midpoint, joint]
    low, high = np.percentile(np.concatenate((particle_mid, truth_mid)), (0.5, 99.5))
    bins = np.linspace(low, high, 55)
    axes[0].hist(
        particle_mid, bins=bins, density=True, alpha=0.55,
        color="steelblue", label="Posterior particles",
    )
    axes[0].hist(
        truth_mid, bins=bins, density=True, histtype="step",
        linewidth=2, color="black", label="Ground truth",
    )
    axes[0].set(
        title=f"{label}\nParticles vs truth at midpoint",
        xlabel=f"Joint {joint} position",
        ylabel="Density",
    )
    axes[0].legend(frameon=False, fontsize=8)

    predicted_y = _predictive_observations(samples, obs_std, noise, rng)
    observed_y = _subsample(observations, 300_000, rng)
    low, high = np.percentile(np.concatenate((predicted_y, observed_y)), (0.5, 99.5))
    bins = np.linspace(low, high, 70)
    axes[1].hist(
        observed_y, bins=bins, density=True, alpha=0.55,
        color="steelblue", label="Test observations",
    )
    axes[1].hist(
        predicted_y, bins=bins, density=True, alpha=0.55,
        color="darkorange", label="Posterior predictive",
    )
    axes[1].set(
        title=f"Observation predictive ({noise.upper()})",
        xlabel="Observation value",
        ylabel="Density",
    )
    axes[1].legend(frameon=False, fontsize=8)

    time = np.arange(truth.shape[1])
    for particle in range(min(30, samples.shape[1])):
        axes[2].plot(
            time, samples[0, particle, :, joint],
            color=plt.cm.viridis(0.1 + 0.8 * particle / 30), alpha=0.3, linewidth=0.7,
        )
    axes[2].plot(time, truth[0, :, joint], color="black", linewidth=2, label="Truth")
    axes[2].plot(
        time, mean[0, :, joint], color="crimson", linestyle="--",
        linewidth=1.5, label="Posterior mean",
    )
    axes[2].set(
        title=f"Particle trajectories\nRMSE={rmse:.4f}, coverage={coverage * 100:.1f}%",
        xlabel="Time step",
        ylabel=f"Joint {joint} position",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    return {"rmse": rmse, "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/results_state_estimation/Aloha_sim_insertion"),
    )
    parser.add_argument("--obs-std", type=float, default=0.1)
    parser.add_argument(
        "--variant",
        default="",
        help="Optional method suffix inserted before _eval, e.g. uflow_obs1_ms5.",
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    runs = list(DEFAULT_RUNS)
    if args.variant:
        variant = args.variant.strip("_")
        runs = [
            (
                label,
                directory,
                filename.replace("_eval_metrics.npz", f"_{variant}_eval_metrics.npz"),
                noise,
            )
            for label, directory, filename, noise in runs
        ]
    for row, (label, directory, filename, noise) in enumerate(runs):
        metrics = args.root / directory / filename
        if not metrics.exists():
            raise FileNotFoundError(f"Missing evaluation metrics: {metrics}")
        _draw_run(axes[row], metrics, label, noise, args.obs_std, seed=42 + row)

        single, single_axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        _draw_run(single_axes, metrics, label, noise, args.obs_std, seed=42 + row)
        single.suptitle(f"RealNVP FlowSDE — {label} uncertainty quantification")
        single_path = metrics.with_name(metrics.stem.replace("_metrics", "_uncertainty") + ".png")
        single.savefig(single_path, dpi=250, bbox_inches="tight")
        plt.close(single)
        print(single_path)

    fig.suptitle(
        "RealNVP FlowSDE — ALOHA insertion uncertainty comparison",
        fontsize=16,
    )
    suffix = f"_{args.variant.strip('_')}" if args.variant else ""
    output = args.root / f"flow_sde{suffix}_uncertainty_comparison.png"
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
