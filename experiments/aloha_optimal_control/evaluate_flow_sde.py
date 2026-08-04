"""Evaluate ALOHA controlled FlowSDE trajectories, accuracy, and uncertainty."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from .train_flow_sde import build_model
from .data import (
    augment_with_causal_velocity, load_episodes, split_episode_indices,
)


PUSHER_STATE_NAMES = (
    [f"q_{index}" for index in range(1, 7)]
    + [f"qdot_{index}" for index in range(1, 7)]
)
PUSHER_PLOT_LABELS = (
    [rf"$q_{index}$" for index in range(1, 7)]
    + [rf"$\dot{{q}}_{index}$" for index in range(1, 7)]
)


def evaluate(args):
    with args.config.open() as handle:
        config = json.load(handle)
    checkpoint = Path(config["checkpoint"])
    normalization_path = Path(config["normalization"])
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    position_dim = state_dim
    if config.get("velocity_augmented", False):
        episodes = augment_with_causal_velocity(
            episodes, dt=config["dt"], window=config["velocity_window"]
        )
        state_dim = 2 * position_dim
    elif config.get("state_is_position_velocity", False):
        if state_dim % 2:
            raise ValueError("existing [q,v] state dimension must be even")
        position_dim = state_dim // 2
    _, _, test_i = split_episode_indices(len(episodes), config["seed"])
    stats = np.load(normalization_path)
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    u_mean, u_std = stats["u_mean"], stats["u_std"]
    model = build_model(SimpleNamespace(**config), state_dim, control_dim)
    model = eqx.tree_deserialise_leaves(checkpoint, model)
    deterministic = eqx.filter_jit(model.deterministic_rollout)
    stochastic = eqx.filter_jit(model.sample_rollout)
    args.output.mkdir(parents=True, exist_ok=True)

    state_names = (
        PUSHER_STATE_NAMES if position_dim == 12
        else [f"state_{index}" for index in range(position_dim)]
    )
    plot_labels = (
        PUSHER_PLOT_LABELS if position_dim == 12 else state_names
    )
    truths, nominals, means = [], [], []
    controls_by_episode, samples_by_episode, episode_rows = [], [], []
    for order, index in enumerate(test_i):
        x = jnp.asarray((episodes[index] - x_mean) / x_std)
        u = jnp.asarray((actions[index] - u_mean) / u_std)
        if args.recondition_every <= 0:
            nominal_n = deterministic(x[0], u)
            samples_n = stochastic(
                x[0], u, jr.PRNGKey(args.eval_seed + order), args.particles,
                diffusion_temperature=args.diffusion_temperature,
            )
        else:
            nominal_parts, sample_parts = [], []
            for segment, start in enumerate(
                    range(0, len(u), args.recondition_every)):
                stop = min(start + args.recondition_every, len(u))
                # Ground truth is used only by this diagnostic, never by the
                # deployable open-loop model.
                nominal_parts.append(np.asarray(deterministic(
                    x[start], u[start:stop], start_step=start
                )))
                sample_parts.append(np.asarray(stochastic(
                    x[start], u[start:stop],
                    jr.PRNGKey(args.eval_seed + order * 1000 + segment),
                    args.particles, start_step=start,
                    diffusion_temperature=args.diffusion_temperature,
                )))
            nominal_n = jnp.asarray(np.concatenate(nominal_parts, axis=0))
            samples_n = jnp.asarray(np.concatenate(sample_parts, axis=1))
        # Evaluation is in the original 14 measured joint-position channels.
        truth = episodes[index][1:, :position_dim]
        nominal = (np.asarray(nominal_n)[..., :position_dim]
                   * x_std[:position_dim] + x_mean[:position_dim])
        samples = (np.asarray(samples_n)[..., :position_dim]
                   * x_std[None, None, :position_dim]
                   + x_mean[None, None, :position_dim])
        mean = samples.mean(0)
        truths.append(truth); nominals.append(nominal); means.append(mean)
        controls_by_episode.append(np.asarray(actions[index]))
        samples_by_episode.append(samples)
        episode_rows.append({
            "episode_id": int(episode_ids[index]),
            "nominal_rmse": float(np.sqrt(np.mean((nominal - truth) ** 2))),
            "particle_mean_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        })
        print(f"evaluated episode={episode_ids[index]} steps={len(truth)}", flush=True)

    truth = np.concatenate(truths, axis=0)
    nominal = np.concatenate(nominals, axis=0)
    mean = np.concatenate(means, axis=0)
    samples = np.concatenate(samples_by_episode, axis=1)
    error = mean - truth
    per_joint_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    per_joint_mae = np.mean(np.abs(error), axis=0)
    levels, calibration = (0.5, 0.8, 0.9, 0.95), []
    for level in levels:
        alpha = (1 - level) / 2
        lower, upper = np.quantile(samples, [alpha, 1 - alpha], axis=0)
        coverage = np.mean((truth >= lower) & (truth <= upper))
        calibration.append({
            "nominal_coverage": level, "empirical_coverage": float(coverage),
            "coverage_error": float(coverage - level),
            "mean_interval_width": float(np.mean(upper - lower)),
        })
    horizon_rmse = np.sqrt(np.mean(np.stack([
        (prediction - target) ** 2 for prediction, target in zip(means, truths)
    ]), axis=(0, 2)))
    metrics = {
        "model": ("SecondOrderControlledRealNVPFlowSDE"
                  if (config.get("velocity_augmented", False)
                      or config.get("state_is_position_velocity", False))
                  else "ControlledRealNVPFlowSDE"),
        "dt": config["dt"],
        "velocity_augmented": config.get("velocity_augmented", False),
        "state_is_position_velocity": config.get(
            "state_is_position_velocity", False
        ),
        "particles": args.particles,
        "diffusion_temperature": args.diffusion_temperature,
        "recondition_every": args.recondition_every,
        "test_episode_ids": [int(episode_ids[i]) for i in test_i],
        "nominal_rmse": float(np.sqrt(np.mean((nominal - truth) ** 2))),
        "particle_mean_rmse": float(np.sqrt(np.mean(error ** 2))),
        "particle_mean_mae": float(np.mean(np.abs(error))),
        "aggregate_r2": float(1 - np.sum(error ** 2) /
                              np.maximum(np.sum((truth - truth.mean(0)) ** 2), 1e-12)),
        "mean_predictive_std": float(np.mean(np.std(samples, axis=0))),
        "per_joint": {name: {"rmse": float(per_joint_rmse[k]),
                              "mae": float(per_joint_mae[k])}
                      for k, name in enumerate(state_names)},
        "calibration": calibration, "episodes": episode_rows,
    }
    with (args.output / "flow_sde_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.output / "flow_sde_horizon_accuracy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["step", "seconds", "rmse"])
        writer.writerows((k + 1, (k + 1) * config["dt"], float(value))
                         for k, value in enumerate(horizon_rmse))

    example_samples = samples_by_episode[0]
    q05, q95 = np.quantile(example_samples, [0.05, 0.95], axis=0)
    seconds = np.arange(len(truths[0])) * config["dt"]
    control_dim = controls_by_episode[0].shape[-1]
    rows = int(np.ceil((position_dim + control_dim) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(15, max(5, 2.6 * rows)),
                             sharex=True, squeeze=False)
    if position_dim == 12:
        state_axes = [
            axes[joint, component]
            for joint in range(6)
            for component in range(2)
        ]
        state_indices = [
            index
            for joint in range(6)
            for index in (joint, joint + 6)
        ]
    else:
        state_axes = list(axes.flat[:position_dim])
        state_indices = list(range(position_dim))
    for axis, k in zip(state_axes, state_indices):
        axis.fill_between(seconds, q05[:, k], q95[:, k], alpha=0.2, color="C0")
        axis.plot(seconds, means[0][:, k], color="C0", lw=1, label="particle mean")
        axis.plot(seconds, truths[0][:, k], color="black", lw=1, label="truth")
        axis.set_title(plot_labels[k]); axis.grid(alpha=0.2)
    example_controls = controls_by_episode[0]
    for k in range(control_dim):
        axis = axes.flat[position_dim + k]
        axis.plot(
            seconds, example_controls[:, k], color="C2", lw=1.1,
            label=rf"$u_{k + 1}$",
        )
        axis.set_title(rf"Control $u_{k + 1}$")
        axis.grid(alpha=0.2)
    for axis in axes.flat[position_dim + control_dim:]:
        axis.set_visible(False)
    axes[0, 0].legend(ncol=3)
    fig.suptitle(f"Controlled FlowSDE — held-out episode {episode_ids[test_i[0]]}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.output / "flow_sde_trajectory_uncertainty.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(seconds, horizon_rmse); axes[0].set(title="Error accumulation", xlabel="horizon [s]", ylabel="RMSE")
    axes[1].barh(plot_labels, per_joint_rmse)
    axes[1].set(title="Accuracy by state", xlabel="RMSE")
    empirical = [row["empirical_coverage"] for row in calibration]
    axes[2].plot([0, 1], [0, 1], "k--", label="ideal")
    axes[2].plot(levels, empirical, "o-", label="FlowSDE")
    axes[2].set(title="Uncertainty calibration", xlabel="nominal coverage", ylabel="empirical coverage", xlim=(0.45, 1), ylim=(0.45, 1))
    axes[2].legend()
    for axis in axes: axis.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(args.output / "flow_sde_accuracy_calibration.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/aloha_mobile_cabinet/flow_sde/evaluation"))
    p.add_argument("--particles", type=int, default=64)
    p.add_argument("--eval-seed", type=int, default=2026)
    p.add_argument("--diffusion-temperature", type=float, default=1.0)
    p.add_argument("--recondition-every", type=int, default=0,
                   help="Reset particles to truth every K steps; 0 is open loop.")
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
