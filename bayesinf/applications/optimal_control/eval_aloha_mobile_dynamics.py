"""Open-loop accuracy and uncertainty evaluation for trained ALOHA dynamics."""

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from .gru_latent_ssm_flow import GRULatentSSMFlow
from .train_aloha_mobile_dynamics import load_episodes, split_episode_indices


JOINT_NAMES = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]


def load_model(args, state_dim, control_dim):
    model = GRULatentSSMFlow(
        state_dim, control_dim, args.history, args.hidden_dim, jr.PRNGKey(args.seed + 1),
        predict_delta=True, n_coupling=args.blocks, hidden_size=args.width,
        n_hidden=args.hidden_layers, flow_type=args.flow_type,
        num_bins=args.num_bins, max_rho=args.max_rho,
    )
    return eqx.tree_deserialise_leaves(args.checkpoint, model)


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    _, _, test_indices = split_episode_indices(len(episodes), args.seed)
    stats = np.load(args.normalization)
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    u_mean, u_std = stats["u_mean"], stats["u_std"]
    model = load_model(args, state_dim, control_dim)
    deterministic_fn = eqx.filter_jit(model.rollout_deterministic)
    sample_fn = eqx.filter_jit(model.sample_rollout)

    all_truth, all_nominal, all_mean, all_samples = [], [], [], []
    episode_results = []
    for order, index in enumerate(test_indices):
        x = jnp.asarray((episodes[index] - x_mean) / x_std)
        u = jnp.asarray((actions[index] - u_mean) / u_std)
        controls = u[args.history - 1:]
        nominal = np.asarray(deterministic_fn(x[:args.history], controls)) * x_std + x_mean
        samples = np.asarray(sample_fn(
            x[:args.history], u[:args.history - 1], controls,
            jr.PRNGKey(args.eval_seed + order), args.particles
        )) * x_std[None, None, :] + x_mean[None, None, :]
        truth = episodes[index][args.history:]
        mean = samples.mean(axis=0)
        all_truth.append(truth); all_nominal.append(nominal)
        all_mean.append(mean); all_samples.append(samples)
        episode_results.append({
            "episode_id": int(episode_ids[index]),
            "nominal_rmse": float(np.sqrt(np.mean((nominal - truth) ** 2))),
            "particle_mean_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
            "particle_mean_mae": float(np.mean(np.abs(mean - truth))),
        })
        print(f"evaluated episode={episode_ids[index]} horizon={len(truth)}", flush=True)

    truth = np.concatenate(all_truth)
    nominal = np.concatenate(all_nominal)
    mean = np.concatenate(all_mean)
    samples = np.concatenate(all_samples, axis=1)
    error = mean - truth
    nominal_error = nominal - truth
    per_joint_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    per_joint_mae = np.mean(np.abs(error), axis=0)
    ss_res = np.sum(error ** 2, axis=0)
    ss_tot = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    per_joint_r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    levels = [0.5, 0.8, 0.9, 0.95]
    calibration = []
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lower, upper = np.quantile(samples, [alpha, 1.0 - alpha], axis=0)
        covered = (truth >= lower) & (truth <= upper)
        calibration.append({
            "nominal_coverage": level,
            "empirical_coverage": float(covered.mean()),
            "mean_interval_width": float((upper - lower).mean()),
            "coverage_error": float(covered.mean() - level),
        })

    # Horizon curves average like-for-like times across all held-out episodes.
    horizon_rmse = np.sqrt(np.mean(np.stack([
        (prediction - target) ** 2 for prediction, target in zip(all_mean, all_truth)
    ]), axis=(0, 2)))
    q05, q95 = np.quantile(all_samples[0], [0.05, 0.95], axis=0)
    example_truth, example_mean = all_truth[0], all_mean[0]

    metrics = {
        "test_episode_ids": [int(episode_ids[i]) for i in test_indices],
        "particles": args.particles, "horizon_steps": int(len(example_truth)),
        "horizon_seconds": float(len(example_truth) / 50.0),
        "nominal_rmse": float(np.sqrt(np.mean(nominal_error ** 2))),
        "particle_mean_rmse": float(np.sqrt(np.mean(error ** 2))),
        "particle_mean_mae": float(np.mean(np.abs(error))),
        "aggregate_r2": float(1.0 - np.sum(error ** 2) /
                              np.sum((truth - truth.mean(axis=0)) ** 2)),
        "per_joint": {name: {"rmse": float(per_joint_rmse[i]),
                              "mae": float(per_joint_mae[i]),
                              "r2": float(per_joint_r2[i])}
                      for i, name in enumerate(JOINT_NAMES)},
        "calibration": calibration,
        "episodes": episode_results,
    }
    with (args.output / "open_loop_uq_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.output / "open_loop_horizon_accuracy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["horizon_step", "seconds", "rmse"])
        writer.writerows((i + 1, (i + 1) / 50.0, value)
                         for i, value in enumerate(horizon_rmse))
    np.savez_compressed(
        args.output / "open_loop_example_episode.npz", truth=example_truth,
        particle_mean=example_mean, nominal=all_nominal[0], q05=q05, q95=q95,
        samples=all_samples[0], episode_id=int(episode_ids[test_indices[0]]),
    )

    time = np.arange(len(example_truth)) / 50.0
    fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
    for i, axis in enumerate(axes.flat):
        axis.fill_between(time, q05[:, i], q95[:, i], color="C0", alpha=0.2,
                          label="90% interval" if i == 0 else None)
        axis.plot(time, example_mean[:, i], color="C0", lw=1,
                  label="particle mean" if i == 0 else None)
        axis.plot(time, all_nominal[0][:, i], color="C1", lw=1, ls="--",
                  label="zero-latent nominal" if i == 0 else None)
        axis.plot(time, example_truth[:, i], color="black", lw=1,
                  label="truth" if i == 0 else None)
        axis.set_title(JOINT_NAMES[i]); axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("open-loop time [s]"); axes[-1, 1].set_xlabel("open-loop time [s]")
    fig.legend(loc="upper center", ncol=4); fig.suptitle(
        f"ALOHA full-horizon stochastic rollout — held-out episode {episode_ids[test_indices[0]]}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(
        args.output / "open_loop_trajectory_uncertainty.png", dpi=180
    ); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(np.arange(len(horizon_rmse)) / 50.0, horizon_rmse)
    axes[0].set(xlabel="horizon [s]", ylabel="RMSE", title="Error accumulation")
    axes[1].barh(JOINT_NAMES, per_joint_rmse)
    axes[1].set(xlabel="RMSE", title="Particle-mean accuracy by joint")
    nominal_levels = np.asarray(levels)
    empirical = np.asarray([row["empirical_coverage"] for row in calibration])
    axes[2].plot([0, 1], [0, 1], "k--", label="ideal")
    axes[2].plot(nominal_levels, empirical, "o-", label="model")
    axes[2].set(xlim=(0.45, 1), ylim=(0.45, 1), xlabel="nominal coverage",
                ylabel="empirical coverage", title="Interval calibration")
    axes[2].legend()
    for axis in axes: axis.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(args.output / "open_loop_accuracy_calibration.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path("results/results_optimal_control/aloha_mobile_cabinet")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=base / "aloha_realnvp_noise0p05_h8_seg16_seed0.eqx")
    p.add_argument("--normalization", type=Path,
                   default=base / "aloha_realnvp_noise0p05_h8_seg16_seed0_normalization.npz")
    p.add_argument("--output", type=Path, default=base / "evaluation_realnvp_noise0p05")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=2026)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--particles", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--hidden-layers", type=int, default=1)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--flow-type", choices=("realnvp", "nsf"), default="realnvp")
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--max-rho", type=float, default=0.8)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
