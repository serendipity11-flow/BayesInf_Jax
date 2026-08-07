"""Evaluate and plot a trained off-road FlowSDE on held-out terrain maps."""

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

from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE

NAMES = ("x", "y", "sin_yaw", "cos_yaw", "v_long", "v_lat", "yaw_rate", "wheel_slip")


def load_model(result_dir):
    config = json.loads((result_dir / "config.json").read_text())
    model = FlowSDE(8, 2, 8, jr.PRNGKey(config["seed"]), dt=config["dt"],
                    width=config["width"], depth=config["depth"],
                    condition_dim=config["condition_dim"], flow_layers=config["flow_layers"])
    model = eqx.tree_deserialise_leaves(result_dir / "flowsde.eqx", model)
    stats = dict(np.load(result_dir / "normalization.npz"))
    return model, config, stats


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model, config, stats = load_model(args.result)
    with np.load(args.data) as bundle:
        state, action = bundle["observation.state"], bundle["action"]
        next_state, episodes = bundle["next_observation.state"], bundle["episode_index"]
    mask = np.isin(episodes, config["test_episodes"])
    s, a, ns = state[mask], action[mask], next_state[mask]
    sn = (s - stats["state_mean"]) / stats["state_std"]
    an = (a - stats["action_mean"]) / stats["action_std"]
    dn = ((ns - s) - stats["delta_mean"]) / stats["delta_std"]
    predict = jax.jit(jax.vmap(model.mean_delta))
    likelihood = jax.jit(jax.vmap(model.log_prob))
    predicted_dn = np.asarray(predict(jnp.asarray(sn), jnp.asarray(an)))
    predicted_delta = predicted_dn * stats["delta_std"] + stats["delta_mean"]
    predicted_next = s + predicted_delta
    error = predicted_next - ns
    per_state_rmse = np.sqrt(np.mean(error**2, axis=0))
    test_nll = float(-np.mean(np.asarray(likelihood(jnp.asarray(dn), jnp.asarray(sn), jnp.asarray(an)))))

    # One-step stochastic calibration on a fixed held-out subset.
    rng = np.random.default_rng(args.seed)
    cal_idx = rng.choice(len(s), min(args.calibration_samples, len(s)), replace=False)
    keys = jr.split(jr.PRNGKey(args.seed), len(cal_idx) * args.particles).reshape(len(cal_idx), args.particles, 2)
    sample_one = lambda x, u, key: model.step(x, u, key, stochastic=True)
    sample_many = jax.jit(jax.vmap(jax.vmap(sample_one, in_axes=(None, None, 0)), in_axes=(0, 0, 0)))
    sampled_dn = np.asarray(sample_many(jnp.asarray(sn[cal_idx]), jnp.asarray(an[cal_idx]), keys))
    sampled_next = s[cal_idx, None] + sampled_dn * stats["delta_std"] + stats["delta_mean"]
    lo, hi = np.quantile(sampled_next, (0.05, 0.95), axis=1)
    coverage = np.mean((ns[cal_idx] >= lo) & (ns[cal_idx] <= hi), axis=0)
    interval_width = np.mean(hi - lo, axis=0)
    sample_error = np.linalg.norm(sampled_next - ns[cal_idx, None], axis=-1)
    pair_index = np.roll(np.arange(args.particles), 1)
    pair_distance = np.linalg.norm(sampled_next - sampled_next[:, pair_index], axis=-1)
    energy_score = float(np.mean(sample_error) - 0.5 * np.mean(pair_distance))

    # Recursive prediction from the first state of every held-out episode.
    test_ids = np.asarray(config["test_episodes"])
    horizon = min(args.horizon, min(np.sum(episodes == episode) for episode in test_ids))
    truth, controls = [], []
    for episode in test_ids:
        idx = np.flatnonzero(episodes == episode)[:horizon]
        truth.append(np.concatenate((state[idx[:1]], next_state[idx]), axis=0))
        controls.append(action[idx])
    truth, controls = np.stack(truth), np.stack(controls)

    def mean_rollout(initial, commands):
        def step(current, command):
            x = (current - stats["state_mean"]) / stats["state_std"]
            u = (command - stats["action_mean"]) / stats["action_std"]
            delta_n = model.mean_delta(x, u)
            nxt = current + delta_n * stats["delta_std"] + stats["delta_mean"]
            return nxt, nxt
        return jax.lax.scan(step, initial, commands)[1]

    recursive = np.asarray(jax.jit(jax.vmap(mean_rollout))(jnp.asarray(truth[:, 0]), jnp.asarray(controls)))
    recursive = np.concatenate((truth[:, :1], recursive), axis=1)
    horizon_rmse = np.sqrt(np.mean((recursive - truth)**2, axis=(0, 2)))
    xy_error = np.linalg.norm(recursive[..., :2] - truth[..., :2], axis=-1)

    metrics = {
        "checkpoint_selection": "best validation NLL",
        "test_transitions": int(len(s)), "test_episodes": int(len(test_ids)),
        "test_normalized_nll": test_nll,
        "one_step_aggregate_rmse": float(np.sqrt(np.mean(error**2))),
        "one_step_energy_score": energy_score,
        "mean_90pct_coverage": float(np.mean(coverage)),
        "recursive_horizon_steps": int(horizon),
        "recursive_final_aggregate_rmse": float(horizon_rmse[-1]),
        "recursive_final_xy_error_mean": float(np.mean(xy_error[:, -1])),
        "recursive_final_xy_error_median": float(np.median(xy_error[:, -1])),
        "per_state_one_step_rmse": dict(zip(NAMES, map(float, per_state_rmse))),
        "per_state_90pct_coverage": dict(zip(NAMES, map(float, coverage))),
        "per_state_90pct_width": dict(zip(NAMES, map(float, interval_width))),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output / "per_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("state", "rmse", "coverage_90", "width_90"))
        writer.writerows(zip(NAMES, per_state_rmse, coverage, interval_width))
    def stochastic_rollout(initial, commands, keys):
        def step(current, inputs):
            command, key = inputs
            x = (current - stats["state_mean"]) / stats["state_std"]
            u = (command - stats["action_mean"]) / stats["action_std"]
            delta_n = model.step(x, u, key, stochastic=True)
            nxt = current + delta_n * stats["delta_std"] + stats["delta_mean"]
            return nxt, nxt
        return jax.lax.scan(step, initial, (commands, keys))[1]

    count = min(args.plot_episodes, len(test_ids))
    trajectory_keys = jr.split(jr.PRNGKey(args.seed + 1), count * args.plot_particles * horizon)
    trajectory_keys = trajectory_keys.reshape(count, args.plot_particles, horizon, 2)
    particle_rollout = jax.jit(jax.vmap(jax.vmap(stochastic_rollout, in_axes=(None, None, 0)),
                                        in_axes=(0, 0, 0)))
    particles = np.asarray(particle_rollout(jnp.asarray(truth[:count, 0]),
                                            jnp.asarray(controls[:count]), trajectory_keys))
    particles = np.concatenate((np.broadcast_to(truth[:count, None, :1],
                                                 (count, args.plot_particles, 1, 8)), particles), axis=2)
    particle_final_xy_error = np.linalg.norm(particles[..., -1, :2] - truth[:count, None, -1, :2], axis=-1)
    metrics["recursive_particle_divergence_fraction_50m"] = float(np.mean(
        np.max(np.linalg.norm(particles[..., :2], axis=-1), axis=-1) > 50.0))
    metrics["recursive_particle_final_xy_error_median"] = float(np.median(particle_final_xy_error))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(args.output / "trajectory_data.npz", truth=truth, recursive_mean=recursive,
                        recursive_particles=particles, controls=controls, horizon_rmse=horizon_rmse,
                        test_episode_ids=test_ids)

    history = json.loads((args.result / "history.json").read_text())
    epoch = np.asarray([row["epoch"] for row in history])
    train_nll = np.asarray([row["train_nll"] for row in history])
    val_nll = np.asarray([row["validation_nll"] for row in history])
    val_rmse = np.asarray([row["validation_normalized_rmse"] for row in history])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epoch, train_nll, alpha=.75, label="train"); axes[0].plot(epoch, val_nll, label="validation")
    axes[0].set_ylim(np.percentile(np.r_[train_nll, val_nll], [2, 95])); axes[0].set_title("Transition NLL (robust limits)")
    axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epoch, val_rmse, color="tab:orange"); axes[1].set_title("Validation normalized RMSE"); axes[1].set_xlabel("epoch")
    axes[2].bar(np.arange(8) - .2, per_state_rmse, .4, label="RMSE")
    ax2 = axes[2].twinx(); ax2.bar(np.arange(8) + .2, coverage, .4, color="tab:green", label="90% coverage")
    axes[2].set_xticks(range(8), NAMES, rotation=45, ha="right"); axes[2].set_title("Held-out one-step performance")
    axes[2].legend(loc="upper left"); ax2.legend(loc="upper right"); ax2.axhline(.9, color="k", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(args.output / "performance_metrics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), squeeze=False)
    for i, ax in enumerate(axes[0]):
        visible = np.concatenate((truth[i, :, :2], recursive[i, :, :2]), axis=0)
        low, high = visible.min(0), visible.max(0); margin = np.maximum(.5, .08 * (high - low))
        ax.plot(truth[i, :, 0], truth[i, :, 1], color="k", lw=2, label="MuJoCo")
        for particle in particles[i]:
            ax.plot(particle[:, 0], particle[:, 1], color="tab:blue", alpha=.10, lw=.7)
        ax.plot(recursive[i, :, 0], recursive[i, :, 1], color="tab:orange", lw=2, label="FlowSDE mean")
        ax.scatter(truth[i, 0, 0], truth[i, 0, 1], marker="o", color="tab:green")
        ax.set_title(f"held-out terrain episode {test_ids[i]}"); ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(low[0] - margin[0], high[0] + margin[0]); ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend()
    fig.tight_layout(); fig.savefig(args.output / "trajectories.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(np.arange(horizon + 1) * config["dt"], horizon_rmse)
    ax.set(xlabel="recursive horizon [s]", ylabel="aggregate state RMSE", title="Held-out recursive rollout error")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(args.output / "horizon_rmse.png", dpi=180); plt.close(fig)
    print(json.dumps(metrics, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/offroad_flowsde/transitions.npz"))
    p.add_argument("--result", type=Path, default=Path("results/offroad_flowsde/seed0"))
    p.add_argument("--output", type=Path, default=Path("results/offroad_flowsde/seed0/evaluation"))
    p.add_argument("--horizon", type=int, default=100); p.add_argument("--particles", type=int, default=64)
    p.add_argument("--calibration-samples", type=int, default=5000)
    p.add_argument("--plot-episodes", type=int, default=3); p.add_argument("--plot-particles", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
