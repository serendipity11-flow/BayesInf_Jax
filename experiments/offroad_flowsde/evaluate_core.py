"""Evaluate and plot a 14-D-core off-road FlowSDE checkpoint."""

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
from .structured_core import reconstruct_next


NAMES = ("x", "y", "psi", "v_long", "v_lat", "yaw_rate", "wheel_fl", "wheel_fr",
         "wheel_rl", "wheel_rr", "z", "roll", "pitch", "v_z", "roll_rate", "pitch_rate")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.result / "config.json").read_text())
    model = FlowSDE(16, 2, 14, jr.PRNGKey(cfg["seed"]), dt=cfg["dt"], width=cfg["width"],
                    depth=cfg["depth"], condition_dim=cfg["condition_dim"],
                    flow_layers=cfg["flow_layers"])
    model = eqx.tree_deserialise_leaves(args.result / "flowsde_core.eqx", model)
    stats = dict(np.load(args.result / "normalization.npz"))
    with np.load(args.data) as data:
        x, u, xn, episode = (data[k] for k in
            ("observation.state", "action", "next_observation.state", "episode_index"))

    mask = np.isin(episode, cfg["test_episodes"])
    xh, uh, xnh = x[mask], u[mask], xn[mask]
    xs = (xh - stats["context_mean"]) / stats["context_std"]
    us = (uh - stats["action_mean"]) / stats["action_std"]
    core_ds = ((xnh[:, 2:] - xh[:, 2:]) - stats["core_delta_mean"]) / stats["core_delta_std"]
    mean_core_dn = np.asarray(jax.jit(jax.vmap(model.mean_delta))(
        jnp.asarray(xs), jnp.asarray(us)))
    predicted_core = xh[:, 2:] + mean_core_dn * stats["core_delta_std"] + stats["core_delta_mean"]
    predicted_next = np.asarray(reconstruct_next(jnp.asarray(xh), jnp.asarray(predicted_core), cfg["dt"]))
    error = predicted_next - xnh
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    nll = float(-np.mean(np.asarray(jax.jit(jax.vmap(model.log_prob))(
        jnp.asarray(core_ds), jnp.asarray(xs), jnp.asarray(us)))))

    rng = np.random.default_rng(args.seed)
    ci = rng.choice(len(xh), min(args.calibration_samples, len(xh)), replace=False)
    keys = jr.split(jr.PRNGKey(args.seed), len(ci) * args.particles).reshape(
        len(ci), args.particles, 2)
    sample = jax.jit(jax.vmap(jax.vmap(
        lambda s, a, k: model.step(s, a, k, True), in_axes=(None, None, 0)),
        in_axes=(0, 0, 0)))
    sampled_dn = np.asarray(sample(jnp.asarray(xs[ci]), jnp.asarray(us[ci]), keys))
    sampled_core = (xh[ci, None, 2:] + sampled_dn * stats["core_delta_std"]
                    + stats["core_delta_mean"])
    sampled_next = np.asarray(jax.vmap(jax.vmap(
        lambda current, core: reconstruct_next(current, core, cfg["dt"]),
        in_axes=(None, 0)), in_axes=(0, 0))(jnp.asarray(xh[ci]), jnp.asarray(sampled_core)))
    lo, hi = np.quantile(sampled_next, (.05, .95), axis=1)
    coverage = np.mean((xnh[ci] >= lo) & (xnh[ci] <= hi), axis=0)
    widths = np.mean(hi - lo, axis=0)
    energy = float(np.mean(np.linalg.norm(sampled_next - xnh[ci, None], axis=-1))
                   - .5 * np.mean(np.linalg.norm(
                       sampled_next - np.roll(sampled_next, 1, axis=1), axis=-1)))

    ids = np.asarray(cfg["test_episodes"])
    horizon = min(args.horizon, min(np.sum(episode == e) for e in ids))
    truth, controls = [], []
    for e in ids:
        idx = np.flatnonzero(episode == e)[:horizon]
        truth.append(np.concatenate((x[idx[:1]], xn[idx]), axis=0))
        controls.append(u[idx])
    truth, controls = np.stack(truth), np.stack(controls)
    cm, cs, dm, ds = (jnp.asarray(stats[k]) for k in
                       ("context_mean", "context_std", "core_delta_mean", "core_delta_std"))
    am, ast = jnp.asarray(stats["action_mean"]), jnp.asarray(stats["action_std"])

    def rollout(initial, commands, rollout_keys):
        def step(current, item):
            command, key = item
            sn, un = (current - cm) / cs, (command - am) / ast
            delta_n = model.mean_delta(sn, un) if rollout_keys is None else model.step(sn, un, key, True)
            next_core = current[2:] + delta_n * ds + dm
            nxt = reconstruct_next(current, next_core, cfg["dt"])
            return nxt, nxt
        scan_keys = jnp.zeros((len(commands), 2), dtype=jnp.uint32) if rollout_keys is None else rollout_keys
        return jax.lax.scan(step, initial, (commands, scan_keys))[1]

    mean = np.asarray(jax.jit(jax.vmap(rollout, in_axes=(0, 0, None)))(
        jnp.asarray(truth[:, 0]), jnp.asarray(controls), None))
    mean = np.concatenate((truth[:, :1], mean), axis=1)
    horizon_rmse = np.sqrt(np.mean((mean - truth) ** 2, axis=(0, 2)))
    xy_error = np.linalg.norm(mean[..., :2] - truth[..., :2], axis=-1)
    count = min(args.plot_episodes, len(ids))
    pkeys = jr.split(jr.PRNGKey(args.seed + 1), count * args.plot_particles * horizon).reshape(
        count, args.plot_particles, horizon, 2)
    particle_fn = jax.jit(jax.vmap(jax.vmap(rollout, in_axes=(None, None, 0)),
                                    in_axes=(0, 0, 0)))
    particles = np.asarray(particle_fn(jnp.asarray(truth[:count, 0]),
                                       jnp.asarray(controls[:count]), pkeys))
    particles = np.concatenate((np.broadcast_to(truth[:count, None, :1],
                               (count, args.plot_particles, 1, 16)), particles), axis=2)
    divergence = np.max(np.linalg.norm(particles[..., :2], axis=-1), axis=-1) > 50

    metrics = {
        "test_episodes": int(len(ids)), "test_transitions": int(len(xh)),
        "test_normalized_nll": nll,
        "one_step_aggregate_rmse": float(np.sqrt(np.mean(error ** 2))),
        "one_step_energy_score": energy, "mean_90pct_coverage": float(np.mean(coverage)),
        "per_state_rmse": dict(zip(NAMES, map(float, rmse))),
        "per_state_90pct_coverage": dict(zip(NAMES, map(float, coverage))),
        "per_state_90pct_width": dict(zip(NAMES, map(float, widths))),
        "recursive_horizon_steps": int(horizon),
        "recursive_final_aggregate_rmse": float(horizon_rmse[-1]),
        "recursive_final_xy_error_mean": float(np.mean(xy_error[:, -1])),
        "recursive_final_xy_error_median": float(np.median(xy_error[:, -1])),
        "recursive_particle_divergence_fraction_50m": float(np.mean(divergence)),
        "horizon_aggregate_rmse": {str(h): float(horizon_rmse[h])
                                     for h in (10, 25, 50, 100) if h <= horizon},
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output / "per_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("state", "rmse", "coverage_90", "width_90"))
        writer.writerows(zip(NAMES, rmse, coverage, widths))
    np.savez_compressed(args.output / "trajectory_data.npz", truth=truth, recursive_mean=mean,
                        recursive_particles=particles, controls=controls,
                        horizon_rmse=horizon_rmse, test_episode_ids=ids)

    history = json.loads((args.result / "history.json").read_text())
    epochs = np.asarray([row["epoch"] for row in history])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [row["train_nll"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_parts"][0] for row in history], label="validation")
    axes[0].set(title="Core transition NLL", xlabel="epoch"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [row["train_total"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_total"] for row in history], label="validation")
    axes[1].set(title="Weighted objective", xlabel="epoch"); axes[1].legend(); axes[1].grid(alpha=.25)
    axes[2].bar(np.arange(16) - .2, rmse, .4, label="RMSE")
    twin = axes[2].twinx(); twin.bar(np.arange(16) + .2, coverage, .4, color="tab:green", label="90% coverage")
    twin.axhline(.9, color="k", ls="--", lw=1); axes[2].set_xticks(range(16), NAMES, rotation=55, ha="right")
    axes[2].set_title("Held-out one-step metrics"); axes[2].legend(loc="upper left"); twin.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(args.output / "performance_metrics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), squeeze=False)
    for i, ax in enumerate(axes[0]):
        visible = np.concatenate((truth[i, :, :2], mean[i, :, :2])); low, high = visible.min(0), visible.max(0)
        margin = np.maximum(.5, .08 * (high - low))
        for particle in particles[i]: ax.plot(particle[:, 0], particle[:, 1], color="tab:blue", alpha=.08, lw=.6)
        ax.plot(truth[i, :, 0], truth[i, :, 1], "k", lw=2, label="MuJoCo")
        ax.plot(mean[i, :, 0], mean[i, :, 1], color="tab:orange", lw=2, label="FlowSDE mean")
        ax.set(title=f"held-out episode {ids[i]}", xlabel="x [m]", ylabel="y [m]",
               xlim=(low[0]-margin[0], high[0]+margin[0]), ylim=(low[1]-margin[1], high[1]+margin[1]))
        ax.set_aspect("equal", adjustable="datalim"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output / "trajectories.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(np.arange(horizon + 1) * cfg["dt"], horizon_rmse)
    ax.set(xlabel="horizon [s]", ylabel="aggregate state RMSE", title="Recursive mean rollout error")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(args.output / "horizon_rmse.png", dpi=180); plt.close(fig)
    print(json.dumps(metrics, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/offroad_flowsde/transitions_core.npz"))
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--horizon", type=int, default=100); p.add_argument("--particles", type=int, default=64)
    p.add_argument("--calibration-samples", type=int, default=5000); p.add_argument("--plot-episodes", type=int, default=3)
    p.add_argument("--plot-particles", type=int, default=32); p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(); args.output = args.output or args.result / "evaluation"; return args


if __name__ == "__main__":
    run(parser())
