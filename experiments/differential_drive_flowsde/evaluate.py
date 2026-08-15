"""Evaluate a trained differential-drive FlowSDE and create diagnostic plots."""
import argparse, json
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE
from .train import angle_delta
from .train_multishoot import causal_contexts
from .env import DifferentialDriveEnv

DEFAULT_NAMES = ("p_x", "p_y", "theta", "v", "omega")


def load_model(result):
    config = json.loads((result / "config.json").read_text())
    output_dim = config.get("output_dim", config.get("state_dim", 5))
    model = FlowSDE(config.get("input_dim", 5), 2, output_dim, jr.PRNGKey(config["seed"]), dt=config["dt"],
                    width=config["width"], depth=config["depth"],
                    condition_dim=config["condition_dim"], flow_layers=config["flow_layers"],
                    flow_scale_limit=config.get("flow_scale_limit", 2.0))
    return eqx.tree_deserialise_leaves(result / "flowsde.eqx", model), config


def project_orientation(x, names):
    if "theta" in names:
        i = names.index("theta")
        x[..., i] = np.arctan2(np.sin(x[..., i]), np.cos(x[..., i]))
    elif "cos_theta" in names:
        ci, si = names.index("cos_theta"), names.index("sin_theta")
        norm = np.maximum(np.hypot(x[..., ci], x[..., si]), 1e-6)
        x[..., ci], x[..., si] = x[..., ci] / norm, x[..., si] / norm
    return x


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model, config = load_model(args.result)
    stats = dict(np.load(args.result / "normalization.npz"))
    with h5py.File(args.data) as f:
        x, u, xn, split, episode, friction = (f[k][:] for k in
            ("state", "action", "next_state", "split", "trajectory_id", "friction"))
        names = tuple(json.loads(f.attrs.get("state_order", json.dumps(DEFAULT_NAMES))))
    state_dim = x.shape[1]
    structured = bool(config.get("structured_kinematics", False))
    dynamic_indices = np.asarray(config.get("dynamic_indices", range(state_dim)))

    def reconstruct(states, dynamic_delta):
        nxt = states.copy(); nxt[..., dynamic_indices] += dynamic_delta
        if structured:
            compact = "cos_theta" in names
            vf_i, vl_i, omega_i = (5, 6, 8) if compact else (3, 4, 5)
            vf, vl = .5 * (states[..., vf_i] + nxt[..., vf_i]), .5 * (states[..., vl_i] + nxt[..., vl_i])
            omega = .5 * (states[..., omega_i] + nxt[..., omega_i])
            theta = (np.arctan2(states[..., 4], states[..., 3]) if compact
                     else states[..., 2]) + config["dt"] * omega
            c, s = np.cos(theta), np.sin(theta)
            nxt[..., 0] = states[..., 0] + config["dt"] * (vf * c - vl * s)
            nxt[..., 1] = states[..., 1] + config["dt"] * (vf * s + vl * c)
            if compact: nxt[..., 3], nxt[..., 4] = c, s
            else: nxt[..., 2] = np.arctan2(s, c)
        return project_orientation(nxt, names)
    test = np.flatnonzero(split == 2)
    angle_index = names.index("theta") if "theta" in names else None
    delta = angle_delta(xn.copy(), x, angle_index)
    if "feature_mean" in stats:
        raw_features = (np.column_stack((x[:, :2], np.sin(x[:, 2]), np.cos(x[:, 2]), x[:, 3:]))
                        if angle_index == 2 else x.copy())
        xs = (raw_features - stats["feature_mean"]) / stats["feature_std"]
        def normalize_state(state):
            raw = (np.column_stack((state[:, :2], np.sin(state[:, 2]), np.cos(state[:, 2]), state[:, 3:]))
                   if angle_index == 2 else state)
            return (raw - stats["feature_mean"]) / stats["feature_std"]
    else:
        xs = (x - stats["state_mean"]) / stats["state_std"]
        def normalize_state(state): return (state - stats["state_mean"]) / stats["state_std"]
    us = (u - stats["action_mean"]) / stats["action_std"]
    target_delta = delta[:, dynamic_indices]
    ds = (target_delta - stats["delta_mean"]) / stats["delta_std"]
    history_steps = config.get("history_steps", 1)
    zero_action = (np.zeros(2, np.float32) - stats["action_mean"]) / stats["action_std"]
    raw_state_context, action_context = causal_contexts(x, us, episode, history_steps, zero_action)
    normalized_state_context = normalize_state(raw_state_context.reshape(-1, state_dim)).reshape(
        len(x), history_steps, -1)
    xs = np.concatenate((normalized_state_context.reshape(len(x), -1),
                         action_context.reshape(len(x), -1)), axis=-1)

    @eqx.filter_jit
    def mean_batch(candidate, bx, bu): return jax.vmap(candidate.mean_delta)(bx, bu)
    @eqx.filter_jit
    def log_prob_batch(candidate, bd, bx, bu): return jax.vmap(candidate.log_prob)(bd, bx, bu)
    predicted_ds = np.asarray(mean_batch(model, jnp.asarray(xs[test]), jnp.asarray(us[test])))
    predicted_dynamic_delta = predicted_ds * stats["delta_std"] + stats["delta_mean"]
    predicted_next = reconstruct(x[test], predicted_dynamic_delta)
    predicted_delta = predicted_next - x[test]
    residual = xn[test] - predicted_next
    full_delta_scale = stats.get("full_delta_std", stats["delta_std"])
    normalized_residual = residual / full_delta_scale
    log_prob_normalized = np.asarray(log_prob_batch(model, jnp.asarray(ds[test]), jnp.asarray(xs[test]), jnp.asarray(us[test])))
    log_prob_physical = log_prob_normalized - np.log(stats["delta_std"]).sum()

    q1, q2 = np.quantile(friction[test], (1 / 3, 2 / 3))
    surface_masks = {"low traction": friction[test] <= q1,
                     "medium traction": (friction[test] > q1) & (friction[test] <= q2),
                     "high traction": friction[test] > q2}
    metrics = {"test_transitions": int(len(test)),
               "mean_nll_physical": float(-log_prob_physical.mean()),
               "rmse": dict(zip(names, np.sqrt(np.mean(residual ** 2, axis=0)).tolist())),
               "normalized_rmse": dict(zip(names, np.sqrt(np.mean(normalized_residual ** 2, axis=0)).tolist())),
               "mae": dict(zip(names, np.mean(np.abs(residual), axis=0).tolist()))}
    metrics["terrain"] = {}
    for name, mask in surface_masks.items():
        metrics["terrain"][name] = {"count": int(mask.sum()),
            "mean_nll_physical": float(-log_prob_physical[mask].mean()) if mask.any() else None,
            "rmse": dict(zip(names, np.sqrt(np.mean(residual[mask] ** 2, axis=0)).tolist())) if mask.any() else {}}

    # Empirical one-step predictive intervals on a fixed subset.
    rng = np.random.default_rng(args.seed)
    calibration_rows = rng.choice(test, min(args.calibration_transitions, len(test)), replace=False)
    keys = jr.split(jr.PRNGKey(args.seed), args.particles)
    # A distinct key is required per transition; fold the row into every particle key.
    @eqx.filter_jit
    def sampled_deltas(candidate, bx, bu, base_keys):
        def particle(key):
            row_keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(bx.shape[0]))
            return jax.vmap(candidate.step)(bx, bu, row_keys)
        return jax.vmap(particle)(base_keys)
    sample_ds = np.asarray(sampled_deltas(model, jnp.asarray(xs[calibration_rows]),
                                          jnp.asarray(us[calibration_rows]), keys))
    sample_dynamic_delta = sample_ds * stats["delta_std"] + stats["delta_mean"]
    sample_next = reconstruct(np.broadcast_to(x[calibration_rows], sample_dynamic_delta.shape[:-1] + (state_dim,)).copy(),
                              sample_dynamic_delta)
    lower, upper = np.quantile(sample_next, (.05, .95), axis=0)
    coverage = np.mean((xn[calibration_rows] >= lower) & (xn[calibration_rows] <= upper), axis=0)
    metrics["coverage_90"] = dict(zip(names, coverage.tolist()))

    # Roll out complete held-out trajectory segments with common recorded commands.
    test_episodes = np.unique(episode[test])[:args.rollout_episodes]
    rollout_records = []
    for ep in test_episodes:
        rows = np.flatnonzero(episode == ep)[:args.horizon]
        if len(rows) < args.horizon: continue
        actions = us[rows]
        initial = x[rows[0]]
        particle_states = np.repeat(initial[None], args.particles, axis=0)
        particle_history = np.repeat(raw_state_context[rows[0]][None], args.particles, axis=0)
        particle_actions = np.repeat(action_context[rows[0]][None], args.particles, axis=0)
        paths = np.empty((args.particles, len(rows) + 1, state_dim), np.float32); paths[:, 0] = initial
        for k, action in enumerate(actions):
            normalized_history = normalize_state(particle_history.reshape(-1, state_dim)).reshape(
                args.particles, history_steps, -1)
            bx = np.concatenate((normalized_history.reshape(args.particles, -1),
                                 particle_actions.reshape(args.particles, -1)), axis=-1)
            bu = np.repeat(action[None], args.particles, axis=0)
            step_keys = jr.split(jr.fold_in(jr.PRNGKey(args.seed + int(ep)), k), args.particles)
            sampled = np.asarray(jax.vmap(model.step)(jnp.asarray(bx), jnp.asarray(bu), step_keys))
            particle_states = reconstruct(particle_states,
                sampled * stats["delta_std"] + stats["delta_mean"])
            particle_history = np.concatenate((particle_history[:, 1:], particle_states[:, None]), axis=1)
            if history_steps > 1:
                particle_actions = np.concatenate((particle_actions[:, 1:], bu[:, None]), axis=1)
            paths[:, k + 1] = particle_states
        truth = np.concatenate((x[rows[:1]], xn[rows]), axis=0)
        position_error = paths[:, :, :2] - truth[None, :, :2]
        rollout_records.append((int(ep), truth, paths, np.sqrt(np.sum(position_error ** 2, axis=-1))))
    all_position_errors = np.concatenate([record[3][:, 1:].reshape(-1) for record in rollout_records])
    mean_position_errors = np.concatenate([np.linalg.norm(record[2].mean(0)[1:, :2] -
                                                           record[1][1:, :2], axis=-1)
                                            for record in rollout_records])
    metrics["open_loop"] = {"episodes": len(rollout_records), "horizon": args.horizon,
                           "mean_position_rmse_m": float(np.sqrt(np.mean(mean_position_errors ** 2))),
                           "mean_position_mae_m": float(np.mean(mean_position_errors)),
                           "particle_position_rmse_m": float(np.sqrt(np.mean(all_position_errors ** 2))),
                           "particle_position_mae_m": float(np.mean(all_position_errors))}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    history = json.loads((args.result / "history.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    drift = [r for r in history if r["stage"] == "drift"]; joint = [r for r in history if r["stage"] == "joint"]
    axes[0, 0].plot([r["stage_epoch"] for r in drift], [r["train_drift_mse"] for r in drift], label="train")
    axes[0, 0].plot([r["stage_epoch"] for r in drift], [r["validation_drift_mse"] for r in drift], label="validation")
    axes[0, 0].set(title="Stage 1: drift", xlabel="epoch", ylabel="normalized MSE"); axes[0, 0].legend()
    axes[0, 1].plot([r["stage_epoch"] for r in joint], [r["train_nll"] for r in joint], label="train")
    axes[0, 1].plot([r["stage_epoch"] for r in joint], [r["validation_nll"] for r in joint], label="validation")
    axes[0, 1].set(title="Stage 2: diffusion likelihood", xlabel="epoch", ylabel="NLL"); axes[0, 1].legend()
    normalized_rmse = metrics["normalized_rmse"]
    axes[1, 0].bar(names, [normalized_rmse[n] for n in names]); axes[1, 0].tick_params(axis="x", rotation=30); axes[1, 0].set(title="Test one-step normalized RMSE", ylabel="RMSE / training delta scale")
    axes[1, 1].bar(names, coverage); axes[1, 1].tick_params(axis="x", rotation=30); axes[1, 1].axhline(.9, color="k", ls="--", label="nominal")
    axes[1, 1].set(title="90% predictive interval coverage", ylim=(0, 1), ylabel="coverage"); axes[1, 1].legend()
    figure.tight_layout(); figure.savefig(args.output / "training_and_one_step.png", dpi=180); figure.savefig(args.output / "training_and_one_step.pdf"); plt.close(figure)

    # Select the most terrain-diverse example rather than the trajectory with
    # largest x, which need not demonstrate the intended test condition.
    terrain_env = DifferentialDriveEnv(seed=args.seed)
    ep, truth, paths, _ = max(rollout_records, key=lambda record:
        np.ptp([terrain_env.friction_at(px, py) for px, py in record[1][:, :2]]))
    time = np.arange(len(truth)) * config["dt"]
    mean, lo, hi = paths.mean(0), np.quantile(paths, .05, axis=0), np.quantile(paths, .95, axis=0)
    columns = 3; rows_count = int(np.ceil((state_dim + 1) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(15, 2.65 * rows_count)); axes = np.asarray(axes).reshape(-1)
    title_size, label_size, tick_size, legend_size = 19, 17, 15, 14
    ax = axes[0]
    for path in paths[:min(20, args.particles)]: ax.plot(path[:, 0], path[:, 1], color="C0", alpha=.1)
    ax.plot(mean[:, 0], mean[:, 1], color="C0", label="FlowSDE mean"); ax.plot(truth[:, 0], truth[:, 1], "k--", label="MuJoCo")
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 80), np.linspace(ymin, ymax, 80))
    traction = np.vectorize(terrain_env.friction_at)(gx, gy)
    ax.contourf(gx, gy, traction, levels=10, cmap="YlGn", alpha=.18, zorder=-1)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(f"Open-loop test, episode {ep}", fontsize=title_size)
    ax.set_xlabel("x [m]", fontsize=label_size); ax.set_ylabel("y [m]", fontsize=label_size)
    ax.tick_params(labelsize=tick_size); ax.legend(fontsize=legend_size)
    for dim, axis in enumerate(axes[1:state_dim + 1]):
        axis.fill_between(time, lo[:, dim], hi[:, dim], alpha=.25, label="90% interval")
        axis.plot(time, mean[:, dim], label="mean"); axis.plot(time, truth[:, dim], "k--", label="truth")
        axis.set_title(names[dim], fontsize=title_size)
        axis.set_xlabel("time [s]", fontsize=label_size); axis.set_ylabel("native units", fontsize=label_size)
        axis.tick_params(labelsize=tick_size)
    for axis in axes[state_dim + 1:]: axis.set_visible(False)
    axes.flat[1].legend(fontsize=legend_size); figure.tight_layout(); figure.savefig(args.output / "rollout_example.png", dpi=220, bbox_inches="tight"); figure.savefig(args.output / "rollout_example.pdf", bbox_inches="tight"); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = list(surface_masks); xloc = np.arange(state_dim); width = .25
    for offset, surface in enumerate(labels):
        values = [metrics["terrain"][surface]["rmse"][name] for name in names]
        axes[0].bar(xloc + (offset - (len(labels) - 1) / 2) * width,
                    values, width, label=surface)
    axes[0].set_xticks(xloc, names, rotation=30); axes[0].set(title="One-step RMSE by traction quantile", ylabel="native units"); axes[0].legend()
    axes[1].bar(labels, [metrics["terrain"][s]["mean_nll_physical"] for s in labels])
    axes[1].set(title="Physical transition NLL by traction quantile", ylabel="NLL")
    figure.tight_layout(); figure.savefig(args.output / "terrain_comparison.png", dpi=180); figure.savefig(args.output / "terrain_comparison.pdf"); plt.close(figure)
    print(json.dumps(metrics, indent=2)); return metrics


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, default=Path("data/differential_drive/smooth_terrain_navigation.h5"))
    p.add_argument("--result", type=Path, default=Path("results/results_optimal_control/differential_drive/smooth_terrain_seed0/joint_stage"))
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/differential_drive/smooth_terrain_seed0/evaluation"))
    p.add_argument("--particles", type=int, default=100); p.add_argument("--calibration-transitions", type=int, default=2000)
    p.add_argument("--rollout-episodes", type=int, default=10); p.add_argument("--horizon", type=int, default=100); p.add_argument("--seed", type=int, default=17)
    return p


if __name__ == "__main__": run(parser().parse_args())
