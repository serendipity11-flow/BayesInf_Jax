"""Optimized two-stage FlowSDE training with multiple shooting and path energy score."""
import argparse, json, time
from pathlib import Path
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE
from .train import angle_delta


def robust_center_scale(values):
    center = np.median(values, axis=0)
    # 1--99% score scale equals sigma for Gaussian data but resists rare slip outliers.
    scale = (np.quantile(values, .99, axis=0) - np.quantile(values, .01, axis=0)) / 4.6527
    fallback = np.std(values, axis=0)
    return center.astype(np.float32), np.maximum(scale, np.maximum(fallback * .1, 1e-5)).astype(np.float32)


def windows(episodes, selected, length, stride):
    result = []
    for episode in selected:
        rows = np.flatnonzero(episodes == episode)
        result.extend(rows[start:start + length] for start in range(0, len(rows) - length + 1, stride))
    return np.stack(result)


def terrain_stratified_sample(all_windows, friction, count, bins, rng):
    """Uniformly sample smooth-terrain friction quantiles at window midpoints."""
    if bins < 1:
        raise ValueError("--terrain-bins must be positive")
    values = friction[all_windows[:, all_windows.shape[1] // 2]]
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    groups = [all_windows[(values >= edges[i]) &
                          (values <= edges[i + 1] if i == bins - 1 else values < edges[i + 1])]
              for i in range(bins)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("terrain stratification produced an empty friction bin")
    counts = np.full(bins, count // bins); counts[:count % bins] += 1
    chosen = np.concatenate([group[rng.choice(len(group), n, replace=n > len(group))]
                             for group, n in zip(groups, counts)])
    return chosen[rng.permutation(len(chosen))]


def balanced_surface_sample(all_windows, friction, count, ice_fraction, rng):
    """Backward-compatible binary sampler for old two-surface datasets."""
    if not 0.0 <= ice_fraction <= 1.0:
        raise ValueError("ice_fraction must lie in [0, 1]")
    values = friction[all_windows[:, all_windows.shape[1] // 2]]
    threshold = .5 * (values.min() + values.max())
    groups = (all_windows[values < threshold], all_windows[values >= threshold])
    counts = (int(round(count * ice_fraction)), count - int(round(count * ice_fraction)))
    chosen = np.concatenate([group[rng.choice(len(group), n, replace=n > len(group))]
                             for group, n in zip(groups, counts)])
    return chosen[rng.permutation(len(chosen))]


def causal_contexts(values, actions, episodes, history_steps, zero_action):
    """Build padded histories ending at each transition, without crossing episodes."""
    n = len(values)
    states = np.empty((n, history_steps, values.shape[1]), values.dtype)
    previous_actions = np.empty((n, history_steps - 1, actions.shape[1]), actions.dtype)
    starts, start = np.empty(n, np.int64), 0
    for i in range(n):
        if i == 0 or episodes[i] != episodes[i - 1]:
            start = i
        starts[i] = start
    row = np.arange(n)
    for lag in range(history_steps):
        states[:, history_steps - 1 - lag] = values[np.maximum(row - lag, starts)]
    for lag in range(1, history_steps):
        indices, slot = row - lag, history_steps - 1 - lag
        valid = indices >= starts
        previous_actions[:, slot] = zero_action
        previous_actions[valid, slot] = actions[indices[valid]]
    return states, previous_actions


def run(args):
    if args.segment_length % args.shooting_interval:
        raise ValueError("--segment-length must be divisible by --shooting-interval")
    with h5py.File(args.data) as f:
        x, u, xn, split, episodes, friction = (f[k][:] for k in
            ("state", "action", "next_state", "split", "trajectory_id", "friction"))
        dt = float(f.attrs["dt"])
        state_order = json.loads(f.attrs["state_order"])
    train_ids, validation_ids = np.unique(episodes[split == 0]), np.unique(episodes[split == 1])
    train_mask = split == 0
    state_dim = x.shape[1]
    structured = bool(args.structured_kinematics)
    compact_structured = structured and "cos_theta" in state_order
    extended_structured = structured and state_order == ["p_x", "p_y", "theta", "v_forward",
                                                          "v_lateral", "omega", "wheel_left", "wheel_right"]
    if structured and not (compact_structured or extended_structured):
        raise ValueError("--structured-kinematics requires compact or extended physical state")
    if compact_structured:
        selected_indices = (5, 6, 7, 8, 9, 10, 11, 12, 2)
    elif extended_structured:
        selected_indices = (3, 4, 5, 6, 7)
    else:
        selected_indices = tuple(range(state_dim))
    dynamic_indices = np.asarray(selected_indices, np.int32)
    angle_index = state_order.index("theta") if "theta" in state_order else None
    delta = angle_delta(xn.copy(), x, angle_index)
    # Circular heading features remove the artificial discontinuity at +/-pi.
    features = (np.column_stack((x[:, :2], np.sin(x[:, 2]), np.cos(x[:, 2]), x[:, 3:]))
                if angle_index == 2 else x.copy())
    feature_center, feature_scale = robust_center_scale(features[train_mask])
    action_center, action_scale = robust_center_scale(u[train_mask])
    full_delta_center, full_delta_scale = robust_center_scale(delta[train_mask])
    delta_center, delta_scale = (full_delta_center[dynamic_indices], full_delta_scale[dynamic_indices])
    _, state_score_scale = robust_center_scale(x[train_mask])
    if angle_index is not None:
        state_score_scale[angle_index] = 1.0
    stats = {"feature_mean": feature_center, "feature_std": feature_scale,
             "action_mean": action_center, "action_std": action_scale,
             "delta_mean": delta_center, "delta_std": delta_scale,
             "full_delta_mean": full_delta_center, "full_delta_std": full_delta_scale,
             "state_score_scale": state_score_scale,
             "normalization": np.asarray("robust_q01_q99_circular_heading")}
    fs = (features - feature_center) / feature_scale
    us = (u - action_center) / action_scale
    ds = (delta[:, dynamic_indices] - delta_center) / delta_scale
    zero_action = (np.zeros(2, np.float32) - action_center) / action_scale
    state_context, action_context = causal_contexts(x, us, episodes, args.history_steps, zero_action)
    feature_context, _ = causal_contexts(fs, us, episodes, args.history_steps, zero_action)
    encoded_context = np.concatenate((feature_context.reshape(len(x), -1),
                                      action_context.reshape(len(x), -1)), axis=-1)
    train_windows = windows(episodes, train_ids, args.segment_length, args.stride)
    validation_windows = windows(episodes, validation_ids, args.segment_length, args.stride)
    feature_dim = features.shape[1]
    input_dim = feature_dim * args.history_steps + 2 * (args.history_steps - 1)
    output_dim = len(dynamic_indices)
    model = FlowSDE(input_dim, 2, output_dim, jr.PRNGKey(args.seed), dt=dt, width=args.width,
                    depth=args.depth, condition_dim=args.condition_dim, flow_layers=args.flow_layers,
                    flow_scale_limit=args.flow_scale_limit)
    fc, fscale, dm, dscale, score_scale = map(jnp.asarray,
        (feature_center, feature_scale, delta_center, delta_scale, state_score_scale))
    dynamic_indices_jax = jnp.asarray(dynamic_indices)

    def encode(state):
        raw = (jnp.concatenate((state[:2], jnp.sin(state[2:3]), jnp.cos(state[2:3]), state[3:]))
               if angle_index == 2 else state)
        return (raw - fc) / fscale

    def encode_history(states, previous_actions):
        return jnp.concatenate((jax.vmap(encode)(states).reshape(-1), previous_actions.reshape(-1)))

    def reconstruct(state, predicted_delta):
        if not structured:
            nxt = state + predicted_delta
            if angle_index is not None:
                return nxt.at[angle_index].set(jnp.arctan2(
                    jnp.sin(nxt[angle_index]), jnp.cos(nxt[angle_index])))
            return nxt
        nxt = state.at[dynamic_indices_jax].add(predicted_delta)
        vf_i, vl_i, omega_i = (5, 6, 8) if compact_structured else (3, 4, 5)
        vf, vl = .5 * (state[vf_i] + nxt[vf_i]), .5 * (state[vl_i] + nxt[vl_i])
        omega = .5 * (state[omega_i] + nxt[omega_i])
        theta = (jnp.arctan2(state[4], state[3]) if compact_structured else state[2]) + dt * omega
        c, s = jnp.cos(theta), jnp.sin(theta)
        nxt = nxt.at[0].set(state[0] + dt * (vf * c - vl * s))
        nxt = nxt.at[1].set(state[1] + dt * (vf * s + vl * c))
        return (nxt.at[3].set(c).at[4].set(s) if compact_structured
                else nxt.at[2].set(jnp.arctan2(jnp.sin(theta), jnp.cos(theta))))

    def state_error(first, second):
        error = first - second
        if angle_index is not None:
            error = error.at[..., angle_index].set(jnp.arctan2(
                jnp.sin(error[..., angle_index]), jnp.cos(error[..., angle_index])))
        return error / score_scale

    def path(candidate, initial_states, previous_actions, actions, keys, stochastic):
        def step(carry, item):
            states, action_history = carry
            action, key = item
            state = states[-1]
            context = encode_history(states, action_history)
            delta_n = (candidate.step(context, action, key) if stochastic
                       else candidate.mean_delta(context, action))
            nxt = reconstruct(state, delta_n * dscale + dm)
            states = jnp.concatenate((states[1:], nxt[None]), axis=0)
            if args.history_steps > 1:
                action_history = jnp.concatenate((action_history[1:], action[None]), axis=0)
            return (states, action_history), nxt
        return jax.lax.scan(step, (initial_states, previous_actions), (actions, keys))[1]

    def components(candidate, bx, bu, bd, state_histories, action_histories, truth, keys, use_energy):
        drift = jnp.mean((jax.vmap(jax.vmap(candidate.mean_delta))(bx, bu) - bd) ** 2)
        transition = jax.lax.cond(use_energy,
            lambda: -jnp.mean(jax.vmap(jax.vmap(candidate.log_prob))(bd, bx, bu)),
            lambda: jnp.asarray(0.0))
        # True multiple shooting: each short interval starts from its observed
        # shooting node. Its predicted endpoint must meet the following node.
        interval = args.shooting_interval
        node_states = state_histories[:, ::interval]
        node_actions = action_histories[:, ::interval]
        block_actions = bu.reshape(bu.shape[0], -1, interval, bu.shape[-1])
        block_keys = keys[:, 0].reshape(keys.shape[0], -1, interval, 2)
        node_count = node_states.shape[0] * node_states.shape[1]
        flat_paths = jax.vmap(path, in_axes=(None, 0, 0, 0, 0, None))(
            candidate, node_states.reshape(-1, args.history_steps, state_dim),
            node_actions.reshape(node_count, args.history_steps - 1, 2), block_actions.reshape(-1, interval, 2),
            block_keys.reshape(-1, interval, 2), False)
        predicted_nodes = flat_paths[:, -1].reshape(node_states.shape[:2] + (state_dim,))
        target_nodes = truth[:, interval - 1::interval]
        continuity = jnp.mean(state_error(predicted_nodes, target_nodes) ** 2)
        def energy_one(initial_states, previous_actions, actions, target, particle_keys):
            samples = jax.vmap(path, in_axes=(None, None, None, None, 0, None))(
                candidate, initial_states, previous_actions, actions, particle_keys, True)
            observation_error = state_error(samples, target[None]).reshape(args.particles, -1)
            observation = jnp.mean(jnp.linalg.norm(observation_error, axis=-1))
            paired = jnp.roll(samples, 1, axis=0)
            diversity_error = state_error(samples, paired).reshape(args.particles, -1)
            diversity = .5 * jnp.mean(jnp.linalg.norm(diversity_error, axis=-1))
            return (observation - diversity) / jnp.sqrt(target.size)
        energy = jax.lax.cond(use_energy,
            lambda: jnp.mean(jax.vmap(energy_one)(state_histories[:, 0], action_histories[:, 0],
                                                  bu, truth, keys)),
            lambda: jnp.asarray(0.0))
        return drift, transition, continuity, energy

    def stage_loss(candidate, bx, bu, bd, state_histories, action_histories, truth, keys, joint):
        drift, nll, continuity, energy = components(candidate, bx, bu, bd, state_histories,
                                                    action_histories, truth, keys, joint)
        drift_total = args.drift_weight * drift + args.continuity_weight * continuity
        total = jax.lax.cond(joint,
            lambda: args.transition_weight * nll + args.joint_drift_weight * drift +
                    args.continuity_weight * continuity + args.energy_weight * energy,
            lambda: drift_total)
        return total, (drift, nll, continuity, energy)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
            name=args.wandb_run_name or None, config={**{k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()}, "dt": dt, "normalization": "robust_q01_q99_circular_heading"})
        wandb_run.define_metric("global_epoch"); wandb_run.define_metric("*", step_metric="global_epoch")

    rng, key, history = np.random.default_rng(args.seed), jr.PRNGKey(args.seed + 1), []
    args.output.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); global_epoch = 0
    for stage, epochs, initial_lr, joint in (("drift", args.drift_epochs, args.drift_lr, False),
                                             ("joint", args.joint_epochs, args.joint_lr, True)):
        batches = max(1, int(np.ceil(min(args.segments_per_epoch, len(train_windows)) / args.batch_size)))
        total_steps = max(2, epochs * batches)
        warmup_steps = min(args.warmup_epochs * batches, total_steps - 1)
        schedule = optax.warmup_cosine_decay_schedule(initial_lr * .1, initial_lr,
            warmup_steps, total_steps, end_value=initial_lr * args.final_lr_ratio)
        optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                                optax.adamw(schedule, weight_decay=args.weight_decay))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        @eqx.filter_jit
        def update(candidate, state, bx, bu, bd, state_histories, action_histories, truth, keys):
            (value, parts), grads = eqx.filter_value_and_grad(stage_loss, has_aux=True)(
                candidate, bx, bu, bd, state_histories, action_histories, truth, keys, joint)
            updates, state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
            return eqx.apply_updates(candidate, updates), state, value, parts
        best_model, best_validation, stale, optimizer_step = model, float("inf"), 0, 0
        for epoch in range(1, epochs + 1):
            if args.terrain_bins <= 1:
                indices = rng.permutation(len(train_windows))[:args.segments_per_epoch]
                chosen = train_windows[indices]
            else:
                chosen = terrain_stratified_sample(train_windows, friction,
                    args.segments_per_epoch, args.terrain_bins, rng)
            values = []
            for offset in range(0, len(chosen), args.batch_size):
                rows = chosen[offset:offset + args.batch_size]
                key, batch_key = jr.split(key)
                keys = jr.split(batch_key, len(rows) * args.particles * args.segment_length).reshape(
                    len(rows), args.particles, args.segment_length, 2)
                model, opt_state, value, parts = update(model, opt_state, jnp.asarray(encoded_context[rows]),
                    jnp.asarray(us[rows]), jnp.asarray(ds[rows]), jnp.asarray(state_context[rows]),
                    jnp.asarray(action_context[rows]), jnp.asarray(xn[rows]), keys)
                values.append((float(value), *(float(part) for part in parts))); optimizer_step += 1
            vi = validation_windows[:min(args.validation_segments, len(validation_windows))]
            key, validation_key = jr.split(key)
            vkeys = jr.split(validation_key, len(vi) * args.particles * args.segment_length).reshape(
                len(vi), args.particles, args.segment_length, 2)
            validation, validation_parts = stage_loss(model, jnp.asarray(encoded_context[vi]), jnp.asarray(us[vi]),
                jnp.asarray(ds[vi]), jnp.asarray(state_context[vi]), jnp.asarray(action_context[vi]),
                jnp.asarray(xn[vi]), vkeys, joint)
            mean = np.mean(values, axis=0); global_epoch += 1
            row = {"stage": stage, "stage_epoch": epoch, "global_epoch": global_epoch,
                "learning_rate": float(schedule(optimizer_step)), "train_total": float(mean[0]),
                "train_drift_mse": float(mean[1]), "train_nll": float(mean[2]),
                "train_continuity_mse": float(mean[3]), "train_shooting_mse": float(mean[3]),
                "train_energy_score": float(mean[4]),
                "validation_total": float(validation), "validation_drift_mse": float(validation_parts[0]),
                "validation_nll": float(validation_parts[1]),
                "validation_continuity_mse": float(validation_parts[2]),
                "validation_shooting_mse": float(validation_parts[2]),
                "validation_energy_score": float(validation_parts[3])}
            history.append(row); print(json.dumps(row), flush=True)
            if wandb_run is not None: wandb_run.log(row)
            if float(validation) < best_validation - args.min_delta:
                best_model, best_validation, stale = model, float(validation), 0
            else: stale += 1
            if stale >= args.patience:
                print(json.dumps({"stage": stage, "early_stop_epoch": epoch}), flush=True); break
        model = best_model
        stage_dir = args.output / f"{stage}_stage"; stage_dir.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(stage_dir / "flowsde.eqx", model); np.savez(stage_dir / "normalization.npz", **stats)
        config = {**vars(args), "data": str(args.data), "output": str(stage_dir), "dt": dt,
                  "input_dim": input_dim, "state_dim": state_dim, "output_dim": output_dim,
                  "dynamic_indices": dynamic_indices.tolist(), "state_order": state_order,
                  "best_validation_total": best_validation,
                  "elapsed_seconds": time.perf_counter() - started}
        (stage_dir / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
        (stage_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.summary["best_joint_validation_total"] = best_validation; wandb_run.finish()
    return model, history


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, default=Path("data/differential_drive/smooth_terrain_navigation.h5"))
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/differential_drive/smooth_terrain_seed0"))
    p.add_argument("--segment-length", type=int, default=100, help="Complete path used by the energy score")
    p.add_argument("--history-steps", type=int, default=1,
                   help="Number of causal state samples; 10 covers 0.2 seconds at dt=0.02")
    p.add_argument("--shooting-interval", type=int, default=5, help="Steps between continuity nodes")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--particles", type=int, default=4); p.add_argument("--segments-per-epoch", type=int, default=4096)
    p.add_argument("--terrain-bins", type=int, default=4,
                   help="Equal-size friction quantile bins for robust terrain coverage")
    p.add_argument("--validation-segments", type=int, default=512); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--drift-epochs", type=int, default=120); p.add_argument("--joint-epochs", type=int, default=180)
    p.add_argument("--drift-lr", type=float, default=3e-4); p.add_argument("--joint-lr", type=float, default=5e-5)
    p.add_argument("--warmup-epochs", type=int, default=5); p.add_argument("--final-lr-ratio", type=float, default=.05)
    p.add_argument("--drift-weight", type=float, default=1.0); p.add_argument("--joint-drift-weight", type=float, default=2.0)
    p.add_argument("--transition-weight", type=float, default=1.0); p.add_argument("--continuity-weight", type=float, default=5.0)
    p.add_argument("--energy-weight", type=float, default=2.0, help="Factor multiplying normalized path energy score")
    p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=25); p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=192); p.add_argument("--depth", type=int, default=3)
    p.add_argument("--condition-dim", type=int, default=96); p.add_argument("--flow-layers", type=int, default=6)
    p.add_argument("--flow-scale-limit", type=float, default=.25,
                   help="Per-coupling |log scale| bound imposed through tanh")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--wandb-project", default="DifferentialDrive-FlowSDE")
    p.add_argument("--wandb-entity", default=""); p.add_argument("--wandb-run-name", default="")
    p.add_argument("--structured-kinematics", action="store_true",
                   help="Predict only non-pose dynamics and integrate planar pose analytically")
    return p


if __name__ == "__main__": run(parser().parse_args())
