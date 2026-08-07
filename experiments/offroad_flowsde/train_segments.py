"""Segment FlowSDE training with transition NLL and recursive energy score."""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE


def split_ids(episodes, seed):
    ids = np.random.default_rng(seed).permutation(np.unique(episodes))
    n = max(1, round(.1 * len(ids)))
    return ids[2*n:], ids[n:2*n], ids[:n]


def segment_indices(episodes, selected, length, stride):
    rows = []
    for episode in selected:
        idx = np.flatnonzero(episodes == episode)
        rows.extend(idx[start:start + length] for start in range(0, len(idx) - length + 1, stride))
    return np.stack(rows)


def train(args):
    if args.data.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(args.data) as data:
            x, u, xn, episodes, split = (data[name][:] for name in
                ("state", "action", "next_state", "trajectory_id", "split"))
        train_ids = np.unique(episodes[split == 0])
        val_ids = np.unique(episodes[split == 1])
        test_ids = np.unique(episodes[split == 2])
    else:
        with np.load(args.data) as data:
            x, u, xn, episodes = (data[name].copy() for name in
                ("observation.state", "action", "next_observation.state", "episode_index"))
        train_ids, val_ids, test_ids = split_ids(episodes, args.seed)
    if x.shape[1] != 10 or u.shape[1] != 2:
        raise ValueError("expected wheel-resolved state dimension 10 and action dimension 2")
    train_mask = np.isin(episodes, train_ids)
    delta = xn - x
    stats = {"state_mean": x[train_mask].mean(0), "state_std": x[train_mask].std(0) + 1e-6,
             "action_mean": u[train_mask].mean(0), "action_std": u[train_mask].std(0) + 1e-6,
             "delta_mean": delta[train_mask].mean(0), "delta_std": delta[train_mask].std(0) + 1e-6}
    xs = (x - stats["state_mean"]) / stats["state_std"]
    us = (u - stats["action_mean"]) / stats["action_std"]
    ds = (delta - stats["delta_mean"]) / stats["delta_std"]
    targets = (xn - stats["state_mean"]) / stats["state_std"]
    train_windows = segment_indices(episodes, train_ids, args.seg_len, args.stride)
    val_windows = segment_indices(episodes, val_ids, args.seg_len, args.stride)
    model = FlowSDE(10, 2, 10, jr.PRNGKey(args.seed), dt=args.dt, width=args.width,
                    depth=args.depth, condition_dim=args.condition_dim, flow_layers=args.flow_layers)
    optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                            optax.adamw(args.lr, weight_decay=args.weight_decay))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state_mean, state_std = map(jnp.asarray, (stats["state_mean"], stats["state_std"]))
    delta_mean, delta_std = map(jnp.asarray, (stats["delta_mean"], stats["delta_std"]))

    def particle_path(candidate, initial_normalized, actions_normalized, keys):
        def step(current_n, inputs):
            action_n, key = inputs
            delta_n = candidate.step(current_n, action_n, key, stochastic=True)
            current = current_n * state_std + state_mean
            nxt = current + delta_n * delta_std + delta_mean
            next_n = (nxt - state_mean) / state_std
            return next_n, next_n
        return jax.lax.scan(step, initial_normalized, (actions_normalized, keys))[1]

    def batch_loss(candidate, bx, bu, bd, by, keys):
        transition = -jnp.mean(jax.vmap(jax.vmap(candidate.log_prob))(bd, bx, bu))
        def window_score(initial, actions, truth, window_keys):
            samples = jax.vmap(particle_path, in_axes=(None, None, None, 0))(
                candidate, initial, actions, window_keys)
            truth_flat = truth.reshape(-1); sample_flat = samples.reshape(args.particles, -1)
            observation = jnp.mean(jnp.linalg.norm(sample_flat - truth_flat, axis=-1))
            diversity = .5 * jnp.mean(jnp.linalg.norm(sample_flat - jnp.roll(sample_flat, 1, axis=0), axis=-1))
            return (observation - diversity) / jnp.sqrt(truth_flat.size)
        energy = jnp.mean(jax.vmap(window_score)(bx[:, 0], bu, by, keys))
        total = args.transition_weight * transition + args.energy_weight * energy
        return total, (transition, energy)

    @eqx.filter_jit
    def update(candidate, state, bx, bu, bd, by, keys):
        (loss, parts), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(
            candidate, bx, bu, bd, by, keys)
        updates, state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), state, loss, parts

    best, best_val, history = model, float("inf"), []
    rng, key, started = np.random.default_rng(args.seed), jr.PRNGKey(args.seed + 1), time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_windows))
        if args.samples_per_epoch > 0:
            order = order[:args.samples_per_epoch]
        epoch_values = []
        for start in range(0, len(order), args.batch_size):
            indices = train_windows[order[start:start + args.batch_size]]
            key, batch_key = jr.split(key)
            keys = jr.split(batch_key, len(indices) * args.particles * args.seg_len).reshape(
                len(indices), args.particles, args.seg_len, 2)
            model, opt_state, loss, parts = update(model, opt_state,
                jnp.asarray(xs[indices]), jnp.asarray(us[indices]), jnp.asarray(ds[indices]),
                jnp.asarray(targets[indices]), keys)
            epoch_values.append((float(loss), float(parts[0]), float(parts[1])))
        # Fixed-size validation keeps segment evaluation predictable on CPU.
        vi = val_windows[:min(args.validation_segments, len(val_windows))]
        key, validation_key = jr.split(key)
        vkeys = jr.split(validation_key, len(vi) * args.particles * args.seg_len).reshape(
            len(vi), args.particles, args.seg_len, 2)
        val, val_parts = batch_loss(model, jnp.asarray(xs[vi]), jnp.asarray(us[vi]),
                                    jnp.asarray(ds[vi]), jnp.asarray(targets[vi]), vkeys)
        means = np.mean(epoch_values, axis=0)
        row = {"epoch": epoch, "train_total": float(means[0]), "train_transition_nll": float(means[1]),
               "train_energy_score": float(means[2]), "validation_total": float(val),
               "validation_transition_nll": float(val_parts[0]), "validation_energy_score": float(val_parts[1])}
        history.append(row); print(json.dumps(row), flush=True)
        if float(val) < best_val: best, best_val = model, float(val)
    args.output.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.output / "flowsde_v2.eqx", best)
    np.savez(args.output / "normalization.npz", **stats)
    config = {**vars(args), "data": str(args.data), "output": str(args.output),
              "state_order": ["p_x", "p_y", "theta", "v_x", "v_y", "omega",
                              "omega_fl", "omega_fr", "omega_rl", "omega_rr"],
              "drift_inputs": ["x", "u"], "diffusion_conditioner_inputs": ["x", "u"],
              "train_episodes": train_ids.tolist(), "validation_episodes": val_ids.tolist(),
              "test_episodes": test_ids.tolist(), "train_segments": len(train_windows),
              "best_validation_total": best_val, "elapsed_seconds": time.perf_counter() - started}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seg-len", type=int, default=50); p.add_argument("--stride", type=int, default=5)
    p.add_argument("--transition-weight", type=float, default=1.0); p.add_argument("--energy-weight", type=float, default=.1)
    p.add_argument("--particles", type=int, default=4); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256); p.add_argument("--validation-segments", type=int, default=1024)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--dt", type=float, default=.05)
    p.add_argument("--width", type=int, default=192); p.add_argument("--depth", type=int, default=3)
    p.add_argument("--condition-dim", type=int, default=128); p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
