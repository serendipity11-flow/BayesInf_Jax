"""Train a conditional FlowSDE on off-road rover transitions."""

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


def load(path):
    with np.load(path) as bundle:
        states = bundle["observation.state"].astype(np.float32)
        actions = bundle["action"].astype(np.float32)
        next_states = bundle["next_observation.state"].astype(np.float32)
        episodes = bundle["episode_index"]
    return states, actions, next_states - states, episodes


def episode_split(episodes, seed):
    unique = np.random.default_rng(seed).permutation(np.unique(episodes))
    n_test = max(1, round(0.1 * len(unique)))
    n_val = max(1, round(0.1 * len(unique)))
    return unique[n_test + n_val:], unique[n_test:n_test + n_val], unique[:n_test]


def train(args):
    states, actions, deltas, episodes = load(args.data)
    train_ids, val_ids, test_ids = episode_split(episodes, args.seed)
    train_mask, val_mask = np.isin(episodes, train_ids), np.isin(episodes, val_ids)
    stats = {
        "state_mean": states[train_mask].mean(0), "state_std": states[train_mask].std(0) + 1e-6,
        "action_mean": actions[train_mask].mean(0), "action_std": actions[train_mask].std(0) + 1e-6,
        "delta_mean": deltas[train_mask].mean(0), "delta_std": deltas[train_mask].std(0) + 1e-6,
    }
    ns = (states - stats["state_mean"]) / stats["state_std"]
    na = (actions - stats["action_mean"]) / stats["action_std"]
    nd = (deltas - stats["delta_mean"]) / stats["delta_std"]
    model = FlowSDE(8, 2, 8, jr.PRNGKey(args.seed), dt=args.dt, width=args.width,
                    depth=args.depth, condition_dim=args.condition_dim,
                    flow_layers=args.flow_layers)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(args.lr, weight_decay=1e-5))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(candidate, state, x, u, dx):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: -jnp.mean(jax.vmap(m.log_prob)(dx, x, u)))(candidate)
        updates, state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), state, loss

    @eqx.filter_jit
    def metrics(candidate, x, u, dx):
        nll = -jnp.mean(jax.vmap(candidate.log_prob)(dx, x, u))
        mean = jax.vmap(candidate.mean_delta)(x, u)
        return nll, jnp.sqrt(jnp.mean((mean - dx) ** 2))

    train_indices, val_indices = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    best, best_nll, rows = model, float("inf"), []
    rng, started = np.random.default_rng(args.seed), time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses, order = [], rng.permutation(train_indices)
        for start in range(0, len(train_indices), args.batch_size):
            batch = order[start:start + args.batch_size]
            model, opt_state, loss = update(model, opt_state, *(jnp.asarray(v[batch]) for v in (ns, na, nd)))
            losses.append(float(loss))
        val_nll, val_rmse = map(float, metrics(model, *(jnp.asarray(v[val_indices]) for v in (ns, na, nd))))
        row = {"epoch": epoch, "train_nll": float(np.mean(losses)), "validation_nll": val_nll,
               "validation_normalized_rmse": val_rmse}
        rows.append(row); print(json.dumps(row), flush=True)
        if val_nll < best_nll: best, best_nll = model, val_nll
    args.output.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.output / "flowsde.eqx", best)
    np.savez(args.output / "normalization.npz", **stats)
    config = {**vars(args), "data": str(args.data), "output": str(args.output),
              "best_validation_nll": best_nll, "train_episodes": train_ids.tolist(),
              "validation_episodes": val_ids.tolist(), "test_episodes": test_ids.tolist(),
              "elapsed_seconds": time.perf_counter() - started}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(rows, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200); p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--width", type=int, default=192); p.add_argument("--depth", type=int, default=3)
    p.add_argument("--condition-dim", type=int, default=128); p.add_argument("--flow-layers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
