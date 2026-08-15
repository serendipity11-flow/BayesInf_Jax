"""Contact-balanced NLL training for the JAX conditional FlowSDE."""

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .data import load, normalize, split, statistics
from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE
from .state import ACTION_DIM, INPUT_DIM, OBJECT_MOVEMENT, STATE_DIM


def contact_weights(raw_delta, threshold=1e-3, active_weight=10.0):
    moving = jnp.linalg.norm(raw_delta[:, OBJECT_MOVEMENT], axis=-1) > threshold
    return jnp.where(moving, active_weight, 1.0)


def train(args):
    histories, actions, deltas, episodes = load(args.data)
    train_ids, validation_ids, test_ids = split(episodes, args.seed)
    train_mask, validation_mask = np.isin(episodes, train_ids), np.isin(episodes, validation_ids)
    stats = statistics(histories[train_mask], actions[train_mask], deltas[train_mask])
    nh, na, nd = normalize(histories, actions, deltas, stats)
    model = FlowSDE(INPUT_DIM, ACTION_DIM, STATE_DIM, jr.PRNGKey(args.seed), dt=args.dt,
                    width=args.width, depth=3, condition_dim=args.condition_dim,
                    flow_layers=args.flow_layers)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0),
                            optax.adamw(args.lr, weight_decay=1e-5))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(candidate, state, batch):
        bh, ba, bd, raw = batch
        def objective(m):
            nll = -jax.vmap(m.log_prob)(bd, bh, ba)
            weights = contact_weights(raw, args.contact_threshold, args.contact_weight)
            return jnp.sum(weights * nll) / jnp.sum(weights)
        loss, grads = eqx.filter_value_and_grad(objective)(candidate)
        updates, state = optimizer.update(grads, state,
                                          eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), state, loss

    @eqx.filter_jit
    def metrics(candidate, h, a, d):
        nll = -jnp.mean(jax.vmap(candidate.log_prob)(d, h, a))
        predicted = jax.vmap(candidate.mean_delta)(h, a)
        return nll, jnp.sqrt(jnp.mean((predicted - d) ** 2))

    train_indices, validation_indices = np.flatnonzero(train_mask), np.flatnonzero(validation_mask)
    best_model, best_nll, history_rows = model, float("inf"), []
    started = time.perf_counter(); rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_indices); losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            batch = tuple(jnp.asarray(value[index]) for value in (nh, na, nd, deltas))
            model, opt_state, loss = update(model, opt_state, batch); losses.append(float(loss))
        validation_nll, validation_rmse = map(float, metrics(
            model, jnp.asarray(nh[validation_indices]), jnp.asarray(na[validation_indices]),
            jnp.asarray(nd[validation_indices])))
        row = {"epoch": epoch, "train_weighted_nll": float(np.mean(losses)),
               "validation_nll": validation_nll, "validation_rmse": validation_rmse}
        history_rows.append(row); print(json.dumps(row), flush=True)
        if validation_nll < best_nll: best_model, best_nll = model, validation_nll

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "flowsde.eqx"
    eqx.tree_serialise_leaves(checkpoint, best_model)
    np.savez(args.output / "normalization.npz", **stats)
    residual = nd[train_indices] - np.asarray(jax.vmap(best_model.mean_delta)(
        jnp.asarray(nh[train_indices]), jnp.asarray(na[train_indices])))
    gaussian_log_std = np.log(np.maximum(residual.std(0) / np.sqrt(args.dt), 1e-5))
    np.save(args.output / "gaussian_log_std.npy", gaussian_log_std)
    config = vars(args).copy()
    config.update({"data": str(args.data), "output": str(args.output),
                   "checkpoint": str(checkpoint), "best_validation_nll": best_nll,
                   "train_episodes": train_ids.tolist(), "validation_episodes": validation_ids.tolist(),
                   "test_episodes": test_ids.tolist(), "elapsed_seconds": time.perf_counter() - started})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(history_rows, indent=2) + "\n")
    return best_model


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200); p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--width", type=int, default=256); p.add_argument("--condition-dim", type=int, default=128)
    p.add_argument("--flow-layers", type=int, default=4); p.add_argument("--contact-threshold", type=float, default=1e-3)
    p.add_argument("--contact-weight", type=float, default=10.0); p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__": train(parser().parse_args())
