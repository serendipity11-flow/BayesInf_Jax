"""Train a history-conditioned LatentSSMFlow on ALOHA Mobile Cabinet.

Training uses fixed-length segments, while evaluation recursively rolls the
model over every complete held-out episode.  The transition is

    x[t + 1] ~ p(. | GRU(x[t-H+1:t])[-1], u[t]).

Only proprioceptive state and action are used; videos and effort are ignored.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd

from .gru_latent_ssm_flow import GRULatentSSMFlow


def load_episodes(path: Path):
    """Load complete, ordered episodes from the LeRobot v3 parquet file."""
    frame = pd.read_parquet(
        path, columns=["observation.state", "action", "episode_index", "frame_index"]
    ).sort_values(["episode_index", "frame_index"])
    states, actions, ids = [], [], []
    for episode_id, episode in frame.groupby("episode_index", sort=True):
        x = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
        u = np.stack(episode["action"].to_numpy()).astype(np.float32)
        if x.shape[0] < 2 or x.shape != u.shape:
            raise ValueError(f"invalid episode {episode_id}: x={x.shape}, u={u.shape}")
        states.append(x)
        # LeRobot row t stores the action taking x[t] to x[t+1]. The final
        # episode action has no in-dataset target and is intentionally dropped.
        actions.append(u[:-1])
        ids.append(int(episode_id))
    if not states:
        raise ValueError(f"no episodes found in {path}")
    state_dim = states[0].shape[1]
    control_dim = actions[0].shape[1]
    if any(x.shape[1] != state_dim for x in states):
        raise ValueError("inconsistent observation.state dimensions")
    return states, actions, np.asarray(ids), state_dim, control_dim


def split_episode_indices(n_episodes, seed):
    """Deterministic 80/10/10 episode split with no temporal leakage."""
    order = np.asarray(jr.permutation(jr.PRNGKey(seed), n_episodes))
    n_train = max(1, int(0.8 * n_episodes))
    n_val = max(1, int(0.1 * n_episodes))
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def normalization(episodes, actions, indices):
    x = np.concatenate([episodes[i] for i in indices], axis=0)
    u = np.concatenate([actions[i] for i in indices], axis=0)
    x_mean, x_std = x.mean(0), np.maximum(x.std(0), 1e-4)
    u_mean, u_std = u.mean(0), np.maximum(u.std(0), 1e-4)
    return tuple(jnp.asarray(v) for v in (x_mean, x_std, u_mean, u_std))


def make_segments(episodes, actions, indices, history, segment_steps, stride,
                  x_mean, x_std, u_mean, u_std):
    """Return H+L state / H+L-1 action windows for segmented likelihood."""
    xs, us = [], []
    length = history + segment_steps
    for i in indices:
        x = (episodes[i] - np.asarray(x_mean)) / np.asarray(x_std)
        u = (actions[i] - np.asarray(u_mean)) / np.asarray(u_std)
        for start in range(0, x.shape[0] - length + 1, stride):
            xs.append(x[start:start + length])
            us.append(u[start:start + length - 1])
    if not xs:
        raise ValueError("segment/history configuration produced no windows")
    return jnp.asarray(np.stack(xs)), jnp.asarray(np.stack(us))


def full_horizon_metrics(model, episodes, actions, indices, history,
                         x_mean, x_std, u_mean, u_std):
    """Teacher-force only H initial states, then recursively predict to the end."""
    squared_error, rows = [], []
    # Equinox partitions array-valued module leaves from static structure;
    # plain jax.jit would try to hash the bound model's JAX arrays.
    rollout = eqx.filter_jit(model.rollout_deterministic_history)
    for i in indices:
        x = jnp.asarray((episodes[i] - np.asarray(x_mean)) / np.asarray(x_std))
        u = jnp.asarray((actions[i] - np.asarray(u_mean)) / np.asarray(u_std))
        prediction = rollout(
            x[:history], u[:history - 1], u[history - 1:]
        )
        target = x[history:]
        error = np.asarray((prediction - target) * x_std)
        squared_error.append(error ** 2)
        per_step = np.sqrt(np.mean(error ** 2, axis=1))
        rows.extend((int(i), step + 1, float(value))
                    for step, value in enumerate(per_step))
    all_sq = np.concatenate(squared_error, axis=0)
    return float(np.sqrt(all_sq.mean())), rows


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    train_i, val_i, test_i = split_episode_indices(len(episodes), args.seed)
    x_mean, x_std, u_mean, u_std = normalization(episodes, actions, train_i)
    train_x, train_u = make_segments(
        episodes, actions, train_i, args.history, args.segment_steps,
        args.stride, x_mean, x_std, u_mean, u_std,
    )
    val_x, val_u = make_segments(
        episodes, actions, val_i, args.history, args.segment_steps,
        args.segment_steps, x_mean, x_std, u_mean, u_std,
    )

    model = GRULatentSSMFlow(
        state_dim, control_dim, args.history, args.hidden_dim,
        jr.PRNGKey(args.seed + 1), predict_delta=True,
        n_coupling=args.blocks, hidden_size=args.width,
        n_hidden=args.hidden_layers, flow_type=args.flow_type,
        num_bins=args.num_bins, max_rho=args.max_rho,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(args.lr, weight_decay=1e-5)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(current_model, current_state, bx, bu, noise_key):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: (m.noisy_context_nll_loss(
                bx, bu, noise_key, args.state_noise_std
            ) if args.state_noise_std > 0 else m.nll_loss(bx, bu))
        )(
            current_model
        )
        updates, next_state = optimizer.update(
            grads, current_state, eqx.filter(current_model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(current_model, updates), next_state, loss

    best_model, best_val = model, float("inf")
    curve, started = [], time.perf_counter()
    for epoch in range(args.epochs):
        order = np.asarray(jr.permutation(jr.PRNGKey(args.seed + epoch + 10), len(train_x)))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            noise_key = jr.fold_in(jr.PRNGKey(args.seed + 1000 + epoch), start)
            model, opt_state, loss = step(
                model, opt_state, train_x[batch], train_u[batch], noise_key
            )
            losses.append(float(loss))
        val_nll = float(model.nll_loss(val_x, val_u))
        train_nll = float(np.mean(losses))
        curve.append((epoch + 1, train_nll, val_nll))
        if val_nll < best_val:
            best_model, best_val = model, val_nll
        print(f"epoch={epoch + 1:03d} train_nll={train_nll:.6f} val_nll={val_nll:.6f}", flush=True)

    test_x, test_u = make_segments(
        episodes, actions, test_i, args.history, args.segment_steps,
        args.segment_steps, x_mean, x_std, u_mean, u_std,
    )
    test_nll = float(best_model.nll_loss(test_x, test_u))
    full_rmse, horizon_rows = full_horizon_metrics(
        best_model, episodes, actions, test_i, args.history,
        x_mean, x_std, u_mean, u_std,
    )
    noise_tag = str(args.state_noise_std).replace(".", "p")
    prefix = (f"aloha_{args.flow_type}_noise{noise_tag}_h{args.history}_"
              f"seg{args.segment_steps}_seed{args.seed}")
    eqx.tree_serialise_leaves(output / f"{prefix}.eqx", best_model)
    np.savez(output / f"{prefix}_normalization.npz", x_mean=np.asarray(x_mean),
             x_std=np.asarray(x_std), u_mean=np.asarray(u_mean), u_std=np.asarray(u_std))
    with (output / f"{prefix}_learning_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["epoch", "train_nll", "validation_nll"])
        writer.writerows(curve)
    with (output / f"{prefix}_full_horizon.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["episode_array_index", "horizon_step", "rmse"])
        writer.writerows(horizon_rows)
    metrics = {
        "dataset": "lerobot/aloha_mobile_cabinet", "fps": 50,
        "flow_type": args.flow_type, "num_bins": args.num_bins,
        "state_noise_std_normalized": args.state_noise_std,
        "state_dim": state_dim, "control_dim": control_dim,
        "history": args.history, "segment_steps": args.segment_steps,
        "train_segments": len(train_x), "validation_segments": len(val_x),
        "train_episode_ids": episode_ids[train_i].tolist(),
        "validation_episode_ids": episode_ids[val_i].tolist(),
        "test_episode_ids": episode_ids[test_i].tolist(),
        "best_validation_nll": best_val, "test_segment_nll": test_nll,
        "test_full_horizon_rmse": full_rmse,
        "test_rollout_steps_per_episode": int(episodes[test_i[0]].shape[0] - args.history),
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output / f"{prefix}_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("results/results_optimal_control/aloha_mobile_cabinet"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--segment-steps", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--flow-type", choices=("realnvp", "nsf"), default="realnvp")
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--state-noise-std", type=float, default=0.05,
                   help="Gaussian noise on normalized state contexts; targets stay clean.")
    p.add_argument("--max-rho", type=float, default=0.8)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
