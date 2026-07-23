"""Train controlled LatentSSMFlow on SO-100 joint trajectories.

Input NPZ format:
  states:   (episodes, steps + 1, 10), ordered [q(5), dq(5)]
  controls: (episodes, steps, 5), the commands actually applied to the robot

Without ``--data`` a labeled stochastic 5-DoF surrogate is generated.  This is
for pipeline validation only and must not be presented as a hardware result.
"""

import argparse
import csv
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .controlled_latent_ssm_flow import ControlledLatentSSMFlow
from .gru_latent_ssm_flow import GRULatentSSMFlow


def synthetic_dataset(key, episodes=256, steps=24, dt=0.05):
    """Nonlinear, heteroscedastic second-order joint surrogate."""
    kq, ku, kn = jr.split(key, 3)
    q0 = 0.6 * jr.uniform(kq, (episodes, 5), minval=-1.0, maxval=1.0)
    x0 = jnp.concatenate([q0, jnp.zeros_like(q0)], axis=-1)
    raw_u = jr.normal(ku, (episodes, steps, 5))
    # Smooth excitation resembles bounded joint-velocity/torque experiments.
    controls = jnp.tanh(jax.lax.associative_scan(
        lambda a, b: 0.82 * a + 0.18 * b, raw_u, axis=1
    ))
    noise = jr.normal(kn, (episodes, steps, 5))

    def episode(x_init, episode_inputs):
        u_seq, eps_seq = episode_inputs
        def step(x, inp):
            u, eps = inp
            q, dq = x[:5], x[5:]
            # Coupling and state-dependent stochasticity make a nontrivial
            # transition density while retaining plausible joint mechanics.
            coupling = 0.08 * (jnp.roll(q, 1) - 2.0 * q + jnp.roll(q, -1))
            accel = 2.2 * u - 0.35 * dq - 0.55 * jnp.sin(q) + coupling
            accel += (0.025 + 0.035 * jnp.abs(u)) * eps
            dq_next = dq + dt * accel
            q_next = q + dt * dq_next
            x_next = jnp.concatenate([q_next, dq_next])
            return x_next, x_next
        _, rest = jax.lax.scan(step, x_init, (u_seq, eps_seq))
        return jnp.concatenate([x_init[None], rest], axis=0)

    states = jax.vmap(episode)(x0, (controls, noise))
    return states, controls


def load_lerobot_parquet(path, steps, history_len=1):
    """Convert LeRobot position/action rows into fixed 5-DoF windows.

    The sixth gripper channel is excluded. Velocity is a causal backward
    difference of measured positions at the metadata rate of 30 Hz.
    Non-overlapping windows retain their source episode for leakage-free splits.
    """
    import pandas as pd

    table = pd.read_parquet(path).sort_values(["episode_index", "frame_index"])
    state_windows, control_windows, source_episodes = [], [], []
    dt = 1.0 / 30.0
    for episode_id, episode in table.groupby("episode_index", sort=True):
        q = np.stack(episode["observation.state"].to_numpy())[:, :5].astype(np.float32)
        action = np.stack(episode["action"].to_numpy())[:, :5].astype(np.float32)
        dq = np.concatenate([np.zeros_like(q[:1]), np.diff(q, axis=0) / dt], axis=0)
        state = np.concatenate([q, dq], axis=-1)
        # For one-step learning retain H conditioning transitions but only the
        # final pair predicts one target. Longer sequence experiments retain
        # their previous non-overlapping behavior.
        window_steps = history_len + steps - 1
        stride = 1 if steps == 1 else steps
        for start in range(0, len(episode) - window_steps, stride):
            state_windows.append(state[start:start + window_steps + 1])
            control_windows.append(action[start:start + window_steps])
            source_episodes.append(int(episode_id))
    if not state_windows:
        raise ValueError(f"no episode in {path} is long enough for {steps} transitions")
    return (jnp.asarray(np.stack(state_windows)),
            jnp.asarray(np.stack(control_windows)), np.asarray(source_episodes))


def load_data(path, key, episodes, steps, history_len=1):
    if path is None:
        states, controls = synthetic_dataset(
            key, episodes, history_len if steps == 1 else steps
        )
        return states, controls, "synthetic_surrogate", np.arange(episodes)
    if Path(path).suffix.lower() == ".parquet":
        states, controls, groups = load_lerobot_parquet(path, steps, history_len)
        return states, controls, "lerobot_hardware_parquet", groups
    with np.load(path) as data:
        if "states" not in data or "controls" not in data:
            raise ValueError("NPZ must contain 'states' and 'controls'")
        states = jnp.asarray(data["states"])
        controls = jnp.asarray(data["controls"])
    if states.ndim != 3 or controls.ndim != 3:
        raise ValueError("states and controls must be rank-3")
    if states.shape[0] != controls.shape[0] or states.shape[1] != controls.shape[1] + 1:
        raise ValueError("states must have one more time step than controls")
    if states.shape[-1] != 10 or controls.shape[-1] != 5:
        raise ValueError("SO-100 experiment expects state_dim=10 and control_dim=5")
    return states, controls, "hardware_npz", np.arange(states.shape[0])


def normalize(train_states, train_controls, states, controls):
    x_mean = jnp.mean(train_states, axis=(0, 1))
    x_std = jnp.maximum(jnp.std(train_states, axis=(0, 1)), 1e-3)
    u_mean = jnp.mean(train_controls, axis=(0, 1))
    u_std = jnp.maximum(jnp.std(train_controls, axis=(0, 1)), 1e-3)
    return ((states - x_mean) / x_std, (controls - u_mean) / u_std,
            x_mean, x_std, u_mean, u_std)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_key, split_key = jr.split(jr.PRNGKey(args.seed))
    states, controls, source, groups = load_data(
        args.data, data_key, args.episodes, args.steps, args.history
    )
    unique_groups = np.unique(groups)
    group_perm = np.asarray(jr.permutation(split_key, jnp.asarray(unique_groups)))
    n_train_groups = max(1, int(0.8 * len(unique_groups)))
    n_val_groups = max(1, int(0.1 * len(unique_groups)))
    train_groups = group_perm[:n_train_groups]
    val_groups = group_perm[n_train_groups:n_train_groups + n_val_groups]
    test_groups = group_perm[n_train_groups + n_val_groups:]
    train_idx = np.flatnonzero(np.isin(groups, train_groups))
    val_idx = np.flatnonzero(np.isin(groups, val_groups))
    test_idx = np.flatnonzero(np.isin(groups, test_groups))
    n_train = len(train_idx)
    train_x_raw, train_u_raw = states[train_idx], controls[train_idx]
    x, u, x_mean, x_std, u_mean, u_std = normalize(
        train_x_raw, train_u_raw, states, controls
    )

    if args.gru:
        model = GRULatentSSMFlow(
            10, 5, args.history, args.condition_dim, jr.PRNGKey(args.seed + 100),
            predict_delta=args.residual, n_coupling=args.blocks,
            hidden_size=args.width, n_hidden=2, max_rho=args.max_rho,
        )
    else:
        model = ControlledLatentSSMFlow(
            10, 5, args.history, jr.PRNGKey(args.seed + 100),
            predict_delta=args.residual,
            d_enc=args.width, d_out=args.condition_dim, n_coupling=args.blocks,
            hidden_size=args.width, n_hidden=2, max_rho=args.max_rho,
        )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr, 1e-5))
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def train_step(current_model, current_opt_state, bx, bu):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: (m.nll_loss(bx, bu) if args.gru else
                       (m.history_one_step_nll(bx, bu)
                        if args.steps == 1 else m.nll_loss(bx, bu)))
        )(current_model)
        updates, next_opt_state = optimizer.update(
            grads, current_opt_state, eqx.filter(current_model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(current_model, updates), next_opt_state, loss

    best_val = float("inf")
    best_model = model
    rows = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_perm = np.asarray(jr.permutation(jr.fold_in(split_key, epoch), n_train))
        losses = []
        for start in range(0, n_train, args.batch_size):
            idx = train_idx[epoch_perm[start:start + args.batch_size]]
            model, opt_state, loss = train_step(model, opt_state, x[idx], u[idx])
            losses.append(float(loss))
        val_nll = float(
            model.nll_loss(x[val_idx], u[val_idx]) if args.gru else
            (model.history_one_step_nll(x[val_idx], u[val_idx])
             if args.steps == 1 else model.nll_loss(x[val_idx], u[val_idx]))
        )
        train_nll = float(np.mean(losses))
        rows.append((epoch + 1, train_nll, val_nll))
        if val_nll < best_val:
            best_val, best_model = val_nll, model
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch + 1:4d} train_nll={train_nll:.5f} val_nll={val_nll:.5f}")

    test_nll = float(
        best_model.nll_loss(x[test_idx], u[test_idx]) if args.gru else
        (best_model.history_one_step_nll(x[test_idx], u[test_idx])
         if args.steps == 1 else best_model.nll_loss(x[test_idx], u[test_idx]))
    )
    # One-step conditional reconstruction is measured by particle means.  Use
    # the first test episode to keep this diagnostic inexpensive and reproducible.
    episode = int(test_idx[0])
    h = args.history
    # The one-step dataset already contains H real context states/actions.
    if args.gru:
        prediction = best_model.sample_rollout(
            x[episode, :h], u[episode, :h - 1], u[episode, h - 1:h],
            jr.PRNGKey(args.seed + 200), n_particles=args.eval_particles,
        ).mean(axis=0)
        target = x[episode, h:h + 1]
    elif args.steps == 1:
        prediction = best_model.sample_rollout(
            x[episode, :h], u[episode, :h - 1], u[episode, h - 1:h],
            jr.PRNGKey(args.seed + 200), n_particles=args.eval_particles,
        ).mean(axis=0)
        target = x[episode, h:h + 1]
    else:
        prediction = best_model.sample_rollout(
            jnp.repeat(x[episode, :1], h, axis=0),
            jnp.zeros((h - 1, 5)), u[episode], jr.PRNGKey(args.seed + 200),
            n_particles=args.eval_particles,
        ).mean(axis=0)
        target = x[episode, 1:]
    rollout_rmse = float(jnp.sqrt(jnp.mean(
        ((prediction - target) * x_std) ** 2
    )))
    elapsed = time.perf_counter() - started

    prefix = f"history{args.history}_seed{args.seed}"
    eqx.tree_serialise_leaves(output / f"{prefix}.eqx", best_model)
    np.savez(output / f"{prefix}_normalization.npz", x_mean=np.asarray(x_mean),
             x_std=np.asarray(x_std), u_mean=np.asarray(u_mean), u_std=np.asarray(u_std))
    with (output / f"{prefix}_learning_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_nll", "validation_nll"])
        writer.writerows(rows)
    metrics = {
        "source": source, "seed": args.seed, "history": args.history,
        "episodes": int(len(unique_groups)), "windows": int(states.shape[0]),
        "context_steps": int(controls.shape[1]), "target_horizon": args.steps,
        "best_validation_nll": best_val, "test_nll": test_nll,
        "open_loop_particle_mean_rmse": rollout_rmse,
        "elapsed_seconds": elapsed,
    }
    with (output / f"{prefix}_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics.keys())
        writer.writeheader(); writer.writerow(metrics)
    print(metrics)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", default="results/results_optimal_control/so100_dynamics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--condition-dim", type=int, default=48)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--max-rho", type=float, default=0.8)
    p.add_argument("--residual", action="store_true",
                   help="Learn normalized state increments and reconstruct x[t+1]=x[t]+delta.")
    p.add_argument("--gru", action="store_true",
                   help="Use final-state GRU encoding and flow condition [h_t, u_t].")
    p.add_argument("--eval-particles", type=int, default=32)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
