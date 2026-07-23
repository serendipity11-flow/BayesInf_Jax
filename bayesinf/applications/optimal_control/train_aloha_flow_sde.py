"""Train a controlled RealNVP FlowSDE on ALOHA Mobile Cabinet transitions."""

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .controlled_flow_sde import ControlledRealNVPFlowSDE
from .second_order_flow_sde import SecondOrderControlledRealNVPFlowSDE
from .aloha_data import (
    augment_with_causal_velocity, load_episodes, normalization,
    split_episode_indices,
)


def transition_arrays(episodes, actions, indices, x_mean, x_std, u_mean, u_std,
                      dt=0.02):
    """Flatten complete episodes without creating cross-episode transitions."""
    x0, x1, u0, times = [], [], [], []
    for index in indices:
        x = (episodes[index] - np.asarray(x_mean)) / np.asarray(x_std)
        u = (actions[index] - np.asarray(u_mean)) / np.asarray(u_std)
        x0.append(x[:-1]); x1.append(x[1:]); u0.append(u)
        times.append(np.arange(len(u), dtype=np.float32) * dt)
    return tuple(jnp.asarray(np.concatenate(values)) for values in (x0, u0, x1, times))


def trajectory_windows(episodes, actions, indices, x_mean, x_std, u_mean, u_std,
                       window_steps=128, stride=64):
    """Create episode-contained windows for stochastic multiple shooting."""
    states, controls, starts = [], [], []
    for index in indices:
        x = (episodes[index] - np.asarray(x_mean)) / np.asarray(x_std)
        u = (actions[index] - np.asarray(u_mean)) / np.asarray(u_std)
        for start in range(0, len(u) - window_steps + 1, stride):
            states.append(x[start:start + window_steps + 1])
            controls.append(u[start:start + window_steps])
            starts.append(start)
    if not states:
        raise ValueError("window configuration produced no training trajectories")
    return (jnp.asarray(np.stack(states)), jnp.asarray(np.stack(controls)),
            jnp.asarray(starts, dtype=jnp.int32))


def build_model(args, state_dim, control_dim):
    if getattr(args, "velocity_augmented", False):
        if state_dim % 2:
            raise ValueError("velocity-augmented state dimension must be even")
        return SecondOrderControlledRealNVPFlowSDE(
            state_dim // 2, control_dim, jr.PRNGKey(args.seed + 1),
            hidden_size=args.width, depth=args.depth,
            diffusion_hidden=args.diffusion_width,
            flow_layers=args.flow_layers, time_emb_dim=args.time_embedding,
            realnvp_alpha=args.realnvp_alpha, dt=args.dt,
            velocity_to_position_scale=args.velocity_to_position_scale,
            velocity_to_position_bias=args.velocity_to_position_bias,
        )
    return ControlledRealNVPFlowSDE(
        state_dim, control_dim, jr.PRNGKey(args.seed + 1),
        hidden_size=args.width, depth=args.depth,
        diffusion_hidden=args.diffusion_width, flow_layers=args.flow_layers,
        time_emb_dim=args.time_embedding, realnvp_alpha=args.realnvp_alpha,
        dt=args.dt,
        diffusion_condition_on_control=not getattr(
            args, "diffusion_no_control", False
        ),
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    position_dim = state_dim
    if args.velocity_augmented:
        episodes = augment_with_causal_velocity(
            episodes, dt=args.dt, window=args.velocity_window
        )
        state_dim = 2 * position_dim
    train_i, val_i, test_i = split_episode_indices(len(episodes), args.seed)
    if len(test_i) == 0:
        raise ValueError("dataset needs enough episodes for a held-out test split")
    stats = normalization(episodes, actions, train_i)
    if args.velocity_augmented:
        # In normalized coordinates, dq_n/dt = scale*v_n + bias.
        q_std = np.asarray(stats[1][:position_dim])
        v_mean = np.asarray(stats[0][position_dim:])
        v_std = np.asarray(stats[1][position_dim:])
        args.velocity_to_position_scale = (v_std / q_std).tolist()
        args.velocity_to_position_bias = (v_mean / q_std).tolist()
    if args.window_steps % args.segment_length:
        raise ValueError("--window-steps must be divisible by --segment-length")
    train = trajectory_windows(
        episodes, actions, train_i, *stats, args.window_steps, args.stride
    )
    validation = trajectory_windows(
        episodes, actions, val_i, *stats, args.window_steps, args.window_steps
    )
    test = transition_arrays(episodes, actions, test_i, *stats, dt=args.dt)
    model = build_model(args, state_dim, control_dim)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-5),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(current_model, current_state, batch, noise_key):
        bx, bu, bstart = batch
        loss, grads = eqx.filter_value_and_grad(
            lambda m: m.multiple_shooting_loss(
                bx, bu, bstart, noise_key,
                segment_length=args.segment_length,
                n_particles=args.train_particles, solver_name=args.solver,
                shooting_noise_std=args.state_noise_std,
                control_noise_std=args.control_noise_std,
                uncertainty_weight=args.uncertainty_weight,
                continuity_weight=args.continuity_weight,
            )
        )(current_model)
        updates, next_state = optimizer.update(
            grads, current_state, eqx.filter(current_model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(current_model, updates), next_state, loss

    best_model, best_validation = model, float("inf")
    curve, started = [], time.perf_counter()
    for epoch in range(args.epochs):
        order = np.asarray(jr.permutation(jr.PRNGKey(args.seed + 10 + epoch), len(train[0])))
        if args.samples_per_epoch > 0:
            order = order[:min(args.samples_per_epoch, len(order))]
        losses = []
        for batch_number, start in enumerate(range(0, len(order), args.batch_size)):
            indices = order[start:start + args.batch_size]
            batch = tuple(value[indices] for value in train)
            key = jr.fold_in(jr.PRNGKey(args.seed + 1000 + epoch), batch_number)
            model, opt_state, loss = train_step(model, opt_state, batch, key)
            losses.append(float(loss))
        # Use a deterministic fixed validation subset/key for comparable epochs.
        validation_order = np.asarray(jr.permutation(
            jr.PRNGKey(args.seed + 8000), len(validation[0])
        ))[:min(args.validation_windows, len(validation[0]))]
        validation_subset = tuple(value[validation_order] for value in validation)
        validation_loss = float(model.multiple_shooting_loss(
            *validation_subset, jr.PRNGKey(args.seed + 9000),
            segment_length=args.segment_length,
            n_particles=args.train_particles, solver_name=args.solver,
            shooting_noise_std=0.0,
            control_noise_std=0.0,
            uncertainty_weight=args.uncertainty_weight,
            continuity_weight=args.continuity_weight,
        ))
        train_loss = float(np.mean(losses))
        curve.append((epoch + 1, train_loss, validation_loss))
        if validation_loss < best_validation:
            best_model, best_validation = model, validation_loss
        print(f"epoch={epoch + 1:03d} train_path_loss={train_loss:.6f} "
              f"validation_path_loss={validation_loss:.6f}", flush=True)

    if args.velocity_augmented:
        diffusion_tag = "_qv"
    else:
        diffusion_tag = "_gx" if args.diffusion_no_control else ""
    tag = (f"aloha_flow_sde_{args.solver}_seg{args.segment_length}{diffusion_tag}_"
           f"noise{str(args.state_noise_std).replace('.', 'p')}_seed{args.seed}")
    checkpoint = args.output / f"{tag}.eqx"
    normalization_path = args.output / f"{tag}_normalization.npz"
    eqx.tree_serialise_leaves(checkpoint, best_model)
    np.savez(normalization_path, x_mean=np.asarray(stats[0]), x_std=np.asarray(stats[1]),
             u_mean=np.asarray(stats[2]), u_std=np.asarray(stats[3]))
    with (args.output / f"{tag}_learning_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["epoch", "train_path_loss", "validation_path_loss"])
        writer.writerows(curve)
    config = vars(args).copy()
    config.update({"data": str(args.data), "output": str(args.output),
                   "checkpoint": str(checkpoint), "normalization": str(normalization_path),
                   "state_dim": state_dim, "control_dim": control_dim,
                   "position_dim": position_dim,
                   "training_window_pool": len(train[0]),
                   "validation_window_pool": len(validation[0]),
                   "effective_training_windows_per_epoch": (
                       min(args.samples_per_epoch, len(train[0]))
                       if args.samples_per_epoch > 0 else len(train[0])
                   ),
                   "train_episode_ids": episode_ids[train_i].tolist(),
                   "validation_episode_ids": episode_ids[val_i].tolist(),
                   "test_episode_ids": episode_ids[test_i].tolist(),
                   "training_objective": "stochastic_multiple_shooting_energy_score",
                   "best_validation_path_loss": best_validation,
                   "test_transition_nll": float(best_model.transition_nll(*test)),
                   "elapsed_seconds": time.perf_counter() - started})
    with (args.output / f"{tag}_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    print(json.dumps(config, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/aloha_mobile_cabinet/flow_sde"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--diffusion-width", type=int, default=64)
    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--time-embedding", type=int, default=16)
    p.add_argument("--realnvp-alpha", type=float, default=3.0)
    p.add_argument("--diffusion-no-control", action="store_true",
                   help="Condition g on (epsilon, x, t), while f still uses (x, u, t).")
    p.add_argument("--velocity-augmented", action="store_true",
                   help="Use structured state [q,v], dq=v and stochastic dv.")
    p.add_argument("--velocity-window", type=int, default=3,
                   help="Backward causal finite-difference window in frames.")
    p.add_argument("--state-noise-std", type=float, default=0.01,
                   help="Gaussian perturbation of multiple-shooting initial states.")
    p.add_argument("--control-noise-std", type=float, default=0.01,
                   help="Gaussian perturbation on normalized control windows.")
    p.add_argument("--window-steps", type=int, default=128)
    p.add_argument("--segment-length", type=int, default=8)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=0,
                   help="Random windows drawn from the dense pool each epoch; 0 uses all.")
    p.add_argument("--train-particles", type=int, default=4)
    p.add_argument("--solver", choices=("euler", "heun", "reversible_heun",
                   "midpoint", "milstein"), default="heun")
    p.add_argument("--uncertainty-weight", type=float, default=0.5)
    p.add_argument("--continuity-weight", type=float, default=0.1)
    p.add_argument("--validation-windows", type=int, default=8)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
