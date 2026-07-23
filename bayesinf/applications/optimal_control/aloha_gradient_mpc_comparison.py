"""Compare learned-flow and fitted physical-model gradient MPC on ALOHA.

The held-out dataset has no counterfactual outcomes for optimized commands, so
both controllers are evaluated against the same fitted second-order servo
plant. Results are an offline simulation benchmark, not a hardware claim.
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
import matplotlib.pyplot as plt
import numpy as np

from .gru_latent_ssm_flow import GRULatentSSMFlow
from .gradient_based_mpc import GradientMPC, GradientMPCConfig
from .train_aloha_mobile_dynamics import load_episodes, split_episode_indices


class SecondOrderServo(eqx.Module):
    """Per-joint servo in normalized state/action coordinates.

    The action and state dataset channels are normalized independently. The
    affine action-to-state conversion is therefore applied before forming the
    physical position-command error.
    """
    momentum: jnp.ndarray
    command_gain: jnp.ndarray
    action_to_state_scale: jnp.ndarray
    action_to_state_bias: jnp.ndarray
    history_len: int = eqx.field(static=True)

    def __init__(self, momentum, command_gain, action_to_state_scale,
                 action_to_state_bias, history_len=8):
        self.momentum = jnp.asarray(momentum)
        self.command_gain = jnp.asarray(command_gain)
        self.action_to_state_scale = jnp.asarray(action_to_state_scale)
        self.action_to_state_bias = jnp.asarray(action_to_state_bias)
        self.history_len = history_len

    def predict_next(self, history, action):
        velocity = history[-1] - history[-2]
        action_as_state = (
            self.action_to_state_scale * action + self.action_to_state_bias
        )
        return history[-1] + self.momentum * velocity + self.command_gain * (
            action_as_state - history[-1]
        )

    def rollout_deterministic_history(self, state_history, previous_controls,
                                      future_controls):
        del previous_controls
        def step(history, action):
            state = self.predict_next(history, action)
            return jnp.concatenate([history[1:], state[None]], axis=0), state
        _, states = jax.lax.scan(step, state_history, future_controls)
        return states

    def sample_rollout(self, state_history, previous_controls, future_controls,
                       key, n_particles=1):
        """Sampling-MPC interface; this simple physical model is deterministic."""
        del key
        trajectory = self.rollout_deterministic_history(
            state_history, previous_controls, future_controls
        )
        return jnp.broadcast_to(
            trajectory, (n_particles,) + trajectory.shape
        )


def fit_servo(episodes, actions, train_indices, x_mean, x_std, u_mean, u_std):
    """Least-squares physical parameters, independently for each joint."""
    action_to_state_scale = u_std / x_std
    action_to_state_bias = (u_mean - x_mean) / x_std
    features, labels = [], []
    for i in train_indices:
        x = (episodes[i] - x_mean) / x_std
        u = (actions[i] - u_mean) / u_std
        velocity = x[1:-1] - x[:-2]
        command_as_state = (
            action_to_state_scale * u[1:] + action_to_state_bias
        )
        command_error = command_as_state - x[1:-1]
        features.append(np.stack([velocity, command_error], axis=-1))
        labels.append(x[2:] - x[1:-1])
    phi, y = np.concatenate(features), np.concatenate(labels)
    coefficients = np.stack([
        np.linalg.lstsq(phi[:, joint], y[:, joint], rcond=None)[0]
        for joint in range(y.shape[1])
    ])
    # Physical stability constraints prevent an unconstrained regression from
    # producing an unstable comparison plant.
    momentum = np.clip(coefficients[:, 0], 0.0, 0.99)
    gain = np.clip(coefficients[:, 1], 1e-4, 1.0)
    return (SecondOrderServo(
        momentum, gain, action_to_state_scale, action_to_state_bias
    ), momentum, gain)


def simulate(controller, plant, initial_history, target, steps):
    history = jnp.asarray(initial_history)
    previous = jnp.zeros((plant.history_len - 1, history.shape[-1]))
    states, controls, solve_times = [], [], []
    for _ in range(steps):
        started = time.perf_counter()
        action, _ = controller.command(history, previous)
        jax.block_until_ready(action)
        solve_times.append(time.perf_counter() - started)
        next_state = plant.predict_next(history, action)
        history = jnp.concatenate([history[1:], next_state[None]], axis=0)
        previous = jnp.concatenate([previous[1:], action[None]], axis=0)
        states.append(next_state); controls.append(action)
    states, controls = np.asarray(jnp.stack(states)), np.asarray(jnp.stack(controls))
    return states, controls, np.asarray(solve_times)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    train_i, _, test_i = split_episode_indices(len(episodes), args.seed)
    stats = np.load(args.normalization)
    x_mean, x_std, u_mean, u_std = (stats[k] for k in ("x_mean", "x_std", "u_mean", "u_std"))
    plant, momentum, gain = fit_servo(
        episodes, actions, train_i, x_mean, x_std, u_mean, u_std
    )
    learned = GRULatentSSMFlow(
        state_dim, control_dim, args.history, args.hidden_dim, jr.PRNGKey(args.seed + 1),
        predict_delta=True, n_coupling=args.blocks, hidden_size=args.width,
        n_hidden=args.hidden_layers, flow_type="realnvp", max_rho=args.max_rho,
    )
    learned = eqx.tree_deserialise_leaves(args.checkpoint, learned)
    train_u = np.concatenate([(actions[i] - u_mean) / u_std for i in train_i])
    low, high = jnp.asarray(np.quantile(train_u, 0.005, axis=0)), jnp.asarray(
        np.quantile(train_u, 0.995, axis=0)
    )
    rows, examples = [], []
    for index in test_i[:args.tasks]:
        x = (episodes[index] - x_mean) / x_std
        start = args.start
        target = jnp.asarray(x[start + args.task_steps])
        initial = jnp.asarray(x[start - args.history + 1:start + 1])
        def cost(state, action, step):
            del step
            return jnp.mean((state - target) ** 2) + args.action_weight * jnp.mean(action ** 2)
        config = GradientMPCConfig(
            horizon=args.horizon, iterations=args.iterations,
            learning_rate=args.learning_rate, momentum=args.momentum,
        )
        learned_mpc = GradientMPC(learned, low, high, cost, config)
        physical_mpc = GradientMPC(plant, low, high, cost, config)
        probe_action = jnp.asarray((actions[index][start] - u_mean) / u_std)
        learned_jacobian = jax.jacrev(
            lambda action: learned.predict_next(initial, action)
        )(probe_action)
        physical_jacobian = jax.jacrev(
            lambda action: plant.predict_next(initial, action)
        )(probe_action)
        learned_x, learned_u, learned_time = simulate(
            learned_mpc, plant, initial, target, args.task_steps
        )
        physical_x, physical_u, physical_time = simulate(
            physical_mpc, plant, initial, target, args.task_steps
        )
        for name, states, controls, timing, jacobian in (
            ("learned_flow", learned_x, learned_u, learned_time, learned_jacobian),
            ("physical_servo", physical_x, physical_u, physical_time, physical_jacobian),
        ):
            native_error = (states - np.asarray(target)) * x_std
            rows.append({
                "episode_id": int(episode_ids[index]), "controller": name,
                "tracking_rmse": float(np.sqrt(np.mean(native_error ** 2))),
                "terminal_rmse": float(np.sqrt(np.mean(native_error[-1] ** 2))),
                "action_rms_normalized": float(np.sqrt(np.mean(controls ** 2))),
                "action_jacobian_frobenius": float(jnp.linalg.norm(jacobian)),
                "median_solve_ms": float(1000 * np.median(timing[1:])),
            })
        examples.append((int(episode_ids[index]), learned_x, physical_x, np.asarray(target)))

    with (args.output / "mpc_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for controller in ("learned_flow", "physical_servo"):
        selected = [r for r in rows if r["controller"] == controller]
        summary[controller] = {key: float(np.mean([r[key] for r in selected]))
                               for key in ("tracking_rmse", "terminal_rmse",
                                           "action_rms_normalized",
                                           "action_jacobian_frobenius",
                                           "median_solve_ms")}
    summary["configuration"] = {
        "offline_simulation": True, "plant": "fitted_second_order_servo",
        "horizon_steps": args.horizon, "horizon_seconds": args.horizon / 50,
        "task_steps": args.task_steps, "tasks": min(args.tasks, len(test_i)),
        "iterations": args.iterations, "servo_momentum": momentum.tolist(),
        "servo_command_gain": gain.tolist(),
    }
    with (args.output / "mpc_comparison.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    episode_id, learned_x, physical_x, target = examples[0]
    time_axis = np.arange(1, args.task_steps + 1) / 50
    fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
    for joint, axis in enumerate(axes.flat):
        axis.plot(time_axis, learned_x[:, joint] * x_std[joint] + x_mean[joint], label="flow MPC")
        axis.plot(time_axis, physical_x[:, joint] * x_std[joint] + x_mean[joint], label="physical MPC")
        axis.axhline(target[joint] * x_std[joint] + x_mean[joint], color="black", ls="--", label="target")
        axis.set_title(f"joint {joint}"); axis.grid(alpha=.2)
    axes[0, 0].legend(); fig.suptitle(f"Short-horizon gradient MPC — held-out episode {episode_id}")
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(args.output / "mpc_trajectories.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path("results/results_optimal_control/aloha_mobile_cabinet")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=base / "aloha_realnvp_noise0p05_h8_seg16_seed0.eqx")
    p.add_argument("--normalization", type=Path,
                   default=base / "aloha_realnvp_noise0p05_h8_seg16_seed0_normalization.npz")
    p.add_argument("--output", type=Path, default=base / "mpc_comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--action-weight", type=float, default=1e-3)
    p.add_argument("--start", type=int, default=400)
    p.add_argument("--task-steps", type=int, default=50)
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--hidden-layers", type=int, default=1)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--max-rho", type=float, default=0.8)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
