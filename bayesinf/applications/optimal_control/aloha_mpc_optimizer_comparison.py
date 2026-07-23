"""Compare flow/physical dynamics with MPPI/gradient MPC on shared plant."""

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

from .aloha_gradient_mpc_comparison import fit_servo
from .gradient_based_mpc import GradientMPC, GradientMPCConfig
from .gru_latent_ssm_flow import GRULatentSSMFlow
from .sampling_mpc import MPPIController, SamplingMPCConfig
from .train_aloha_mobile_dynamics import load_episodes, split_episode_indices


def simulate(controller, optimizer, plant, initial_history, steps, seed):
    history = jnp.asarray(initial_history)
    previous = jnp.zeros((plant.history_len - 1, history.shape[-1]))
    states, controls, timings, ess = [], [], [], []
    for step in range(steps):
        started = time.perf_counter()
        if optimizer == "mppi":
            action, _ = controller.command(
                history, previous, jr.fold_in(jr.PRNGKey(seed), step)
            )
        else:
            action, _ = controller.command(history, previous)
        jax.block_until_ready(action)
        timings.append(time.perf_counter() - started)
        if optimizer == "mppi":
            ess.append(controller.last_effective_sample_size)
        next_state = plant.predict_next(history, action)
        history = jnp.concatenate([history[1:], next_state[None]], axis=0)
        previous = jnp.concatenate([previous[1:], action[None]], axis=0)
        states.append(next_state); controls.append(action)
    return (np.asarray(jnp.stack(states)), np.asarray(jnp.stack(controls)),
            np.asarray(timings), np.asarray(ess))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    train_i, _, test_i = split_episode_indices(len(episodes), args.seed)
    stats = np.load(args.normalization)
    x_mean, x_std, u_mean, u_std = (
        stats[k] for k in ("x_mean", "x_std", "u_mean", "u_std")
    )
    plant, _, _ = fit_servo(
        episodes, actions, train_i, x_mean, x_std, u_mean, u_std
    )
    flow = GRULatentSSMFlow(
        state_dim, control_dim, args.history, args.hidden_dim,
        jr.PRNGKey(args.seed + 1), predict_delta=True,
        n_coupling=args.blocks, hidden_size=args.width,
        n_hidden=args.hidden_layers, flow_type="realnvp", max_rho=args.max_rho,
    )
    flow = eqx.tree_deserialise_leaves(args.checkpoint, flow)
    train_u = np.concatenate([(actions[i] - u_mean) / u_std for i in train_i])
    low = jnp.asarray(np.quantile(train_u, 0.005, axis=0))
    high = jnp.asarray(np.quantile(train_u, 0.995, axis=0))
    rows, first_trajectories = [], {}

    for task_number, index in enumerate(test_i[:args.tasks]):
        x = (episodes[index] - x_mean) / x_std
        start = args.start
        target = jnp.asarray(x[start + args.task_steps])
        initial = jnp.asarray(x[start - args.history + 1:start + 1])

        def cost(state, action, horizon_step):
            terminal_multiplier = jnp.where(
                horizon_step == args.horizon - 1, args.terminal_weight, 1.0
            )
            return (terminal_multiplier * jnp.mean((state - target) ** 2)
                    + args.action_weight * jnp.mean(action ** 2))

        gradient_config = GradientMPCConfig(
            horizon=args.horizon, iterations=args.gradient_iterations,
            learning_rate=args.learning_rate, momentum=args.momentum,
        )
        mppi_config = SamplingMPCConfig(
            horizon=args.horizon, num_candidates=args.candidates,
            num_particles=args.particles, iterations=args.mppi_iterations,
            initial_std=args.exploration_std, temperature=args.temperature,
            risk_weight=args.risk_weight, normalize_costs=True,
        )
        controllers = {
            "flow_gradient": (GradientMPC(flow, low, high, cost, gradient_config), "gradient"),
            "physical_gradient": (GradientMPC(plant, low, high, cost, gradient_config), "gradient"),
            "flow_mppi": (MPPIController(flow, low, high, cost, mppi_config), "mppi"),
            "physical_mppi": (MPPIController(plant, low, high, cost, mppi_config), "mppi"),
        }
        for offset, (name, (controller, optimizer)) in enumerate(controllers.items()):
            states, controls, timings, ess = simulate(
                controller, optimizer, plant, initial, args.task_steps,
                args.seed + 1000 * task_number + offset,
            )
            native_error = (states - np.asarray(target)) * x_std
            rows.append({
                "episode_id": int(episode_ids[index]), "controller": name,
                "tracking_rmse": float(np.sqrt(np.mean(native_error ** 2))),
                "terminal_rmse": float(np.sqrt(np.mean(native_error[-1] ** 2))),
                "action_rms_normalized": float(np.sqrt(np.mean(controls ** 2))),
                "median_solve_ms": float(1000 * np.median(timings[1:])),
                "mean_effective_sample_size": (
                    float(np.mean(ess)) if len(ess) else ""
                ),
            })
            if task_number == 0:
                first_trajectories[name] = states

    with (args.output / "optimizer_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name in ("flow_gradient", "flow_mppi", "physical_gradient", "physical_mppi"):
        selected = [row for row in rows if row["controller"] == name]
        summary[name] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("tracking_rmse", "terminal_rmse",
                        "action_rms_normalized", "median_solve_ms")
        }
        if "mppi" in name:
            summary[name]["mean_effective_sample_size"] = float(np.mean([
                row["mean_effective_sample_size"] for row in selected
            ]))
    summary["configuration"] = {
        "shared_plant": "fitted_second_order_servo",
        "horizon_steps": args.horizon,
        "horizon_seconds": args.horizon / 50.0,
        "task_steps": args.task_steps, "tasks": min(args.tasks, len(test_i)),
        "candidates": args.candidates, "particles": args.particles,
        "mppi_iterations": args.mppi_iterations,
        "gradient_iterations": args.gradient_iterations,
        "temperature": args.temperature, "risk_weight": args.risk_weight,
        "cost_std_normalization": True,
    }
    with (args.output / "optimizer_comparison.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    names = list(first_trajectories)
    colors = dict(zip(names, ("C0", "C1", "C2", "C3")))
    target_native = np.asarray(target) * x_std + x_mean
    times = np.arange(1, args.task_steps + 1) / 50.0
    fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
    for joint, axis in enumerate(axes.flat):
        for name in names:
            values = first_trajectories[name][:, joint] * x_std[joint] + x_mean[joint]
            axis.plot(times, values, color=colors[name], label=name)
        axis.axhline(target_native[joint], color="black", ls="--", label="target")
        axis.set_title(f"joint {joint}"); axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"MPC optimizer comparison — episode {episode_ids[test_i[0]]}")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.output / "optimizer_trajectories.png", dpi=180)
    plt.close(fig)

    metric_names = ("tracking_rmse", "terminal_rmse", "median_solve_ms")
    controller_names = [name for name in summary if name != "configuration"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, metric_names):
        axis.bar(controller_names,
                 [summary[name][metric] for name in controller_names])
        axis.set_title(metric); axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(args.output / "optimizer_metrics.png", dpi=180)
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
    p.add_argument("--output", type=Path, default=base / "mpc_optimizer_comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--task-steps", type=int, default=50)
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--start", type=int, default=400)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--particles", type=int, default=4)
    p.add_argument("--mppi-iterations", type=int, default=3)
    p.add_argument("--gradient-iterations", type=int, default=25)
    p.add_argument("--exploration-std", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--risk-weight", type=float, default=0.1)
    p.add_argument("--terminal-weight", type=float, default=5.0)
    p.add_argument("--action-weight", type=float, default=1e-3)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--hidden-layers", type=int, default=1)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--max-rho", type=float, default=0.8)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
