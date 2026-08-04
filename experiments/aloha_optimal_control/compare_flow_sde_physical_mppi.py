"""Compare solver-integrated FlowSDE MPPI with fitted-physical-model MPPI.

Both controllers use identical MPPI hyperparameters, normalized action bounds,
costs, goals, and the same fitted servo execution plant. The FlowSDE candidate
paths use the solver recorded in its training configuration.
"""

import argparse
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from .data import load_episodes, split_episode_indices
from .physical_model import fit_servo
from bayesinf.applications.optimal_control.flow_sde_mppi import FlowSDEMPPIController
from bayesinf.applications.optimal_control.sampling_mpc import MPPIController, SamplingMPCConfig
from .train_flow_sde import build_model


def execute(controller, controller_type, plant, initial_history, steps,
            episode_start_step, seed):
    history = jnp.asarray(initial_history)
    previous = jnp.zeros((plant.history_len - 1, history.shape[-1]))
    states, controls, solve_times, ess = [], [], [], []
    for step in range(steps):
        key = jr.fold_in(jr.PRNGKey(seed), step)
        started = time.perf_counter()
        if controller_type == "flow_sde":
            action, _ = controller.command(
                history[-1], key, start_step=episode_start_step + step
            )
        else:
            action, _ = controller.command(history, previous, key)
        jax.block_until_ready(action)
        solve_times.append(time.perf_counter() - started)
        ess.append(controller.last_effective_sample_size)
        next_state = plant.predict_next(history, action)
        history = jnp.concatenate([history[1:], next_state[None]], axis=0)
        previous = jnp.concatenate([previous[1:], action[None]], axis=0)
        states.append(next_state); controls.append(action)
    return (np.asarray(jnp.stack(states)), np.asarray(jnp.stack(controls)),
            np.asarray(solve_times), np.asarray(ess))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with args.flow_config.open() as handle:
        flow_config = json.load(handle)
    solver = flow_config["solver"]
    if args.solver is not None and args.solver != solver:
        raise ValueError(
            f"requested solver {args.solver!r} differs from training solver {solver!r}"
        )
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(args.data)
    train_i, _, test_i = split_episode_indices(len(episodes), flow_config["seed"])
    stats = np.load(flow_config["normalization"])
    x_mean, x_std, u_mean, u_std = (
        stats[name] for name in ("x_mean", "x_std", "u_mean", "u_std")
    )
    flow = build_model(SimpleNamespace(**flow_config), state_dim, control_dim)
    flow = eqx.tree_deserialise_leaves(flow_config["checkpoint"], flow)
    plant, momentum, gain = fit_servo(
        episodes, actions, train_i, x_mean, x_std, u_mean, u_std,
        history_len=args.history,
    )
    train_controls = np.concatenate([
        (actions[index] - u_mean) / u_std for index in train_i
    ])
    low = jnp.asarray(np.quantile(train_controls, args.action_quantile, axis=0))
    high = jnp.asarray(np.quantile(train_controls, 1 - args.action_quantile, axis=0))
    mppi_config = SamplingMPCConfig(
        horizon=args.horizon, num_candidates=args.candidates,
        num_particles=args.particles, iterations=args.iterations,
        initial_std=args.exploration_std, temperature=args.temperature,
        risk_weight=args.risk_weight, normalize_costs=True,
    )

    rows, examples = [], {}
    for task_number, index in enumerate(test_i[:args.tasks]):
        normalized_states = (episodes[index] - x_mean) / x_std
        start = args.start
        if start < args.history - 1 or start + args.task_steps >= len(normalized_states):
            raise ValueError("start/history/task configuration exceeds episode")
        goal = jnp.asarray(normalized_states[start + args.task_steps])
        initial_history = jnp.asarray(
            normalized_states[start - args.history + 1:start + 1]
        )

        def stage_cost(state, action, horizon_step):
            terminal_scale = jnp.where(
                horizon_step == args.horizon - 1, args.terminal_weight, 1.0
            )
            return (terminal_scale * jnp.mean((state - goal) ** 2)
                    + args.action_weight * jnp.mean(action ** 2))

        controllers = {
            "flow_sde_mppi": (
                FlowSDEMPPIController(
                    flow, low, high, stage_cost, mppi_config, solver_name=solver
                ), "flow_sde"
            ),
            "physical_mppi": (
                MPPIController(plant, low, high, stage_cost, mppi_config), "physical"
            ),
        }
        for name, (controller, controller_type) in controllers.items():
            # Use the same task seed for identical MPPI control perturbations.
            states, controls, timings, effective_size = execute(
                controller, controller_type, plant, initial_history,
                args.task_steps, start, args.seed + 1000 * task_number,
            )
            native_error = (states - np.asarray(goal)) * x_std
            row = {
                "episode_id": int(episode_ids[index]), "controller": name,
                "tracking_rmse": float(np.sqrt(np.mean(native_error ** 2))),
                "terminal_rmse": float(np.sqrt(np.mean(native_error[-1] ** 2))),
                "action_rms_normalized": float(np.sqrt(np.mean(controls ** 2))),
                "median_solve_ms": float(1000 * np.median(timings[1:])),
                "mean_effective_sample_size": float(np.mean(effective_size)),
            }
            rows.append(row)
            if task_number == 0:
                examples[name] = states

    with (args.output / "mppi_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name in ("flow_sde_mppi", "physical_mppi"):
        selected = [row for row in rows if row["controller"] == name]
        summary[name] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in ("tracking_rmse", "terminal_rmse",
                           "action_rms_normalized", "median_solve_ms",
                           "mean_effective_sample_size")
        }
    summary["configuration"] = {
        "comparison": "offline common fitted-servo execution plant",
        "flow_solver": solver, "training_segment_length": flow_config["segment_length"],
        "horizon_steps": args.horizon, "horizon_seconds": args.horizon * flow_config["dt"],
        "task_steps": args.task_steps, "tasks": min(args.tasks, len(test_i)),
        "candidates": args.candidates, "particles": args.particles,
        "iterations": args.iterations, "temperature": args.temperature,
        "risk_weight": args.risk_weight, "same_control_perturbation_seeds": True,
        "servo_momentum": momentum.tolist(), "servo_command_gain": gain.tolist(),
    }
    with (args.output / "mppi_comparison.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    time_axis = np.arange(1, args.task_steps + 1) * flow_config["dt"]
    goal_native = np.asarray(goal) * x_std + x_mean
    fig, axes = plt.subplots(7, 2, figsize=(15, 18), sharex=True)
    for joint, axis in enumerate(axes.flat):
        for name, states in examples.items():
            axis.plot(time_axis, states[:, joint] * x_std[joint] + x_mean[joint],
                      label=name)
        axis.axhline(goal_native[joint], color="black", ls="--", label="goal")
        axis.set_title(f"joint {joint}"); axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"FlowSDE versus physical MPPI — episode {episode_ids[test_i[0]]}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output / "mppi_trajectories.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path("results/results_optimal_control/aloha_mobile_cabinet")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--flow-config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=base / "flow_sde_mppi_comparison")
    p.add_argument("--solver", choices=("euler", "heun", "reversible_heun",
                   "midpoint", "milstein"), default=None,
                   help="Optional assertion; must equal the training solver.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--task-steps", type=int, default=50)
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--start", type=int, default=400)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--particles", type=int, default=4)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--exploration-std", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--risk-weight", type=float, default=0.1)
    p.add_argument("--terminal-weight", type=float, default=5.0)
    p.add_argument("--action-weight", type=float, default=1e-3)
    p.add_argument("--action-quantile", type=float, default=0.005)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
