"""Open-loop Jackal dynamics evaluation and closed-loop CEM comparison.

The learned LatentSSMFlow-CEM and exact-model stochastic CEM control identical
noisy exact plants. Results are saved as JSON, CSV, NPZ, and a trajectory plot.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control.gru_latent_ssm_flow import GRULatentSSMFlow
from .model import (
    ExactJackalModel, JackalConfig, generate_identification_data,
    minimum_clearance, nominal_step, obstacle_cost, state_from_pose,
    yaw_from_state,
)
from bayesinf.applications.optimal_control.sampling_mpc import CEMStochasticMPC, SamplingMPCConfig


def load_model(checkpoint, args, *, history_control_conditioned=False):
    template = GRULatentSSMFlow(
        6, 2, args.history, args.hidden_dim, jr.PRNGKey(1),
        predict_delta=True, n_coupling=args.blocks, hidden_size=args.width,
        n_hidden=args.hidden_layers, max_rho=args.max_rho,
        history_control_conditioned=history_control_conditioned,
    )
    return eqx.tree_deserialise_leaves(checkpoint, template)


def angle_error(predicted, target):
    difference = yaw_from_state(predicted) - yaw_from_state(target)
    return jnp.arctan2(jnp.sin(difference), jnp.cos(difference))


def open_loop_evaluation(model, args):
    states, actions = generate_identification_data(
        jr.PRNGKey(args.evaluation_seed), args.open_loop_episodes,
        args.open_loop_steps, process_noise=args.process_noise,
    )
    exact = ExactJackalModel(process_noise=args.process_noise)
    learned_predictions, exact_predictions, targets, particle_predictions = [], [], [], []
    for episode in range(args.open_loop_episodes):
        history = states[episode, :args.history]
        future_actions = actions[episode, args.history - 1:]
        target = states[episode, args.history:]
        learned_predictions.append(model.rollout_deterministic_history(
            history, actions[episode, :args.history - 1], future_actions
        ))
        exact_predictions.append(exact.rollout_deterministic(history[-1], future_actions))
        particle_predictions.append(model.sample_rollout(
            history, actions[episode, :args.history - 1], future_actions,
            jr.fold_in(jr.PRNGKey(args.evaluation_seed + 1), episode),
            args.open_loop_particles,
        ))
        targets.append(target)
    learned = jnp.stack(learned_predictions)
    exact_prediction = jnp.stack(exact_predictions)
    particles = jnp.stack(particle_predictions)
    target = jnp.stack(targets)

    def metrics(prediction):
        position_error = prediction[..., :2] - target[..., :2]
        velocity_error = prediction[..., 4:] - target[..., 4:]
        return {
            "position_rmse_m": float(jnp.sqrt(jnp.mean(position_error ** 2))),
            "yaw_rmse_rad": float(jnp.sqrt(jnp.mean(angle_error(prediction, target) ** 2))),
            "velocity_rmse": float(jnp.sqrt(jnp.mean(velocity_error ** 2))),
            "final_position_rmse_m": float(jnp.sqrt(jnp.mean(
                position_error[:, -1] ** 2
            ))),
        }

    lower = jnp.quantile(particles[..., :2], 0.05, axis=1)
    upper = jnp.quantile(particles[..., :2], 0.95, axis=1)
    coverage = jnp.mean((target[..., :2] >= lower) & (target[..., :2] <= upper))
    return {
        "learned_latent_ssm_flow": metrics(learned),
        "exact_deterministic": metrics(exact_prediction),
        "learned_position_90pct_coverage": float(coverage),
    }


def plant_step(state, action, key, config, process_noise):
    state = nominal_step(state, action, config)
    return state.at[4:].add(process_noise * jr.normal(key, (2,)))


def run_episode(controller, model_name, episode, args, config, obstacles, goal):
    del model_name
    state = state_from_pose(0.0, -0.35 + 0.7 * (episode % 2),
                            0.05 * (episode - (args.closed_loop_episodes - 1) / 2))
    history = jnp.repeat(state[None], args.history, axis=0)
    prior_actions = jnp.zeros((args.history - 1, 2))
    trajectory, controls = [state], []
    elapsed = 0.0
    episode_key = jr.fold_in(jr.PRNGKey(args.evaluation_seed + 100), episode)
    controller.reset()
    for step in range(args.closed_loop_steps):
        before = time.perf_counter()
        action, _ = controller.command(
            history, prior_actions, jr.fold_in(episode_key, 2 * step)
        )
        # Synchronize before timing asynchronous JAX execution.
        action.block_until_ready()
        elapsed += time.perf_counter() - before
        state = plant_step(
            state, action, jr.fold_in(episode_key, 2 * step + 1),
            config, args.process_noise,
        )
        trajectory.append(state)
        controls.append(action)
        history = jnp.concatenate([history[1:], state[None]])
        prior_actions = jnp.concatenate([prior_actions[1:], action[None]])
        if float(jnp.linalg.norm(state[:2] - goal)) <= args.goal_tolerance:
            break
    trajectory = jnp.stack(trajectory)
    controls = jnp.stack(controls) if controls else jnp.zeros((0, 2))
    clearance = float(minimum_clearance(trajectory, obstacles))
    goal_error = float(jnp.linalg.norm(trajectory[-1, :2] - goal))
    return {
        "success": goal_error <= args.goal_tolerance,
        "collision": clearance < config.robot_radius,
        "safety_violation": clearance < config.robot_radius + config.safety_margin,
        "goal_error_m": goal_error,
        "minimum_clearance_m": clearance,
        "steps": len(controls),
        "path_length_m": float(jnp.sum(jnp.linalg.norm(
            jnp.diff(trajectory[:, :2], axis=0), axis=-1
        ))),
        "mean_command_time_s": elapsed / max(1, len(controls)),
    }, np.asarray(trajectory), np.asarray(controls)


def aggregate(rows):
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "safety_violation_rate": float(np.mean(
            [row["safety_violation"] for row in rows]
        )),
        "mean_goal_error_m": float(np.mean([row["goal_error_m"] for row in rows])),
        "mean_minimum_clearance_m": float(np.mean(
            [row["minimum_clearance_m"] for row in rows]
        )),
        "mean_path_length_m": float(np.mean([row["path_length_m"] for row in rows])),
        "mean_command_time_s": float(np.mean(
            [row["mean_command_time_s"] for row in rows]
        )),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args)
    config = JackalConfig()
    goal = jnp.asarray([4.0, 0.0])
    obstacles = jnp.asarray([[2.0, 0.0, 0.42], [3.0, 0.7, 0.28]])
    cost = obstacle_cost(goal, obstacles, config)
    mpc_config = SamplingMPCConfig(
        horizon=args.horizon, num_candidates=args.candidates,
        num_particles=args.particles, iterations=args.cem_iterations,
        elite_fraction=args.elite_fraction, initial_std=args.initial_std,
        min_std=0.02, risk_weight=args.risk_weight,
        smoothness_weight=args.smoothness_weight,
        smoothness_scale=jnp.asarray([1.0, 1.5]),
    )
    bounds = (jnp.asarray([0.0, -1.5]), jnp.asarray([1.0, 1.5]))
    controllers = {
        "learned_latent_ssm_flow_cem": CEMStochasticMPC(
            model, *bounds, cost, mpc_config
        ),
        "exact_simulator_cem": CEMStochasticMPC(
            ExactJackalModel(config, args.process_noise),
            *bounds, cost, mpc_config,
        ),
    }
    open_loop = None if args.skip_open_loop else open_loop_evaluation(model, args)
    # Exclude one-time JIT compilation from the per-command runtime reported
    # below.  This is especially important when comparing accelerator runs.
    warm_history = jnp.repeat(state_from_pose(0, 0, 0)[None], args.history, 0)
    warm_actions = jnp.zeros((args.history - 1, 2))
    compile_times = {}
    for name, controller in controllers.items():
        started = time.perf_counter()
        command, _ = controller.command(
            warm_history, warm_actions, jr.PRNGKey(999)
        )
        command.block_until_ready()
        compile_times[name] = time.perf_counter() - started
        controller.reset()
    all_rows, trajectories, controls = [], {}, {}
    for name, controller in controllers.items():
        rows, model_trajectories, model_controls = [], [], []
        for episode in range(args.closed_loop_episodes):
            row, trajectory, episode_controls = run_episode(
                controller, name, episode, args, config, obstacles, goal
            )
            row.update(model=name, episode=episode)
            rows.append(row)
            model_trajectories.append(trajectory)
            model_controls.append(episode_controls)
            print(name, episode, row, flush=True)
        all_rows.extend(rows)
        trajectories[name] = model_trajectories
        controls[name] = model_controls

    summary = {
        "open_loop": open_loop,
        "closed_loop": {
            name: aggregate([row for row in all_rows if row["model"] == name])
            for name in controllers
        },
        "jit_compile_time_s": compile_times,
        "compute_device": {
            "platform": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "settings": {
            "horizon": args.horizon, "candidates": args.candidates,
            "particles": args.particles, "cem_iterations": args.cem_iterations,
            "risk_weight": args.risk_weight,
            "goal": np.asarray(goal).tolist(),
            "obstacles": np.asarray(obstacles).tolist(),
        },
    }
    (args.output / "jackal_cem_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (args.output / "jackal_cem_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    np.savez(
        args.output / "jackal_cem_trajectories.npz",
        **{f"{name}_episode_{index}": trajectory
           for name, values in trajectories.items()
           for index, trajectory in enumerate(values)},
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = {"learned_latent_ssm_flow_cem": "tab:blue",
              "exact_simulator_cem": "tab:orange"}
    for name, values in trajectories.items():
        for index, trajectory in enumerate(values):
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=colors[name],
                      alpha=0.35, label=name if index == 0 else None)
    for x, y, radius in np.asarray(obstacles):
        axis.add_patch(plt.Circle((x, y), radius + config.robot_radius,
                                 color="tab:red", alpha=0.25))
    axis.scatter(*goal, marker="*", s=180, color="green", label="goal")
    axis.set_aspect("equal"); axis.grid(alpha=0.3); axis.legend()
    axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]")
    figure.tight_layout()
    figure.savefig(args.output / "jackal_cem_trajectories.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--evaluation-seed", type=int, default=2026)
    result.add_argument("--history", type=int, default=5)
    result.add_argument("--hidden-dim", type=int, default=64)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--hidden-layers", type=int, default=2)
    result.add_argument("--blocks", type=int, default=4)
    result.add_argument("--max-rho", type=float, default=0.9)
    result.add_argument("--process-noise", type=float, default=0.01)
    result.add_argument("--open-loop-episodes", type=int, default=32)
    result.add_argument("--open-loop-steps", type=int, default=100)
    result.add_argument("--open-loop-particles", type=int, default=64)
    result.add_argument("--skip-open-loop", action="store_true")
    result.add_argument("--closed-loop-episodes", type=int, default=10)
    result.add_argument("--closed-loop-steps", type=int, default=100)
    result.add_argument("--goal-tolerance", type=float, default=0.3)
    result.add_argument("--horizon", type=int, default=20)
    result.add_argument("--candidates", type=int, default=256)
    result.add_argument("--particles", type=int, default=8)
    result.add_argument("--cem-iterations", type=int, default=5)
    result.add_argument("--elite-fraction", type=float, default=0.1)
    result.add_argument("--initial-std", type=float, default=0.4)
    result.add_argument("--risk-weight", type=float, default=1.0)
    result.add_argument("--smoothness-weight", type=float, default=0.05)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
