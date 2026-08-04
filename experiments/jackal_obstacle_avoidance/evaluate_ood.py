"""Evaluate learned and nominal-model CEM on unseen Jackal plant mismatch."""

import argparse
import csv
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import angle_error, load_model
from .model import (
    ExactJackalModel, HighFidelityJackalConfig, delayed_high_fidelity_step,
    minimum_clearance, obstacle_cost, state_from_pose,
)
from bayesinf.applications.optimal_control.sampling_mpc import CEMStochasticMPC, SamplingMPCConfig


def scenarios():
    """Held-out parameters; each exceeds or differs from the training range."""
    return {
        "long_delay": HighFidelityJackalConfig(
            command_delay_steps=4, base_traction=0.82,
            terrain_amplitude=0.10, terrain_phase=2.2,
        ),
        "heavy_payload": HighFidelityJackalConfig(
            payload_factor=1.15, motor_time_constant=0.38,
            base_traction=0.80, terrain_phase=0.4,
        ),
        "crosswind": HighFidelityJackalConfig(
            disturbance_y=0.11, disturbance_yaw=0.045,
            base_traction=0.82, terrain_phase=2.8,
        ),
        "combined_extreme": HighFidelityJackalConfig(
            payload_factor=0.95, command_delay_steps=3,
            base_traction=0.48, terrain_amplitude=0.20,
            left_wheel_slip=0.19, right_wheel_slip=0.11,
            lateral_slip=0.12, disturbance_y=-0.08,
            disturbance_yaw=-0.035, terrain_phase=1.7,
        ),
    }


def scenario_rollout(config, actions, initial_state, key):
    state = initial_state
    queue = jnp.zeros((max(1, config.command_delay_steps), 2))
    values = [state]
    for action, step_key in zip(actions, jr.split(key, len(actions))):
        state, queue = delayed_high_fidelity_step(
            state, queue, action, step_key, config
        )
        values.append(state)
    return jnp.stack(values)


def open_loop(model, args):
    nominal = ExactJackalModel(process_noise=0.0)
    rows = {}
    for scenario_index, (name, config) in enumerate(scenarios().items()):
        learned_values, nominal_values, targets, particles = [], [], [], []
        for episode in range(args.open_loop_episodes):
            key = jr.fold_in(
                jr.PRNGKey(args.evaluation_seed + 10 * scenario_index), episode
            )
            command_key, plant_key, particle_key = jr.split(key, 3)
            raw = jr.uniform(command_key, (args.open_loop_steps, 2),
                             minval=-1.0, maxval=1.0)
            actions = raw.at[:, 0].set(0.05 + 0.9 * (raw[:, 0] + 1) / 2)
            actions = actions.at[:, 1].multiply(1.35)
            initial = state_from_pose(0, 0, 0.15 * (episode - 1))
            truth = scenario_rollout(config, actions, initial, plant_key)
            history = truth[:args.history]
            future_actions = actions[args.history - 1:]
            target = truth[args.history:]
            learned_values.append(model.rollout_deterministic_history(
                history, actions[:args.history - 1], future_actions
            ))
            nominal_values.append(nominal.rollout_deterministic(
                history[-1], future_actions
            ))
            particles.append(model.sample_rollout(
                history, actions[:args.history - 1], future_actions,
                particle_key, args.open_loop_particles,
            ))
            targets.append(target)
        learned = jnp.stack(learned_values)
        nominal_prediction = jnp.stack(nominal_values)
        target = jnp.stack(targets)
        particle_values = jnp.stack(particles)

        def calculate(prediction):
            return {
                "position_rmse_m": float(jnp.sqrt(jnp.mean(
                    (prediction[..., :2] - target[..., :2]) ** 2
                ))),
                "yaw_rmse_rad": float(jnp.sqrt(jnp.mean(
                    angle_error(prediction, target) ** 2
                ))),
                "velocity_rmse": float(jnp.sqrt(jnp.mean(
                    (prediction[..., 4:] - target[..., 4:]) ** 2
                ))),
                "final_position_rmse_m": float(jnp.sqrt(jnp.mean(
                    (prediction[:, -1, :2] - target[:, -1, :2]) ** 2
                ))),
            }

        lower = jnp.quantile(particle_values[..., :2], 0.05, axis=1)
        upper = jnp.quantile(particle_values[..., :2], 0.95, axis=1)
        rows[name] = {
            "latent_ssm_flow": calculate(learned),
            "nominal_unicycle": calculate(nominal_prediction),
            "learned_position_90pct_coverage": float(jnp.mean(
                (target[..., :2] >= lower) & (target[..., :2] <= upper)
            )),
            "horizon_position_rmse_m": {
                "latent_ssm_flow": np.asarray(jnp.sqrt(jnp.mean(
                    (learned[..., :2] - target[..., :2]) ** 2,
                    axis=(0, 2),
                ))).tolist(),
                "nominal_unicycle": np.asarray(jnp.sqrt(jnp.mean(
                    (nominal_prediction[..., :2] - target[..., :2]) ** 2,
                    axis=(0, 2),
                ))).tolist(),
            },
        }
    return rows


def dynamics_timing(model, args):
    """JIT-warmed throughput for matched deterministic and particle rollouts."""
    config = scenarios()["combined_extreme"]
    nominal = ExactJackalModel(process_noise=args.nominal_process_noise)
    histories, control_histories, future_controls = [], [], []
    for index in range(args.timing_batch_size):
        key = jr.fold_in(jr.PRNGKey(args.evaluation_seed + 5000), index)
        action_key, plant_key = jr.split(key)
        actions = jr.uniform(
            action_key, (args.history - 1 + args.timing_horizon, 2),
            minval=-1.0, maxval=1.0,
        )
        actions = actions.at[:, 0].set(0.05 + 0.9 * (actions[:, 0] + 1) / 2)
        actions = actions.at[:, 1].multiply(1.35)
        truth = scenario_rollout(
            config, actions, state_from_pose(0, 0, 0), plant_key
        )
        histories.append(truth[:args.history])
        control_histories.append(actions[:args.history - 1])
        future_controls.append(actions[args.history - 1:])
    histories = jnp.stack(histories)
    control_histories = jnp.stack(control_histories)
    future_controls = jnp.stack(future_controls)

    learned_rollout = jax.jit(jax.vmap(
        model.rollout_deterministic_history
    ))
    nominal_rollout = jax.jit(jax.vmap(
        lambda history, controls, actions: nominal.rollout_deterministic(
            history[-1], actions
        )
    ))
    particle_keys = jr.split(
        jr.PRNGKey(args.evaluation_seed + 6000), args.timing_batch_size
    )
    learned_particles = jax.jit(jax.vmap(
        lambda history, controls, actions, key: model.sample_rollout(
            history, controls, actions, key, args.timing_particles
        )
    ))
    nominal_particles = jax.jit(jax.vmap(
        lambda history, controls, actions, key: nominal.sample_rollout(
            history, controls, actions, key, args.timing_particles
        )
    ))

    def measure(function, function_args, trajectories, particles_per_input=1):
        compile_started = time.perf_counter()
        output = function(*function_args)
        output.block_until_ready()
        compile_time = time.perf_counter() - compile_started
        # Additional warmups keep one-off runtime initialization out of the
        # samples for both models.
        for _ in range(2):
            function(*function_args).block_until_ready()
        samples = []
        for _ in range(args.timing_repeats):
            started = time.perf_counter()
            output = function(*function_args)
            output.block_until_ready()
            samples.append(time.perf_counter() - started)
        median = float(np.median(samples))
        transitions = trajectories * args.timing_horizon
        return {
            "jit_compile_time_s": compile_time,
            "median_batch_time_s": median,
            "milliseconds_per_trajectory": 1000 * median / trajectories,
            "microseconds_per_transition": 1e6 * median / transitions,
            "transitions_per_second": transitions / median,
            "input_batch_size": args.timing_batch_size,
            "particles_per_input": particles_per_input,
            "total_trajectories": trajectories,
            "horizon": args.timing_horizon,
            "repeats": args.timing_repeats,
        }

    deterministic_args = (histories, control_histories, future_controls)
    deterministic = {
        "latent_ssm_flow": measure(
            learned_rollout, deterministic_args, args.timing_batch_size
        ),
        "nominal_unicycle": measure(
            nominal_rollout, deterministic_args, args.timing_batch_size
        ),
    }
    deterministic["latent_to_nominal_time_ratio"] = (
        deterministic["latent_ssm_flow"]["median_batch_time_s"]
        / deterministic["nominal_unicycle"]["median_batch_time_s"]
    )
    particle_args = (
        histories, control_histories, future_controls, particle_keys
    )
    particle_trajectories = args.timing_batch_size * args.timing_particles
    stochastic = {
        "latent_ssm_flow": measure(
            learned_particles, particle_args, particle_trajectories,
            args.timing_particles,
        ),
        "nominal_unicycle": measure(
            nominal_particles, particle_args, particle_trajectories,
            args.timing_particles,
        ),
    }
    stochastic["latent_to_nominal_time_ratio"] = (
        stochastic["latent_ssm_flow"]["median_batch_time_s"]
        / stochastic["nominal_unicycle"]["median_batch_time_s"]
    )
    return {
        "methodology": (
            "Same CPU, inputs, batch size, horizon, JIT warmups, synchronization, "
            "and repeats for both models. Particle timing also uses the same "
            "number of stochastic trajectories."
        ),
        "deterministic_rollout": deterministic,
        "matched_particle_rollout": stochastic,
        "batch_size": args.timing_batch_size,
        "horizon": args.timing_horizon,
        "particles": args.timing_particles,
    }


def closed_loop_episode(controller, config, scenario_index, episode, args,
                        obstacles, goal):
    initial_y = -0.35 + 0.7 * (episode % 2)
    key = jr.fold_in(
        jr.PRNGKey(args.evaluation_seed + 1000 + 100 * scenario_index), episode
    )
    # Keep the initial-condition distribution fixed as the episode count grows.
    # The previous index-based expression reached 1.9 rad at episode 49.
    initial_yaw = jr.uniform(
        jr.fold_in(key, 10_000), (), minval=-0.15, maxval=0.15
    )
    state = state_from_pose(0.0, initial_y, initial_yaw)
    history = jnp.repeat(state[None], args.history, axis=0)
    prior_actions = jnp.zeros((args.history - 1, 2))
    queue = jnp.zeros((max(1, config.command_delay_steps), 2))
    trajectory, actions = [state], []
    elapsed = 0.0
    controller.reset()
    for step in range(args.closed_loop_steps):
        started = time.perf_counter()
        action, _ = controller.command(
            history, prior_actions, jr.fold_in(key, 2 * step)
        )
        action.block_until_ready()
        elapsed += time.perf_counter() - started
        state, queue = delayed_high_fidelity_step(
            state, queue, action, jr.fold_in(key, 2 * step + 1), config
        )
        trajectory.append(state)
        actions.append(action)
        history = jnp.concatenate([history[1:], state[None]])
        prior_actions = jnp.concatenate([prior_actions[1:], action[None]])
        if float(jnp.linalg.norm(state[:2] - goal)) <= args.goal_tolerance:
            break
    trajectory = jnp.stack(trajectory)
    clearance = float(minimum_clearance(trajectory, obstacles))
    goal_error = float(jnp.linalg.norm(trajectory[-1, :2] - goal))
    return {
        "success": goal_error <= args.goal_tolerance,
        "collision": clearance < config.robot_radius,
        "safety_violation": clearance < config.robot_radius + config.safety_margin,
        "goal_error_m": goal_error,
        "minimum_clearance_m": clearance,
        "steps": len(actions),
        "path_length_m": float(jnp.sum(jnp.linalg.norm(
            jnp.diff(trajectory[:, :2], axis=0), axis=-1
        ))),
        "command_time_s": elapsed,
        "mean_command_time_s": elapsed / max(1, len(actions)),
    }, np.asarray(trajectory)


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
        "total_mpc_commands": int(sum(row["steps"] for row in rows)),
        "total_command_time_s": float(sum(row["command_time_s"] for row in rows)),
        "mean_command_time_s": float(np.mean(
            [row["mean_command_time_s"] for row in rows]
        )),
    }


def plot(summary, trajectory_store, output, obstacles, goal):
    scenario_names = list(scenarios())
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {"latent_ssm_flow_cem": "tab:blue",
              "nominal_unicycle_cem": "tab:orange"}
    for scenario_index, (axis, scenario_name) in enumerate(
            zip(axes.flat[:-1], scenario_names)):
        for controller_name in colors:
            for episode in range(summary["settings"]["closed_loop_episodes"]):
                trajectory = trajectory_store[
                    f"{controller_name}_{scenario_name}_{episode}"
                ]
                axis.plot(
                    trajectory[:, 0], trajectory[:, 1],
                    color=colors[controller_name], alpha=0.45,
                    label=controller_name.replace("_", " ")
                    if episode == 0 else None,
                )
        for x, y, radius in np.asarray(obstacles):
            axis.add_patch(plt.Circle(
                (x, y), radius + 0.27 + 0.18, color="tab:red", alpha=0.14
            ))
        axis.scatter(*goal, marker="*", s=110, color="green")
        axis.set_title(scenario_name.replace("_", " "))
        axis.set_aspect("equal"); axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    timing_axis = axes.flat[4]
    controller_names = list(colors)
    mean_command_ms = [
        1000 * np.mean([
            summary["closed_loop"][scenario][controller]["mean_command_time_s"]
            for scenario in scenario_names
        ])
        for controller in controller_names
    ]
    timing_bars = timing_axis.bar(
        ["LatentSSMFlow", "Nominal model"], mean_command_ms,
        color=[colors[name] for name in controller_names],
    )
    timing_axis.bar_label(
        timing_bars,
        labels=[f"{value:.1f} ms" for value in mean_command_ms],
        padding=3,
    )
    timing_axis.set_ylabel("Mean time per MPC command [ms]")
    timing_axis.set_title("Closed-loop computation time")
    timing_axis.set_ylim(0, 1.28 * max(mean_command_ms))
    timing_axis.grid(axis="y", alpha=0.25)
    compile_times = summary.get("jit_compile_time_s", {})
    if all(name in compile_times for name in controller_names):
        timing_axis.text(
            0.5, 0.96,
            "JIT compile: " + ", ".join(
                f"{label} {compile_times[name]:.2f} s"
                for label, name in zip(("flow", "nominal"), controller_names)
            ),
            transform=timing_axis.transAxes, ha="center", va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    summary_axis = axes.flat[5]
    learned = [
        summary["closed_loop"][name]["latent_ssm_flow_cem"]["success_rate"]
        for name in scenario_names
    ]
    nominal = [
        summary["closed_loop"][name]["nominal_unicycle_cem"]["success_rate"]
        for name in scenario_names
    ]
    positions = np.arange(len(scenario_names))
    summary_axis.bar(positions - 0.18, learned, 0.36, label="LatentSSMFlow")
    summary_axis.bar(positions + 0.18, nominal, 0.36, label="Nominal model")
    summary_axis.set_xticks(
        positions, [name.replace("_", "\n") for name in scenario_names],
        fontsize=8,
    )
    summary_axis.set_ylim(0, 1.1); summary_axis.set_ylabel("Success rate")
    summary_axis.set_title("OOD closed-loop success")
    summary_axis.legend(); summary_axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Jackal CEM under unseen friction, delay, payload and disturbance",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_open_loop(summary, output):
    scenario_names = list(scenarios())
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, scenario_name in zip(axes.flat[:-1], scenario_names):
        curves = summary["open_loop"][scenario_name]["horizon_position_rmse_m"]
        horizon = np.arange(1, len(curves["latent_ssm_flow"]) + 1)
        axis.plot(horizon, curves["latent_ssm_flow"], label="LatentSSMFlow")
        axis.plot(horizon, curves["nominal_unicycle"], label="Nominal model")
        axis.set_title(scenario_name.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.set_xlabel("Prediction horizon [steps]")
        axis.set_ylabel("Position RMSE [m]")
    axes.flat[0].legend()
    timing_axis = axes.flat[5]
    timing = summary["dynamics_computation_time"]["deterministic_rollout"]
    names = ["latent_ssm_flow", "nominal_unicycle"]
    values = [timing[name]["microseconds_per_transition"] for name in names]
    bars = timing_axis.bar(
        ["LatentSSMFlow", "Nominal model"], values,
        color=["tab:blue", "tab:orange"],
    )
    timing_axis.bar_label(
        bars, labels=[f"{value:.1f} µs" for value in values], padding=3
    )
    timing_axis.set_ylabel("JIT rollout time [µs/transition]")
    timing_axis.set_title(
        f"Matched rollout throughput\n"
        f"ratio = {timing['latent_to_nominal_time_ratio']:.1f}×"
    )
    timing_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Open-loop reconstruction under held-out plant mismatch")
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args)
    if args.open_loop_only and args.skip_open_loop:
        raise ValueError("--open-loop-only cannot be combined with --skip-open-loop")
    skip_open_loop = args.skip_open_loop or args.closed_loop_only
    skip_timing = args.skip_timing or args.closed_loop_only
    open_loop_results = None if skip_open_loop else open_loop(model, args)
    dynamics_computation_time = (
        None if skip_timing else dynamics_timing(model, args)
    )
    if args.open_loop_only:
        summary = {
            "open_loop": open_loop_results,
            "dynamics_computation_time": dynamics_computation_time,
            "settings": {
                "open_loop_episodes": args.open_loop_episodes,
                "open_loop_steps": args.open_loop_steps,
                "open_loop_particles": args.open_loop_particles,
            },
        }
        (args.output / "jackal_ood_open_loop_reconstruction.json").write_text(
            json.dumps(summary, indent=2)
        )
        if dynamics_computation_time is not None:
            plot_open_loop(
                summary, args.output / "jackal_ood_open_loop_reconstruction.png"
            )
        print(json.dumps(summary, indent=2), flush=True)
        return
    goal = jnp.asarray([4.0, 0.0])
    obstacles = jnp.asarray([[2.0, 0.0, 0.42], [3.0, 0.7, 0.28]])
    cost = obstacle_cost(goal, obstacles)
    mpc_config = SamplingMPCConfig(
        horizon=args.horizon, num_candidates=args.candidates,
        num_particles=args.particles, iterations=args.cem_iterations,
        elite_fraction=0.1, initial_std=0.4, min_std=0.02,
        risk_weight=args.risk_weight, smoothness_weight=0.05,
        smoothness_scale=jnp.asarray([1.0, 1.5]),
    )
    low, high = jnp.asarray([0.0, -1.5]), jnp.asarray([1.0, 1.5])
    controllers = {
        "latent_ssm_flow_cem": CEMStochasticMPC(
            model, low, high, cost, mpc_config
        ),
        # This controller is intentionally given only the nominal model.
        "nominal_unicycle_cem": CEMStochasticMPC(
            ExactJackalModel(process_noise=args.nominal_process_noise),
            low, high, cost, mpc_config,
        ),
    }
    # Compile both cost kernels once. Compilation is reported but excluded from
    # steady-state per-command timing.
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

    rows, trajectory_store, closed_results = [], {}, {}
    for scenario_index, (scenario_name, plant_config) in enumerate(
            scenarios().items()):
        closed_results[scenario_name] = {}
        for controller_name, controller in controllers.items():
            controller_rows = []
            for episode in range(args.closed_loop_episodes):
                row, trajectory = closed_loop_episode(
                    controller, plant_config, scenario_index, episode, args,
                    obstacles, goal,
                )
                row.update(
                    scenario=scenario_name, controller=controller_name,
                    episode=episode,
                )
                rows.append(row); controller_rows.append(row)
                trajectory_store[
                    f"{controller_name}_{scenario_name}_{episode}"
                ] = trajectory
                print(row, flush=True)
            closed_results[scenario_name][controller_name] = aggregate(
                controller_rows
            )
    summary = {
        "open_loop": open_loop_results,
        "closed_loop": closed_results,
        "dynamics_computation_time": dynamics_computation_time,
        "jit_compile_time_s": compile_times,
        "settings": {
            "closed_loop_episodes": args.closed_loop_episodes,
            "horizon": args.horizon, "candidates": args.candidates,
            "particles": args.particles, "cem_iterations": args.cem_iterations,
            "risk_weight": args.risk_weight,
        },
    }
    (args.output / "jackal_ood_cem_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (args.output / "jackal_ood_cem_episodes.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez(args.output / "jackal_ood_cem_trajectories.npz", **trajectory_store)
    plot(
        summary, trajectory_store,
        args.output / "jackal_ood_cem_comparison.png", obstacles, goal,
    )
    if open_loop_results is not None and dynamics_computation_time is not None:
        plot_open_loop(
            summary, args.output / "jackal_ood_open_loop_reconstruction.png"
        )
    print(json.dumps(summary, indent=2), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--evaluation-seed", type=int, default=3030)
    result.add_argument("--history", type=int, default=8)
    result.add_argument("--hidden-dim", type=int, default=96)
    result.add_argument("--width", type=int, default=96)
    result.add_argument("--hidden-layers", type=int, default=2)
    result.add_argument("--blocks", type=int, default=6)
    result.add_argument("--max-rho", type=float, default=0.92)
    result.add_argument("--open-loop-episodes", type=int, default=12)
    result.add_argument("--open-loop-steps", type=int, default=80)
    result.add_argument("--open-loop-particles", type=int, default=64)
    result.add_argument("--closed-loop-episodes", type=int, default=2)
    result.add_argument("--closed-loop-steps", type=int, default=60)
    result.add_argument("--goal-tolerance", type=float, default=0.3)
    result.add_argument("--horizon", type=int, default=15)
    result.add_argument("--candidates", type=int, default=64)
    result.add_argument("--particles", type=int, default=4)
    result.add_argument("--cem-iterations", type=int, default=3)
    result.add_argument("--risk-weight", type=float, default=1.0)
    result.add_argument("--nominal-process-noise", type=float, default=0.01)
    result.add_argument("--timing-batch-size", type=int, default=128)
    result.add_argument("--timing-horizon", type=int, default=40)
    result.add_argument("--timing-particles", type=int, default=8)
    result.add_argument("--timing-repeats", type=int, default=20)
    result.add_argument("--open-loop-only", action="store_true")
    result.add_argument(
        "--skip-open-loop", action="store_true",
        help="Skip reconstruction evaluation before closed-loop control.",
    )
    result.add_argument(
        "--skip-timing", action="store_true",
        help="Skip the standalone dynamics throughput benchmark.",
    )
    result.add_argument(
        "--closed-loop-only", action="store_true",
        help="Run only closed-loop CEM (equivalent to both skip flags).",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
