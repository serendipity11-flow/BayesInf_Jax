"""Closed-loop waypoint navigation with stochastic FlowSDE MPC."""
import argparse, json, time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .env import DifferentialDriveConfig, DifferentialDriveEnv
from .evaluate import load_model
from .control_objective import trajectory_cost_jax


DIRECT_TASKS = [((-3.5, -2.4, .2), [(3.4, 2.3)]),
                ((-3.2, 2.5, -.4), [(3.5, -2.2)]),
                ((3.4, -2.5, 2.7), [(-3.4, 2.2)])]
COMPLEX_TASKS = [
    ((-3.5, -2.5, .2), [(-2.0, 1.8), (0.0, -1.8), (2.0, 1.8), (3.5, -2.5)]),
    ((-3.8, 0.0, 0.0), [(-2.2, 2.3), (-.5, -2.2), (1.4, 2.2), (3.7, -.5)]),
    ((-3.0, -2.2, 0.0), [(3.0, -2.2), (3.0, 2.2), (-3.0, 2.2), (-3.0, -1.5)]),
]


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model, config = load_model(args.result)
    stats = dict(np.load(args.result / "normalization.npz"))
    history_steps = int(config.get("history_steps", 1))
    feature_mean = jnp.asarray(stats["feature_mean"])
    feature_std = jnp.asarray(stats["feature_std"])
    action_mean = jnp.asarray(stats["action_mean"])
    action_std = jnp.asarray(stats["action_std"])
    delta_mean = jnp.asarray(stats["delta_mean"])
    delta_std = jnp.asarray(stats["delta_std"])
    state_dim = int(config.get("state_dim", len(delta_mean)))
    state_order = tuple(config.get("state_order", ("p_x", "p_y", "theta", "v", "omega")))
    compact = "cos_theta" in state_order
    structured = bool(config.get("structured_kinematics", False))
    dynamic_indices = jnp.asarray(config.get("dynamic_indices", range(state_dim)))

    def encode(states):
        raw = (states if compact else jnp.concatenate((states[..., :2],
               jnp.sin(states[..., 2:3]), jnp.cos(states[..., 2:3]), states[..., 3:]), axis=-1))
        return (raw - feature_mean) / feature_std

    sqrt_dt = jnp.sqrt(config["dt"])

    def advance(states, old_actions, action, base_noise):
            normalized_action = (action - action_mean) / action_std
            context = jnp.concatenate((encode(states).reshape(-1), old_actions.reshape(-1)))
            joint = jnp.concatenate((context, normalized_action))
            first_mean = config["dt"] * model.drift(joint)
            first_residual = model.flow.forward(base_noise, model.conditioner(joint))[0]
            first_delta = first_mean + sqrt_dt * first_residual

            def reconstruct(dynamic_delta):
                delta = dynamic_delta * delta_std + delta_mean
                nxt = states[-1].at[dynamic_indices].add(delta)
                if structured:
                    vf_i, vl_i, omega_i = (5, 6, 8) if compact else (3, 4, 5)
                    vf, vl = .5 * (states[-1, vf_i] + nxt[vf_i]), .5 * (states[-1, vl_i] + nxt[vl_i])
                    omega = .5 * (states[-1, omega_i] + nxt[omega_i])
                    theta = (jnp.arctan2(states[-1, 4], states[-1, 3]) if compact
                             else states[-1, 2]) + config["dt"] * omega
                    c, s = jnp.cos(theta), jnp.sin(theta)
                    nxt = nxt.at[0].set(states[-1, 0] + config["dt"] * (vf*c-vl*s))
                    nxt = nxt.at[1].set(states[-1, 1] + config["dt"] * (vf*s+vl*c))
                    nxt = (nxt.at[3].set(c).at[4].set(s) if compact
                           else nxt.at[2].set(jnp.arctan2(s, c)))
                return nxt

            predictor = reconstruct(first_delta)
            predictor_states = jnp.concatenate((states[1:], predictor[None]), axis=0)
            predictor_actions = (jnp.concatenate((old_actions[1:], normalized_action[None]), axis=0)
                                 if history_steps > 1 else old_actions)
            predictor_context = jnp.concatenate((encode(predictor_states).reshape(-1),
                                                  predictor_actions.reshape(-1)))
            predictor_joint = jnp.concatenate((predictor_context, normalized_action))
            second_mean = config["dt"] * model.drift(predictor_joint)
            second_residual = model.flow.forward(base_noise, model.conditioner(predictor_joint))[0]
            nxt = reconstruct(.5 * (first_delta + second_mean + sqrt_dt * second_residual))
            if compact:
                ci, si = state_order.index("cos_theta"), state_order.index("sin_theta")
                norm = jnp.maximum(jnp.hypot(nxt[ci], nxt[si]), 1e-6)
                nxt = nxt.at[ci].set(nxt[ci] / norm).at[si].set(nxt[si] / norm)
            else:
                nxt = nxt.at[2].set(jnp.arctan2(jnp.sin(nxt[2]), jnp.cos(nxt[2])))
            new_states = jnp.concatenate((states[1:], nxt[None]), axis=0)
            new_actions = (jnp.concatenate((old_actions[1:], normalized_action[None]), axis=0)
                           if history_steps > 1 else old_actions)
            return new_states, new_actions, nxt

    def rollout_one(state_history, previous_actions, physical_actions, base_noises):
        def step(carry, inputs):
            states, old_actions = carry
            action, noise = inputs
            states, old_actions, nxt = advance(states, old_actions, action, noise)
            return (states, old_actions), nxt
        return jax.lax.scan(step, (state_history, previous_actions), (physical_actions, base_noises))[1]

    @jax.jit
    def predict_paths(state_history, previous_actions, actions, noises):
        return jax.vmap(rollout_one, in_axes=(None, None, 0, 0))(
            state_history, previous_actions, actions, noises)

    @jax.jit
    def score_candidates(state_history, previous_actions, actions, noises, goal):
        # One candidate action sequence is evaluated under independent samples
        # from the FlowSDE's learned conditional transition distribution.
        candidate_count, stochastic_samples = noises.shape[:2]
        repeated_actions = jnp.repeat(actions[:, None], stochastic_samples, axis=1)
        flat_paths = predict_paths(
            state_history, previous_actions,
            repeated_actions.reshape((-1, args.mpc_horizon, 2)),
            noises.reshape((-1, args.mpc_horizon, len(dynamic_indices))))
        paths = flat_paths.reshape((candidate_count, stochastic_samples,
                                    args.mpc_horizon, state_dim))
        costs = trajectory_cost_jax(paths, repeated_actions, goal)
        return jnp.mean(costs, axis=1)

    @jax.jit
    def plan_mppi(state_history, previous_actions, mean, std, goal, key):
        """Run all MPPI refinements on-device with a single host synchronization."""
        def iteration(_, carry):
            current_mean, current_key, _, _, _ = carry
            current_key, control_key, process_key = jax.random.split(current_key, 3)
            control_noise = jax.random.normal(
                control_key, (args.candidates, args.mpc_horizon, 2)) * std
            candidates = jnp.clip(current_mean[None] + control_noise,
                                  jnp.asarray((0., -3.)), jnp.asarray((2.2, 3.)))
            model_noise = jax.random.normal(
                process_key, (args.candidates, args.stochastic_samples,
                              args.mpc_horizon, len(dynamic_indices)))
            values = score_candidates(state_history, previous_actions, candidates, model_noise, goal)
            weights = jax.nn.softmax(-(values - jnp.min(values)) / args.temperature)
            updated_mean = jnp.sum(weights[:, None, None] * candidates, axis=0)
            return updated_mean, current_key, candidates, model_noise, values
        empty_candidates = jnp.zeros((args.candidates, args.mpc_horizon, 2))
        empty_process = jnp.zeros((args.candidates, args.stochastic_samples,
                                   args.mpc_horizon, len(dynamic_indices)))
        empty_values = jnp.zeros((args.candidates,))
        return jax.lax.fori_loop(0, args.cem_iterations, iteration,
                                 (mean, key, empty_candidates, empty_process, empty_values))

    rng = np.random.default_rng(args.seed)
    planning_key = jax.random.PRNGKey(args.seed)
    tasks = COMPLEX_TASKS if args.task_set == "complex" else DIRECT_TASKS
    records = []
    all_latencies = []
    for task_id, (start, waypoint_tuples) in enumerate(tasks[:args.tasks]):
        waypoints = np.asarray(waypoint_tuples); waypoint_index = 0; goal = waypoints[0]
        env = DifferentialDriveEnv(DifferentialDriveConfig(
            extended_state=5 < state_dim <= 8, full_state=state_dim > 8 and not compact,
            compact_physical_state=compact),
                                   seed=args.seed + task_id)
        state = env.reset(start)
        state_history = np.repeat(state[None], history_steps, axis=0)
        zero_action = (np.zeros(2) - stats["action_mean"]) / stats["action_std"]
        previous_actions = np.repeat(zero_action[None], max(0, history_steps - 1), axis=0)
        mean = np.zeros((args.mpc_horizon, 2)); std = np.tile((.75, 1.4), (args.mpc_horizon, 1))
        states, controls, costs, latencies, reached_steps, sample_clouds = [state.copy()], [], [], [], [], []
        for step in range(args.max_steps):
            planning_state_history = state_history.copy()
            planning_previous_actions = previous_actions.copy()
            planning_started = time.perf_counter()
            if args.optimizer == "mppi":
                mean_jax, planning_key, candidates_jax, process_noise_jax, values_jax = plan_mppi(
                    jnp.asarray(state_history), jnp.asarray(previous_actions), jnp.asarray(mean),
                    jnp.asarray(std), jnp.asarray(goal), planning_key)
                mean, candidates, process_noise, values = map(np.asarray,
                    (mean_jax, candidates_jax, process_noise_jax, values_jax))
            else:
              for _ in range(args.cem_iterations):
                control_noise = rng.normal(size=(args.candidates, args.mpc_horizon, 2)) * std
                candidates = mean[None] + control_noise
                candidates = np.clip(candidates, (0., -3.), (2.2, 3.))
                process_noise = rng.normal(size=(args.candidates, args.stochastic_samples,
                                                 args.mpc_horizon, len(dynamic_indices)))
                values = np.asarray(score_candidates(jnp.asarray(state_history),
                    jnp.asarray(previous_actions), jnp.asarray(candidates), jnp.asarray(process_noise),
                    jnp.asarray(goal)))
                elite = candidates[np.argpartition(values, args.elites)[:args.elites]]
                mean, std = elite.mean(0), np.maximum(elite.std(0), (.05, .1))
            latencies.append(time.perf_counter() - planning_started)
            action = mean[0]
            state = env.step(action, stochastic=True)
            normalized_action = (action - stats["action_mean"]) / stats["action_std"]
            state_history = np.concatenate((state_history[1:], state[None]), axis=0)
            if history_steps > 1:
                previous_actions = np.concatenate((previous_actions[1:], normalized_action[None]), axis=0)
            states.append(state.copy()); controls.append(action.copy()); costs.append(float(values.min()))
            mean = np.concatenate((mean[1:], mean[-1:])); std = np.concatenate((std[1:], std[-1:]))
            distance = float(np.linalg.norm(state[:2] - goal))
            if step == 0 or (step + 1) % args.report_every == 0:
                print(json.dumps({"task": task_id, "step": step + 1,
                                  "waypoint": waypoint_index, "distance_m": distance,
                                  "planning_ms": 1000 * latencies[-1]}), flush=True)
            if args.save_samples and (step == 0 or (step + 1) % args.report_every == 0):
                repeated = np.repeat(candidates[:, None], args.stochastic_samples, axis=1)
                paths = np.asarray(predict_paths(
                    jnp.asarray(planning_state_history), jnp.asarray(planning_previous_actions),
                    jnp.asarray(repeated.reshape((-1, args.mpc_horizon, 2))),
                    jnp.asarray(process_noise.reshape((-1, args.mpc_horizon,
                                                       len(dynamic_indices))))))
                paths = paths.reshape((args.candidates, args.stochastic_samples,
                                       args.mpc_horizon, state_dim))
                sample_clouds.append((step + 1, paths[..., :2], values.copy(), candidates.copy()))
            if distance <= args.goal_tolerance:
                reached_steps.append(step + 1); waypoint_index += 1
                if waypoint_index == len(waypoints): break
                goal = waypoints[waypoint_index]; mean.fill(0.0); std[:] = (.75, 1.4)
        states, controls = np.asarray(states), np.asarray(controls)
        final_distance = float(np.linalg.norm(states[-1, :2] - waypoints[-1]))
        record = {"task": task_id, "task_set": args.task_set, "start": start,
                  "goal": waypoint_tuples[-1], "waypoints": waypoint_tuples, "states": states,
                  "controls": controls, "final_distance_m": final_distance,
                  "completed_waypoints": waypoint_index, "waypoint_reached_steps": reached_steps,
                  "steps": len(controls), "success": waypoint_index == len(waypoints),
                  "mean_planning_ms": 1000 * float(np.mean(latencies)),
                  "p95_planning_ms": 1000 * float(np.quantile(latencies, .95))}
        records.append(record); all_latencies.extend(latencies)
        np.savez_compressed(args.output / f"flowsde_{args.optimizer}_task{task_id}.npz",
                            states=states, controls=controls, waypoints=waypoints)
        if args.save_samples and sample_clouds:
            np.savez_compressed(args.output / f"flowsde_{args.optimizer}_samples_task{task_id}.npz",
                                steps=np.asarray([x[0] for x in sample_clouds]),
                                paths=np.stack([x[1] for x in sample_clouds]),
                                costs=np.stack([x[2] for x in sample_clouds]),
                                controls=np.stack([x[3] for x in sample_clouds]))

    metrics = {"controller": f"FlowSDE stochastic Heun {args.optimizer.upper()}-MPC", "tasks": len(records),
               "task_set": args.task_set, "candidates": args.candidates,
               "stochastic_samples_per_candidate": args.stochastic_samples,
               "horizon": args.mpc_horizon,
               "success_rate": float(np.mean([r["success"] for r in records])),
               "goal_tolerance_m": args.goal_tolerance,
               "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in records])),
               "mean_planning_ms": 1000 * float(np.mean(all_latencies)),
               "p95_planning_ms": 1000 * float(np.quantile(all_latencies, .95)),
               "control_frequency_hz": float(1 / np.mean(all_latencies)),
               "episodes": [{k: v for k, v in r.items() if k not in ("states", "controls")}
                            for r in records]}
    (args.output / "closed_loop_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figure, ax = plt.subplots(figsize=(8, 6)); terrain = DifferentialDriveEnv(seed=0)
    gx, gy = np.meshgrid(np.linspace(-4.5, 4.5, 120), np.linspace(-3.5, 3.5, 100))
    mu = np.vectorize(terrain.friction_at)(gx, gy)
    image = ax.contourf(gx, gy, mu, levels=14, cmap="YlGn", alpha=.55)
    for record in records:
        xy = record["states"][:, :2]
        ax.plot(xy[:, 0], xy[:, 1], lw=2, label=f"task {record['task']}")
        color = ax.lines[-1].get_color(); ax.plot(*xy[0], "o", color=color)
        wp = np.asarray(record["waypoints"]); ax.plot(wp[:, 0], wp[:, 1], "*--", ms=11, color=color)
    figure.colorbar(image, ax=ax, label="local friction coefficient")
    ax.set(title="Closed-loop FlowSDE MPC navigation", xlabel="x [m]", ylabel="y [m]",
           xlim=(-4.5, 4.5), ylim=(-3.5, 3.5)); ax.legend(); ax.set_aspect("equal")
    figure.tight_layout(); figure.savefig(args.output / "closed_loop_navigation.png", dpi=180)
    figure.savefig(args.output / "closed_loop_navigation.pdf"); plt.close(figure)
    print(json.dumps(metrics, indent=2)); return metrics


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, default=Path("results/results_optimal_control/differential_drive/smooth_terrain_seed0/joint_stage"))
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/differential_drive/smooth_terrain_seed0/evaluation"))
    p.add_argument("--tasks", type=int, default=3); p.add_argument("--max-steps", type=int, default=350)
    p.add_argument("--task-set", choices=("direct", "complex"), default="direct")
    p.add_argument("--mpc-horizon", type=int, default=100); p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--stochastic-samples", type=int, default=64,
                   help="Independent learned FlowSDE rollouts per input candidate")
    p.add_argument("--elites", type=int, default=32); p.add_argument("--cem-iterations", type=int, default=3)
    p.add_argument("--optimizer", choices=("cem", "mppi"), default="cem")
    p.add_argument("--temperature", type=float, default=.35)
    p.add_argument("--goal-tolerance", type=float, default=.35); p.add_argument("--seed", type=int, default=31)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--save-samples", action="store_true")
    return p


if __name__ == "__main__": run(parser().parse_args())
