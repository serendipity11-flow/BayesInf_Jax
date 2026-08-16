"""MPPI with a vectorized nonlinear bicycle-like model controlling full MuJoCo."""
import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from .env import DifferentialDriveConfig, DifferentialDriveEnv
from .extended_physical_model import ExtendedPhysicalModel
from .control_objective import trajectory_cost_numpy


DIRECT_TASKS = [((-3.5, -2.4, .2), [(3.4, 2.3)]),
                ((-3.2, 2.5, -.4), [(3.5, -2.2)]),
                ((3.4, -2.5, 2.7), [(-3.4, 2.2)])]
COMPLEX_TASKS = [
    ((-3.5, -2.5, .2), [(-2.0, 1.8), (0.0, -1.8), (2.0, 1.8), (3.5, -2.5)]),
    ((-3.8, 0.0, 0.0), [(-2.2, 2.3), (-.5, -2.2), (1.4, 2.2), (3.7, -.5)]),
    ((-3.0, -2.2, 0.0), [(3.0, -2.2), (3.0, 2.2), (-3.0, 2.2), (-3.0, -1.5)]),
]


def friction_at(x, y):
    variation = (.55 * np.sin(.72 * x) * np.cos(.51 * y)
                 + .30 * np.sin(1.31 * y + .37 * x)
                 + .15 * np.cos(1.73 * x - .44 * y))
    return np.clip(.72 + .16 * variation, .48, 1.0)


def rollout_costs(initial, controls, goal, args, process_noise=None, return_paths=False):
    """Roll out every candidate simultaneously through a nonlinear 5D model."""
    if args.extended_model is not None:
        state = np.broadcast_to(initial, (len(controls), len(initial))).copy()
        paths = np.empty((len(controls), controls.shape[1] + 1, 2), np.float32)
        paths[:, 0] = state[:, :2]
        for k in range(controls.shape[1]):
            action = controls[:, k]
            noise = None if process_noise is None else process_noise[:, k]
            state = args._extended_model.step(state, action, noise)
            paths[:, k + 1] = state[:, :2]
        total = trajectory_cost_numpy(paths[:, 1:], controls, goal)
        return (total, paths) if return_paths else total
    count = len(controls)
    x = np.full(count, initial[0]); y = np.full(count, initial[1])
    theta = np.full(count, initial[2]); v = np.full(count, initial[3])
    omega = np.full(count, initial[4])
    paths = np.empty((count, controls.shape[1] + 1, 2), np.float32)
    paths[:, 0] = np.column_stack((x, y))
    for k in range(controls.shape[1]):
        action = controls[:, k]; v_cmd, omega_cmd = action[:, 0], action[:, 1]
        traction = friction_at(x, y) / .72
        # First-order wheel/actuator response with nonlinear drag and turn-induced slip.
        v_dot = (traction * (args.gain_v * v_cmd - v) / args.tau_v
                 - args.drag_v * v * np.abs(v) - args.turn_drag * np.abs(omega) * v)
        omega_dot = (traction * (args.gain_omega * omega_cmd - omega) / args.tau_omega
                     - args.drag_omega * omega * np.abs(omega))
        v_next = v + args.dt * v_dot; omega_next = omega + args.dt * omega_dot
        if process_noise is not None:
            v_next += args.process_std_v * process_noise[:, k, 0]
            omega_next += args.process_std_omega * process_noise[:, k, 1]
        v_mid, omega_mid = .5 * (v + v_next), .5 * (omega + omega_next)
        theta_mid = theta + .5 * args.dt * omega_mid
        x += args.dt * v_mid * np.cos(theta_mid)
        y += args.dt * v_mid * np.sin(theta_mid)
        theta += args.dt * omega_mid
        paths[:, k + 1] = np.column_stack((x, y))
        v, omega = v_next, omega_next
    total = trajectory_cost_numpy(paths[:, 1:], controls, goal)
    return (total, paths) if return_paths else total


def write_partial(output, records, active=None):
    payload = {"completed_tasks": len(records),
               "episodes": [{k: v for k, v in r.items() if k not in ("states", "actions")}
                            for r in records]}
    if active is not None: payload["active"] = active
    (output / "nonlinear_physical_mppi_progress.json").write_text(json.dumps(payload, indent=2) + "\n")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(args.seed)
    if args.extended_model is not None: args._extended_model = ExtendedPhysicalModel.load(args.extended_model)
    tasks = COMPLEX_TASKS if args.task_set == "complex" else DIRECT_TASKS
    records = []
    all_latencies = []
    for task_id, (start, waypoint_tuples) in enumerate(tasks[:args.tasks]):
        waypoints = np.asarray(waypoint_tuples); waypoint_index = 0; goal = waypoints[0]
        plant = DifferentialDriveEnv(DifferentialDriveConfig(extended_state=args.extended_model is not None),
                                     seed=args.seed + task_id)
        state = plant.reset(start); mean = np.zeros((args.horizon, 2))
        states, actions, sample_clouds, latencies, reached_steps = [state.copy()], [], [], [], []
        initial = float(np.linalg.norm(state[:2] - goal))
        for step in range(args.max_steps):
            planning_started = time.perf_counter()
            for _ in range(args.iterations):
                noise = rng.normal(size=(args.samples, args.horizon, 2))
                noise *= np.asarray((args.sigma_v, args.sigma_omega))
                candidates = np.clip(mean[None] + noise, (0., -3.), (2.2, 3.))
                repeated = np.repeat(candidates[:, None], args.stochastic_samples, axis=1)
                flat_controls = repeated.reshape((-1, args.horizon, 2))
                noise_dim = len(args._extended_model.process_cholesky) if args.extended_model is not None else 2
                process_noise = rng.normal(size=(len(flat_controls), args.horizon, noise_dim))
                sampled_costs = rollout_costs(state, flat_controls, goal, args, process_noise)
                costs = sampled_costs.reshape(args.samples, args.stochastic_samples).mean(axis=1)
                weights = np.exp(-(costs - costs.min()) / args.temperature)
                weights /= np.maximum(weights.sum(), 1e-12)
                mean = np.sum(weights[:, None, None] * candidates, axis=0)
            latencies.append(time.perf_counter() - planning_started)
            action = mean[0].copy(); state = plant.step(action, stochastic=True)
            states.append(state.copy()); actions.append(action)
            mean = np.concatenate((mean[1:], mean[-1:]))
            distance = float(np.linalg.norm(state[:2] - goal))
            if step == 0 or (step + 1) % args.report_every == 0:
                sampled_costs, sampled_paths = rollout_costs(
                    states[-2], flat_controls, goal, args, process_noise, return_paths=True)
                if args.save_samples:
                    sample_clouds.append((step + 1, sampled_paths, sampled_costs, candidates.copy()))
                active = {"task": task_id, "step": step + 1, "distance_m": distance,
                          "command": action.tolist(), "best_predicted_cost": float(costs.min())}
                write_partial(args.output, records, active); print(json.dumps(active), flush=True)
            if distance <= args.goal_tolerance:
                reached_steps.append(step + 1); waypoint_index += 1
                if waypoint_index == len(waypoints): break
                goal = waypoints[waypoint_index]; mean.fill(0.0)
        states, actions = np.asarray(states), np.asarray(actions)
        final_goal = waypoints[-1]; final = float(np.linalg.norm(states[-1, :2] - final_goal))
        success = waypoint_index == len(waypoints)
        record = {"task": task_id, "task_set": args.task_set, "start": start,
                  "goal": waypoint_tuples[-1], "waypoints": waypoint_tuples, "states": states,
                  "actions": actions, "initial_distance_m": initial, "final_distance_m": final,
                  "completed_waypoints": waypoint_index, "waypoint_reached_steps": reached_steps,
                  "steps": len(actions), "success": success,
                  "mean_v_command": float(actions[:, 0].mean()),
                  "mean_abs_omega_command": float(np.abs(actions[:, 1]).mean()),
                  "mean_planning_ms": 1000 * float(np.mean(latencies)),
                  "p95_planning_ms": 1000 * float(np.quantile(latencies, .95))}
        all_latencies.extend(latencies)
        records.append(record); write_partial(args.output, records)
        np.savez_compressed(args.output / f"nonlinear_physical_mppi_task{task_id}.npz",
                            states=states, actions=actions, goal=goal)
        if args.save_samples:
            np.savez_compressed(args.output / f"nonlinear_physical_mppi_samples_task{task_id}.npz",
                                steps=np.asarray([c[0] for c in sample_clouds]),
                                paths=np.stack([c[1] for c in sample_clouds]),
                                costs=np.stack([c[2] for c in sample_clouds]),
                                controls=np.stack([c[3] for c in sample_clouds]))

    model_info = ({"state_order": ["p_x", "p_y", "theta", "v_forward", "v_lateral",
                                   "omega", "wheel_left", "wheel_right"],
                   "identified_model": str(args.extended_model)} if args.extended_model is not None else
                  {k: getattr(args, k) for k in ("dt", "gain_v", "gain_omega", "tau_v",
                   "tau_omega", "drag_v", "drag_omega", "turn_drag")})
    metrics = {"controller": ("state-matched 8D nonlinear physical-model MPPI on MuJoCo"
                               if args.extended_model is not None else
                               "nonlinear physical-model MPPI on MuJoCo"),
               "tasks": len(records), "samples": args.samples, "horizon": args.horizon,
               "stochastic_samples_per_candidate": args.stochastic_samples,
               "task_set": args.task_set,
               "success_rate": float(np.mean([r["success"] for r in records])),
               "goal_tolerance_m": args.goal_tolerance,
               "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in records])),
               "mean_planning_ms": 1000 * float(np.mean(all_latencies)),
               "p95_planning_ms": 1000 * float(np.quantile(all_latencies, .95)),
               "control_frequency_hz": float(1 / np.mean(all_latencies)),
               "model": model_info,
               "episodes": [{k: v for k, v in r.items() if k not in ("states", "actions")}
                            for r in records]}
    (args.output / "nonlinear_physical_mppi_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figure, ax = plt.subplots(figsize=(8, 6))
    for record in records:
        xy = record["states"][:, :2]; line, = ax.plot(*xy.T, lw=2, label=f"task {record['task']}")
        ax.plot(*xy[0], "o", color=line.get_color())
        wp = np.asarray(record["waypoints"]); ax.plot(wp[:, 0], wp[:, 1], "*--", ms=11, color=line.get_color())
    ax.set(title="Nonlinear physical-model MPPI on full MuJoCo", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal"); ax.legend(); figure.tight_layout()
    figure.savefig(args.output / "nonlinear_physical_mppi_navigation.png", dpi=180); plt.close(figure)
    if args.save_samples:
        figure, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 5), squeeze=False)
        for ax, record in zip(axes[0], records):
            cloud = np.load(args.output / f"nonlinear_physical_mppi_samples_task{record['task']}.npz")
            paths = cloud["paths"][-1]
            for path in paths: ax.plot(path[:, 0], path[:, 1], color="C0", alpha=.025, lw=.7)
            xy = record["states"][:, :2]; ax.plot(*xy.T, "k", lw=2, label="executed")
            wp = np.asarray(record["waypoints"]); ax.plot(wp[:, 0], wp[:, 1], "*--", ms=11, color="C3")
            ax.set_title(f"task {record['task']}: all {len(paths)} samples")
            ax.set_aspect("equal"); ax.legend()
        figure.tight_layout(); figure.savefig(args.output / "nonlinear_physical_mppi_sample_clouds.png", dpi=180); plt.close(figure)
    print(json.dumps(metrics, indent=2)); return metrics


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=Path(
        "results/results_optimal_control/differential_drive/nonlinear_physical_mppi"))
    p.add_argument("--tasks", type=int, default=3); p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--task-set", choices=("direct", "complex"), default="direct")
    p.add_argument("--save-samples", action="store_true",
                   help="Save every candidate at reporting checkpoints (large output)")
    p.add_argument("--horizon", type=int, default=100); p.add_argument("--samples", type=int, default=64,
                   help="Sampled input candidates per MPPI update")
    p.add_argument("--stochastic-samples", type=int, default=64,
                   help="Gaussian process-noise rollouts per input candidate")
    p.add_argument("--iterations", type=int, default=3); p.add_argument("--temperature", type=float, default=.35)
    p.add_argument("--sigma-v", type=float, default=.75); p.add_argument("--sigma-omega", type=float, default=1.4)
    p.add_argument("--goal-tolerance", type=float, default=.35); p.add_argument("--seed", type=int, default=31)
    p.add_argument("--report-every", type=int, default=25); p.add_argument("--dt", type=float, default=.02)
    p.add_argument("--gain-v", type=float, default=.225); p.add_argument("--gain-omega", type=float, default=.36)
    p.add_argument("--tau-v", type=float, default=.25); p.add_argument("--tau-omega", type=float, default=.18)
    p.add_argument("--drag-v", type=float, default=.08); p.add_argument("--drag-omega", type=float, default=.04)
    p.add_argument("--turn-drag", type=float, default=.08)
    p.add_argument("--process-std-v", type=float, default=.001,
                   help="Per-step velocity increment noise for the fallback 5D model")
    p.add_argument("--process-std-omega", type=float, default=.001,
                   help="Per-step yaw-rate increment noise for the fallback 5D model")
    p.add_argument("--extended-model", type=Path,
                   help="Fitted state-matched 8D nonlinear physical model")
    return p


if __name__ == "__main__": run(parser().parse_args())
