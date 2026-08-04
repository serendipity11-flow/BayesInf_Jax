"""Offline and closed-loop benchmark for FlowSDE-MPPI and three baselines."""

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

from .data import load, normalize
from .env_wrapper import NoisyPushingEnv
from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE, GaussianSDE
from .mppi_planner import MPPIConfig, MPPIPlanner, NormalizedDynamics, SimulatorMPPIPlanner
from .state import ACTION_DIM, HistoryBuffer, INPUT_DIM, OBJECT_POSITION, OBJECT_YAW, STATE_DIM


def load_models(config_path):
    config = json.loads(Path(config_path).read_text())
    model = FlowSDE(INPUT_DIM, ACTION_DIM, STATE_DIM, jr.PRNGKey(config["seed"]),
                    dt=config["dt"], width=config["width"], depth=3,
                    condition_dim=config["condition_dim"], flow_layers=config["flow_layers"])
    model = eqx.tree_deserialise_leaves(Path(config["checkpoint"]), model)
    stats_file = np.load(Path(config["output"]) / "normalization.npz")
    stats = {name: stats_file[name] for name in stats_file.files}
    log_std = np.load(Path(config["output"]) / "gaussian_log_std.npy")
    return model, GaussianSDE(model.drift, STATE_DIM, config["dt"], log_std), stats, config


def rbf_mmd(x, y):
    joined = np.concatenate((x, y)); distances = np.sum((joined[:, None] - joined[None]) ** 2, -1)
    positive = distances[distances > 0]; bandwidth = np.median(positive) if len(positive) else 1.0
    kernel = lambda a, b: np.exp(-np.sum((a[:, None] - b[None]) ** 2, -1) / max(2 * bandwidth, 1e-8))
    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean())


def offline_metrics(flow, gaussian, stats, config, limit=1000):
    histories, actions, deltas, episodes = load(config["data"])
    mask = np.isin(episodes, config["test_episodes"])
    h, a, d = normalize(histories[mask], actions[mask], deltas[mask], stats)
    h, a, d = map(jnp.asarray, (h[:limit], a[:limit], d[:limit]))
    flow_nll = float(-jnp.mean(jax.vmap(flow.log_prob)(d, h, a)))
    gaussian_nll = float(-jnp.mean(jax.vmap(gaussian.log_prob)(d, h, a)))
    deterministic = jax.vmap(flow.mean_delta)(h, a)
    keys = jr.split(jr.PRNGKey(991), len(h))
    samples = jax.vmap(flow.step)(h, a, keys)
    return {"flow_nll": flow_nll, "gaussian_nll": gaussian_nll,
            "deterministic_rmse": float(jnp.sqrt(jnp.mean((deterministic - d) ** 2))),
            "flow_mmd": rbf_mmd(np.asarray(samples), np.asarray(d))}


def run_controller(name, planner, env, target, episodes, steps, seed):
    rows = []
    for episode in range(episodes):
        state = env.reset(randomize=True); history = HistoryBuffer.from_state(state); planner.reset()
        collision, latencies = False, []
        for step in range(steps):
            started = time.perf_counter()
            if name == "Sim-MPPI": action, _ = planner.command(history.vector, target)
            else: action, _ = planner.command(history.vector, target, jr.PRNGKey(seed + episode * steps + step))
            latencies.append(time.perf_counter() - started)
            state = env.step(action, noisy=True); history.append(state, action)
            collision |= bool(np.any(state[-2:] < 0))
        position_error = float(np.linalg.norm(state[OBJECT_POSITION] - target[:3]))
        yaw_error = float(abs(np.arctan2(np.sin(state[OBJECT_YAW] - target[3]),
                                         np.cos(state[OBJECT_YAW] - target[3]))))
        rows.append({"controller": name, "episode": episode,
                     "success": position_error < 0.02 and yaw_error < np.deg2rad(5) and not collision,
                     "position_error_m": position_error, "yaw_error_rad": yaw_error,
                     "collision": collision, "latency_ms": 1000 * np.mean(latencies),
                     "frequency_hz": 1 / np.mean(latencies)})
    return rows


def evaluate(args):
    flow, gaussian, stats, training = load_models(args.config)
    cfg = MPPIConfig(args.horizon, args.samples, args.temperature)
    target = np.asarray((*args.target, args.target_yaw), dtype=np.float32)
    controller_factories = {
        "FlowSDE-MPPI": lambda env: MPPIPlanner(NormalizedDynamics(flow, stats), cfg),
        "DetMLP-MPPI": lambda env: MPPIPlanner(NormalizedDynamics(flow, stats), cfg, deterministic=True),
        "GaussSDE-MPPI": lambda env: MPPIPlanner(NormalizedDynamics(gaussian, stats), cfg),
        "Sim-MPPI": lambda env: SimulatorMPPIPlanner(env, cfg, args.seed),
    }
    rows = []
    for order, (name, factory) in enumerate(controller_factories.items()):
        env = NoisyPushingEnv(args.model, seed=args.seed + order)
        rows.extend(run_controller(name, factory(env), env, target, args.episodes,
                                   args.steps, args.seed + order * 100000))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name in controller_factories:
        selected = [row for row in rows if row["controller"] == name]
        summary[name] = {"success_rate": float(np.mean([r["success"] for r in selected])),
                         "final_position_error_m": float(np.mean([r["position_error_m"] for r in selected])),
                         "yaw_error_rad": float(np.mean([r["yaw_error_rad"] for r in selected])),
                         "latency_ms": float(np.mean([r["latency_ms"] for r in selected])),
                         "frequency_hz": float(np.mean([r["frequency_hz"] for r in selected]))}
    result = {"closed_loop": summary, "offline": offline_metrics(flow, gaussian, stats, training)}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    names = list(summary); figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(names, [100 * summary[n]["success_rate"] for n in names]); axes[0].set_ylabel("success [%]")
    axes[1].bar(names, [summary[n]["final_position_error_m"] for n in names]); axes[1].set_ylabel("final error [m]")
    axes[2].bar(names, [summary[n]["latency_ms"] for n in names]); axes[2].set_ylabel("latency [ms]")
    for axis in axes: axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=.2)
    figure.tight_layout(); figure.savefig(args.output / "comparison.png", dpi=180); plt.close(figure)
    print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=150); p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--samples", type=int, default=1000); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--target", type=float, nargs=3, default=(0.68, 0.0, 0.055))
    p.add_argument("--target-yaw", type=float, default=0.0); p.add_argument("--seed", type=int, default=2026)
    return p


if __name__ == "__main__": evaluate(parser().parse_args())
