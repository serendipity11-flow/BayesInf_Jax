"""Compare exact-MuJoCo and simulator-trained FlowSDE MPPI on one ALOHA plant."""

from __future__ import annotations

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
import pandas as pd

from experiments.aloha_optimal_control.data import load_episodes, split_episode_indices
from bayesinf.applications.optimal_control.sampling_mpc import MPPIController, SamplingMPCConfig
from experiments.aloha_optimal_control.train_flow_sde import build_model
from .mujoco_aloha import DEFAULT_XML, MujocoAlohaPlant


class NoisyMujocoPlant:
    """Exact MuJoCo plant with optional additive Gaussian state noise."""

    def __init__(self, xml, control_dt, state_noise_std=0.0, seed=0):
        self.plant = MujocoAlohaPlant(xml, control_dt)
        self.state_noise_std = float(state_noise_std)
        self.rng = np.random.default_rng(seed)

    @property
    def history_len(self):
        return 2

    @property
    def action_low(self):
        return self.plant.action_low

    @property
    def action_high(self):
        return self.plant.action_high

    def reset(self, state):
        return self.plant.reset(state)

    def step(self, action):
        next_state = self.plant.step(action)
        if self.state_noise_std > 0:
            next_state = next_state + self.rng.normal(
                scale=self.state_noise_std, size=next_state.shape
            ).astype(np.float32)
            self.plant.reset(next_state)
        return next_state


class FlowSDEDynamicsAdapter:
    """Normalize ALOHA states/actions around the trained FlowSDE interface."""

    def __init__(self, model, x_mean, x_std, u_mean, u_std, solver_name):
        self.model = model
        self.x_mean = jnp.asarray(x_mean)
        self.x_std = jnp.asarray(x_std)
        self.u_mean = jnp.asarray(u_mean)
        self.u_std = jnp.asarray(u_std)
        self.solver_name = solver_name

    def sample_rollout(self, state_history, previous_controls, future_controls, key,
                       n_particles=1, start_step=0, diffusion_temperature=1.0):
        del previous_controls
        normalized_history = (
            jnp.asarray(state_history) - self.x_mean
        ) / self.x_std
        x0 = normalized_history[-1]
        controls = (jnp.asarray(future_controls) - self.u_mean) / self.u_std
        paths = self.model.sample_rollout(
            x0, controls, key, n_particles=n_particles,
            start_step=start_step, diffusion_temperature=diffusion_temperature,
            solver_name=self.solver_name,
            state_history=normalized_history[:-1],
        )
        return paths * self.x_std + self.x_mean


class ExactMujocoRollout:
    """Deterministic rollout model backed by the same Menagerie MJCF as the plant."""

    def __init__(self, xml, control_dt):
        self.plant = MujocoAlohaPlant(xml, control_dt)

    def rollout(self, state, controls):
        self.plant.reset(np.asarray(state))
        return np.stack([self.plant.step(action) for action in np.asarray(controls)])


class ExactMujocoMPPIController:
    """Existing MPPI update rule with external MuJoCo candidate evaluation.

    MuJoCo's native CPU API cannot execute inside JAX ``jit``. This class keeps
    the same perturbation, clipping, normalized-cost softmax, and plan update as
    ``MPPIController`` while delegating only candidate rollouts to MuJoCo.
    """

    def __init__(self, model, low, high, cost_fn, config):
        self.model, self.low, self.high = model, jnp.asarray(low), jnp.asarray(high)
        self.cost_fn, self.config = cost_fn, config
        self.plan = jnp.zeros((config.horizon, len(low)))

    def command(self, history, previous, key):
        del previous
        cfg = self.config
        nominal = jnp.concatenate([self.plan[1:], self.plan[-1:]])
        for iteration in range(cfg.iterations):
            noise = cfg.initial_std * jr.normal(
                jr.fold_in(key, iteration), (cfg.num_candidates,) + nominal.shape
            )
            candidates = jnp.clip(nominal + noise, self.low, self.high)
            costs = np.asarray([
                self.cost_fn(self.model.rollout(history[-1], candidate), candidate)
                for candidate in np.asarray(candidates)
            ])
            scale = max(float(costs.std()), 1e-6) if cfg.normalize_costs else 1.0
            weights = jax.nn.softmax(jnp.asarray(-(costs - costs.min()) / (cfg.temperature * scale)))
            nominal = jnp.clip(
                nominal + jnp.sum(weights[:, None, None] * (candidates - nominal), axis=0),
                self.low, self.high,
            )
        self.plan = nominal
        return nominal[0], nominal.copy()


def run_closed_loop(controller, plant, initial, goal, steps, seed, x_std):
    plant.reset(initial)
    history = jnp.stack([jnp.asarray(initial), jnp.asarray(initial)])
    previous = jnp.empty((0, 14))
    controller.plan = jnp.broadcast_to(jnp.asarray(initial[:14]), controller.plan.shape)
    states, actions, runtimes = [], [], []
    for step in range(steps):
        start = time.perf_counter()
        action, _ = controller.command(history, previous, jr.fold_in(jr.PRNGKey(seed), step))
        jax.block_until_ready(action)
        runtimes.append(time.perf_counter() - start)
        state = plant.step(np.asarray(action))
        history = jnp.stack([history[-1], jnp.asarray(state)])
        states.append(state); actions.append(np.asarray(action))
    states, actions = np.stack(states), np.stack(actions)
    error = (states - goal) / x_std
    return states, actions, {
        "final_normalized_rmse": float(np.sqrt(np.mean(error[-1] ** 2))),
        "cumulative_normalized_error": float(np.sum(np.mean(error ** 2, axis=-1))),
        "control_smoothness": float(np.sqrt(np.mean((np.diff(actions, axis=0)) ** 2))),
        "mean_runtime_ms": float(1e3 * np.mean(runtimes[1:])),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cached_rows = {}
    cached_metrics_path = args.output / "case_metrics.csv"
    if args.resume and cached_metrics_path.exists():
        cached = pd.read_csv(cached_metrics_path)
        cached_rows = {
            (int(row["horizon"]), int(row["task"]), str(row["model"])): row.to_dict()
            for _, row in cached.iterrows()
        }
    with args.flow_config.open() as handle:
        flow_config = json.load(handle)
    episodes, actions, episode_ids, state_dim, action_dim = load_episodes(args.data)
    if (state_dim, action_dim) != (28, 14):
        raise ValueError(f"expected simulator state/action dimensions 28/14, got {state_dim}/{action_dim}")
    train_i, _, test_i = split_episode_indices(len(episodes), flow_config["seed"])
    stats = np.load(flow_config["normalization"])
    x_mean, x_std, u_mean, u_std = [stats[k] for k in ("x_mean", "x_std", "u_mean", "u_std")]
    state_data = np.concatenate(episodes)
    plant_template = MujocoAlohaPlant(args.xml, args.control_dt)
    state_low, state_high = state_data.min(0), state_data.max(0)
    action_low, action_high = plant_template.action_low, plant_template.action_high
    flow = build_model(SimpleNamespace(**flow_config), state_dim, action_dim)
    flow = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(flow_config["checkpoint"], flow))
    flow_adapter = FlowSDEDynamicsAdapter(
        flow, x_mean, x_std, u_mean, u_std, flow_config["solver"],
    )
    rows = []
    episode_rollouts = {}
    horizon_grid = [int(item) for item in args.horizon_grid.split(",") if item.strip()] if args.horizon_grid else [args.horizon]
    if args.episode_scope == "all":
        episode_indices = np.arange(len(episodes))
    elif args.episode_scope == "train":
        episode_indices = np.asarray(train_i)
    else:
        episode_indices = np.asarray(test_i)

    for horizon in horizon_grid:
        cfg = SamplingMPCConfig(
            horizon=horizon, num_candidates=args.candidates,
            num_particles=args.particles, iterations=args.iterations,
            initial_std=jnp.asarray(args.exploration_fraction * (action_high - action_low)),
            temperature=args.temperature, risk_weight=args.risk_weight,
            smoothness_weight=args.smoothness_weight,
            smoothness_scale=jnp.asarray(x_std[:action_dim]),
            normalize_costs=True,
        )
        for task, episode_index in enumerate(episode_indices[:args.tasks if args.tasks > 0 else len(episode_indices)]):
            initial = episodes[episode_index][args.start]
            goal = episodes[episode_index][args.start + args.goal_offset]
            weights = np.ones(state_dim, dtype=np.float32); weights[14:] = args.velocity_weight

            def numpy_cost(paths, controls):
                error = (paths - goal) / x_std
                command = (controls - paths[:, :14]) / x_std[:14]
                stage = np.mean(weights * error ** 2, axis=-1) + args.action_weight * np.mean(command ** 2, axis=-1)
                stage[-1] *= args.terminal_weight
                return float(stage.sum())

            def jax_cost(state, action, step):
                error = (state - jnp.asarray(goal)) / jnp.asarray(x_std)
                command = (action - state[:14]) / jnp.asarray(x_std[:14])
                terminal = jnp.where(step == horizon - 1, args.terminal_weight, 1.0)
                return terminal * jnp.mean(jnp.asarray(weights) * error ** 2) + args.action_weight * jnp.mean(command ** 2)

            controllers = {
                "exact_mujoco": ExactMujocoMPPIController(
                    ExactMujocoRollout(args.xml, args.control_dt), action_low, action_high,
                    numpy_cost, cfg,
                ),
                "flow_sde": MPPIController(
                    flow_adapter, action_low, action_high, jax_cost, cfg,
                ),
            }
            if args.controller != "both":
                controllers = {args.controller: controllers[args.controller]}
            for name, controller in controllers.items():
                cache_key = (horizon, task, name)
                rollout_path = args.output / f"h{horizon}_task{task}_{name}.npz"
                if cache_key in cached_rows and rollout_path.exists():
                    saved = np.load(rollout_path)
                    episode_rollouts.setdefault((horizon, int(episode_ids[episode_index])), {})[name] = {
                        "states": saved["states"],
                        "actions": saved["actions"],
                        "goal": saved["goal"],
                    }
                    rows.append(cached_rows[cache_key])
                    continue
                plant_seed = args.seed + 1000 * task + 100000 * horizon
                plant = NoisyMujocoPlant(args.xml, args.control_dt, args.plant_noise_std, seed=plant_seed)
                states, selected_actions, metrics = run_closed_loop(
                    controller, plant, initial, goal, args.control_steps, plant_seed, x_std,
                )
                episode_rollouts.setdefault((horizon, int(episode_ids[episode_index])), {})[name] = {
                    "states": states,
                    "actions": selected_actions,
                    "goal": goal,
                }
                np.savez_compressed(
                    rollout_path, states=states,
                    actions=selected_actions, initial=initial, goal=goal,
                    episode_id=episode_ids[episode_index], seed=plant_seed,
                    horizon=horizon, plant_noise_std=args.plant_noise_std,
                )
                rows.append({
                    "horizon": horizon,
                    "plant_noise_std": args.plant_noise_std,
                    "task": task,
                    "episode_id": int(episode_ids[episode_index]),
                    "model": name,
                    **metrics,
                })
    with (args.output / "case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for horizon in horizon_grid:
        selected_models = ("exact_mujoco", "flow_sde") if args.controller == "both" else (args.controller,)
        for name in selected_models:
            chosen = [r for r in rows if r["model"] == name and r["horizon"] == horizon]
            summary.append({
                "horizon": horizon,
                "plant_noise_std": args.plant_noise_std,
                "model": name,
                **{
                    key: float(np.mean([r[key] for r in chosen]))
                    for key in ("final_normalized_rmse", "cumulative_normalized_error",
                                "control_smoothness", "mean_runtime_ms")
                },
                **{
                    f"{key}_std": float(np.std([r[key] for r in chosen]))
                    for key in ("final_normalized_rmse", "cumulative_normalized_error",
                                "control_smoothness", "mean_runtime_ms")
                },
            })
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)

    # Sweep plots.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in [("exact_mujoco", "tab:blue"), ("flow_sde", "tab:orange")]:
        if name not in selected_models:
            continue
        chosen = [r for r in summary if r["model"] == name]
        horizons = np.asarray([r["horizon"] for r in chosen])
        final = np.asarray([r["final_normalized_rmse"] for r in chosen])
        final_std = np.asarray([r["final_normalized_rmse_std"] for r in chosen])
        cumulative = np.asarray([r["cumulative_normalized_error"] for r in chosen])
        cumulative_std = np.asarray([r["cumulative_normalized_error_std"] for r in chosen])
        axes[0].plot(horizons, final, marker="o", label=name, color=color)
        axes[0].fill_between(horizons, final - final_std, final + final_std, color=color, alpha=0.2)
        axes[1].plot(horizons, cumulative, marker="o", label=name, color=color)
        axes[1].fill_between(horizons, cumulative - cumulative_std, cumulative + cumulative_std, color=color, alpha=0.2)
    axes[0].set(title="Final normalized RMSE vs horizon", xlabel="horizon", ylabel="final normalized RMSE")
    axes[1].set(title="Cumulative normalized error vs horizon", xlabel="horizon", ylabel="cumulative normalized error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "horizon_sweep_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color in [("exact_mujoco", "tab:blue"), ("flow_sde", "tab:orange")]:
        if name not in selected_models:
            continue
        chosen = [r for r in summary if r["model"] == name]
        horizons = np.asarray([r["horizon"] for r in chosen])
        runtime = np.asarray([r["mean_runtime_ms"] for r in chosen])
        runtime_std = np.asarray([r["mean_runtime_ms_std"] for r in chosen])
        ax.plot(horizons, runtime, marker="o", label=name, color=color)
        ax.fill_between(horizons, runtime - runtime_std, runtime + runtime_std, color=color, alpha=0.2)
    ax.set(title="Controller runtime vs horizon", xlabel="horizon", ylabel="mean runtime [ms]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "horizon_sweep_runtime.png", dpi=180)
    plt.close(fig)

    if args.skip_episode_plots:
        print(json.dumps(summary, indent=2))
        return

    # Per-episode bar plots for the first horizon in the sweep.
    first_horizon = horizon_grid[0]
    episode_fig_dir = args.output / "episode_plots"
    episode_fig_dir.mkdir(parents=True, exist_ok=True)
    per_episode = [r for r in rows if r["horizon"] == first_horizon]
    per_episode.sort(key=lambda r: (r["episode_id"], r["model"]))
    grouped = {}
    for row in per_episode:
        grouped.setdefault(row["episode_id"], {})[row["model"]] = row
    episode_rows = []
    for episode_id, models in grouped.items():
        if "exact_mujoco" not in models or "flow_sde" not in models:
            continue
        episode_rows.append({
            "episode_id": episode_id,
            "exact_mujoco_rmse": models["exact_mujoco"]["final_normalized_rmse"],
            "flow_sde_rmse": models["flow_sde"]["final_normalized_rmse"],
            "exact_mujoco_runtime_ms": models["exact_mujoco"]["mean_runtime_ms"],
            "flow_sde_runtime_ms": models["flow_sde"]["mean_runtime_ms"],
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["exact_mujoco", "flow_sde"]
        rmse = [models[m]["final_normalized_rmse"] for m in labels]
        runtime = [models[m]["mean_runtime_ms"] for m in labels]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, rmse, width=width, label="final normalized RMSE")
        ax.bar(x + width/2, runtime, width=width, label="mean runtime [ms]")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(f"Episode {episode_id} at horizon {first_horizon}")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(episode_fig_dir / f"episode_{episode_id:03d}.png", dpi=180)
        plt.close(fig)

        rollout_key = (first_horizon, episode_id)
        if rollout_key in episode_rollouts:
            exact_states = episode_rollouts[rollout_key]["exact_mujoco"]["states"]
            flow_states = episode_rollouts[rollout_key]["flow_sde"]["states"]
            goal_state = episode_rollouts[rollout_key]["exact_mujoco"]["goal"]
            time_axis = np.arange(exact_states.shape[0]) * args.control_dt
            fig, axes = plt.subplots(7, 4, figsize=(18, 18), sharex=True)
            axes = np.asarray(axes)
            for dim, axis in enumerate(axes.flat[:28]):
                axis.plot(time_axis, exact_states[:, dim], color="tab:blue", lw=1.2, label="exact_mujoco")
                axis.plot(time_axis, flow_states[:, dim], color="tab:orange", lw=1.2, label="flow_sde")
                axis.axhline(goal_state[dim], color="black", ls="--", lw=0.9, label="goal")
                axis.set_title(f"dim {dim}", fontsize=9)
                axis.grid(alpha=0.2)
            for axis in axes.flat[28:]:
                axis.set_visible(False)
            axes[-1, 0].set_xlabel("time [s]")
            axes[-1, 1].set_xlabel("time [s]")
            axes[-1, 2].set_xlabel("time [s]")
            axes[-1, 3].set_xlabel("time [s]")
            axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
            fig.suptitle(f"Episode {episode_id} trajectory overlay (all 28 state dims)")
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            fig.savefig(episode_fig_dir / f"episode_{episode_id:03d}_trajectory.png", dpi=180)
            plt.close(fig)
    pd.DataFrame(episode_rows).to_csv(args.output / "episode_summary.csv", index=False)

    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--data", type=Path, default=Path("data/aloha_mujoco/trajectories.parquet"))
    p.add_argument("--flow-config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/aloha_mujoco/mppi_comparison"))
    p.add_argument("--control-dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--episode-scope", choices=("test", "train", "all"), default="test")
    p.add_argument("--start", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=10)
    p.add_argument("--control-steps", type=int, default=10)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--horizon-grid", type=str, default="",
                   help="Comma-separated horizon values to sweep, e.g. 4,6,8,10")
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--particles", type=int, default=4)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--exploration-fraction", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--risk-weight", type=float, default=0.0,
                   help="Weight on particle-cost standard deviation (mean + weight * std).")
    p.add_argument("--smoothness-weight", type=float, default=0.0,
                   help="Weight on summed normalized squared changes within each candidate action sequence.")
    p.add_argument("--controller", choices=("both", "exact_mujoco", "flow_sde"),
                   default="both", help="Run both controllers or one controller only.")
    p.add_argument("--action-weight", type=float, default=1e-2)
    p.add_argument("--terminal-weight", type=float, default=10.0)
    p.add_argument("--velocity-weight", type=float, default=0.1)
    p.add_argument("--plant-noise-std", type=float, default=0.0,
                   help="Additive Gaussian noise on the plant state after each step.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse complete rollout/metric pairs already in the output directory.")
    p.add_argument("--skip-episode-plots", action="store_true",
                   help="Skip detailed per-episode figures while retaining metrics and sweep plots.")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
