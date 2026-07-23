"""Paired comparison of AmortizedSeqFlow and Kalman-filtered MPC."""
import argparse
import csv
import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .amortized_seq_flow import AmortizedSeqFlow
from .bicycle_dataset import BicycleConfig, _path, wrap_angle
from .mpc_kalman import LinearLateralMPC, NonlinearLateralMPC, LateralKalmanFilter


@eqx.filter_jit
def _flow_chunk(model, observations, reference, past_controls, previous_steering, key):
    return model.sample(
        observations, reference, key, deterministic=True,
        past_controls=past_controls, previous_steering=previous_steering,
    )[0, 0]


def make_flow_chunk_evaluator():
    """Return an isolated JIT cache for one static model configuration."""
    @eqx.filter_jit
    def evaluate(model, observations, reference, past_controls, previous_steering, key):
        return model.sample(
            observations, reference, key, deterministic=True,
            past_controls=past_controls, previous_steering=previous_steering,
        )[0, 0]
    return evaluate


def make_scenario(key, cfg, steps, obs_std):
    """A common unseen path and sensor-noise realization for both methods."""
    kp, kn = jr.split(key)
    _, y_ref, psi_ref, curvature = _path(kp, steps + cfg.horizon + 1, cfg)
    noise = obs_std * jr.normal(kn, (steps,))
    return tuple(np.asarray(x) for x in (y_ref, psi_ref, curvature, noise))


def _advance(ey, ep, delta, kappa, cfg):
    ey = ey + cfg.dt * cfg.velocity * np.sin(ep)
    raw_ep = ep + cfg.dt * (cfg.velocity / cfg.wheelbase * np.tan(delta)
                            - cfg.velocity * kappa)
    return ey, np.arctan2(np.sin(raw_ep), np.cos(raw_ep))


def _initial_history(initial_state, curvature, cfg):
    """Build a dynamically consistent measurement history ending at x_0.

    Before the evaluated controller takes over, zero steering and locally
    constant road curvature are assumed.  Backward inversion of the nonlinear
    error dynamics makes nonzero heading error visible through lateral motion.
    """
    ey, ep = map(float, initial_state)
    history = np.empty(cfg.obs_history)
    history[-1] = ey
    kappa = float(curvature[0])
    for index in range(cfg.obs_history - 2, -1, -1):
        ep = np.arctan2(
            np.sin(ep - cfg.dt * (0.0 - cfg.velocity * kappa)),
            np.cos(ep - cfg.dt * (0.0 - cfg.velocity * kappa)),
        )
        ey = ey - cfg.dt * cfg.velocity * np.sin(ep)
        history[index] = ey
    return history


def run_flow(model, scenario, key, cfg, steps, initial_state=(0.55, 0.12),
             chunk_evaluator=None):
    y_ref, psi_ref, curvature, noise = scenario
    ey, ep = map(float, initial_state)
    history = _initial_history((ey, ep), curvature, cfg)
    control_history = np.zeros(cfg.obs_history)
    steering, lateral, runtimes = [], [], []
    evaluate_chunk = _flow_chunk if chunk_evaluator is None else chunk_evaluator
    ref0 = jnp.stack([jnp.asarray(y_ref[:cfg.horizon] - y_ref[0]),
                      wrap_angle(jnp.asarray(psi_ref[:cfg.horizon] - psi_ref[0])),
                      jnp.asarray(curvature[:cfg.horizon])], axis=-1)[None]
    evaluate_chunk(
        model, jnp.asarray(history)[None, :, None], ref0,
        jnp.asarray(control_history)[None, :, None], jnp.asarray([0.0]), key,
    ).block_until_ready()
    for t in range(steps):
        history = np.r_[history[1:], ey + noise[t]]
        sl = slice(t, t + cfg.horizon)
        ref = jnp.stack([jnp.asarray(y_ref[sl] - y_ref[t]),
                         wrap_angle(jnp.asarray(psi_ref[sl] - psi_ref[t])),
                         jnp.asarray(curvature[sl])], axis=-1)[None]
        tic = time.perf_counter()
        chunk = evaluate_chunk(
            model, jnp.asarray(history)[None, :, None], ref,
            jnp.asarray(control_history)[None, :, None],
            jnp.asarray([control_history[-1]]), key,
        )
        chunk.block_until_ready()
        runtimes.append(time.perf_counter() - tic)
        delta = float(chunk[0])
        control_history = np.r_[control_history[1:], delta]
        ey, ep = _advance(ey, ep, delta, curvature[t], cfg)
        steering.append(delta); lateral.append(ey)
    return np.asarray(lateral), np.asarray(steering), np.asarray(runtimes), 0


def _run_mpc_kf(scenario, cfg, steps, obs_std, controller_type,
                initial_state=(0.55, 0.12)):
    _, _, curvature, noise = scenario
    estimator, controller = LateralKalmanFilter(cfg), controller_type(cfg)
    ey, ep = map(float, initial_state)
    history = _initial_history((ey, ep), curvature, cfg)
    for index, measurement in enumerate(history):
        estimator.update(measurement, obs_std)
        if index < len(history) - 1:
            estimator.predict(0.0, curvature[0])
    steering, lateral, runtimes = [], [], []
    failures = 0
    for t in range(steps):
        estimate = estimator.update(ey + noise[t], obs_std)
        tic = time.perf_counter()
        delta, _ = controller.command(estimate, curvature[t:t + cfg.horizon])
        runtimes.append(time.perf_counter() - tic)
        failures += int(not controller.last_success)
        estimator.predict(delta, curvature[t])
        ey, ep = _advance(ey, ep, delta, curvature[t], cfg)
        steering.append(delta); lateral.append(ey)
    return np.asarray(lateral), np.asarray(steering), np.asarray(runtimes), failures


def run_linear_mpc_kf(scenario, cfg, steps, obs_std,
                      initial_state=(0.55, 0.12)):
    return _run_mpc_kf(
        scenario, cfg, steps, obs_std, LinearLateralMPC, initial_state
    )


def run_nonlinear_mpc_kf(scenario, cfg, steps, obs_std,
                         initial_state=(0.55, 0.12)):
    return _run_mpc_kf(
        scenario, cfg, steps, obs_std, NonlinearLateralMPC, initial_state
    )


# Backward-compatible name used by earlier plotting scripts.
run_mpc_kf = run_nonlinear_mpc_kf


def metrics(method, noise, lateral, steering, runtimes, failures, cfg):
    return {
        "method": method, "noise_std": noise,
        "lateral_rmse": float(np.sqrt(np.mean(lateral**2))),
        "max_abs_lateral": float(np.max(np.abs(lateral))),
        "steering_rms": float(np.sqrt(np.mean(steering**2))),
        "steering_rate_rms": float(np.sqrt(np.mean(np.diff(steering, prepend=steering[0])**2))),
        "max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "constraint_violations": int(np.sum(np.abs(steering) > cfg.delta_max + 1e-6)),
        "mean_runtime_ms": float(1e3 * np.mean(runtimes)), "solver_failures": failures,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("results/optimal_control/lane_flow.eqx"))
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--csv", type=Path, default=Path("results/optimal_control/comparison.csv"))
    p.add_argument("--saturation-factor", type=float, default=1.0)
    args = p.parse_args()
    cfg = BicycleConfig()
    skeleton = AmortizedSeqFlow(
        cfg.horizon, cfg.delta_max, jr.PRNGKey(0),
        saturation_factor=args.saturation_factor,
    )
    with args.checkpoint.open("rb") as f:
        model = eqx.tree_deserialise_leaves(f, skeleton)
    rows = []
    for i, noise in enumerate((0.0, cfg.obs_std, 2.0 * cfg.obs_std)):
        key = jr.fold_in(jr.PRNGKey(args.seed), i)
        scenario = make_scenario(key, cfg, args.steps, noise)
        results = (
            ("amortized_seq_flow", run_flow(model, scenario, key, cfg, args.steps)),
            ("linear_mpc", run_linear_mpc_kf(scenario, cfg, args.steps, noise)),
            ("nonlinear_mpc", run_nonlinear_mpc_kf(scenario, cfg, args.steps, noise)),
        )
        for name, result in results:
            row = metrics(name, noise, *result, cfg)
            rows.append(row)
            print(f"{name:20s} noise={noise:.3f} rmse={row['lateral_rmse']:.4f} "
                  f"rate_rms={row['steering_rate_rms']:.4f} max_delta_deg={row['max_abs_delta_deg']:.2f} "
                  f"violations={row['constraint_violations']} runtime_ms={row['mean_runtime_ms']:.3f}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
