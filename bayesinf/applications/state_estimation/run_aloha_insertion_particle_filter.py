"""Run a bootstrap particle filter with the fitted ALOHA controlled dynamics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.scipy.special import logsumexp

from .aloha_insertion_dataset import (
    load_aloha_insertion_dataset,
    read_aloha_insertion_parquet,
)
from .evaluation_plots import save_state_estimation_plots
from .filtering_baselines import _observation_log_likelihood
from .run_aloha_insertion_ekf import ControlledResidualDynamics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/aloha_sim_insertion_human")
    parser.add_argument(
        "--output_dir",
        default=(
            "results/results_state_estimation/Aloha_sim_insertion/"
            "pseudo_process_uncertainty"
        ),
    )
    parser.add_argument("--obs_std", type=float, default=0.1)
    parser.add_argument(
        "--obs_noise_type",
        choices=("gaussian", "student_t", "gmm"),
        default="gaussian",
    )
    parser.add_argument("--obs_df", type=float, default=4.0)
    parser.add_argument("--gmm_weights", default="0.70,0.15,0.15")
    parser.add_argument("--gmm_offsets", default="0.0,2.5,-2.5")
    parser.add_argument("--gmm_stds", default="0.50,1.50,1.50")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--window_stride", type=int, default=25)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dynamics_epochs", type=int, default=300)
    parser.add_argument("--dynamics_batch_size", type=int, default=256)
    parser.add_argument("--dynamics_learning_rate", type=float, default=3e-4)
    parser.add_argument("--process_floor", type=float, default=1e-5)
    parser.add_argument("--initial_covariance", type=float, default=1e-4)
    parser.add_argument("--num_particles", type=int, default=500)
    parser.add_argument(
        "--reuse_ekf_dynamics",
        action="store_true",
        help=(
            "Load controlled_dynamics.eqx and the process covariance from "
            "ekf_results.npz instead of fitting a separate PF dynamics model."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pseudo_process_uncertainty", action="store_true")
    parser.add_argument("--disturbance_dim", type=int, default=4)
    parser.add_argument("--rho_d", type=float, default=0.98)
    parser.add_argument("--sigma_d", type=float, default=0.05)
    parser.add_argument("--regime_gain", type=float, default=2.0)
    parser.add_argument("--mixing_scale", type=float, default=0.35)
    args = parser.parse_args()
    if args.num_particles < 2:
        parser.error("--num_particles must be at least 2")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gmm_weights = tuple(float(v) for v in args.gmm_weights.split(","))
    gmm_offsets = tuple(float(v) for v in args.gmm_offsets.split(","))
    gmm_stds = tuple(float(v) for v in args.gmm_stds.split(","))

    splits, metadata = load_aloha_insertion_dataset(
        args.data_path,
        horizon=args.horizon,
        stride=args.window_stride,
        obs_std=args.obs_std,
        obs_noise_type=args.obs_noise_type,
        obs_df=args.obs_df,
        gmm_weights=gmm_weights,
        gmm_offsets=gmm_offsets,
        gmm_stds=gmm_stds,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        normalize=True,
        clean_initial_state=True,
        pseudo_process_uncertainty=args.pseudo_process_uncertainty,
        disturbance_dim=args.disturbance_dim,
        rho_d=args.rho_d,
        sigma_d=args.sigma_d,
        regime_gain=args.regime_gain,
        mixing_scale=args.mixing_scale,
    )
    test = splits["test"]
    d_x, d_u = int(test["x"].shape[-1]), int(test["u"].shape[-1])
    raw_for_controls = read_aloha_insertion_parquet(args.data_path)
    normalized_controls = (
        raw_for_controls["action"] - metadata.action_mean
    ) / metadata.action_std
    filter_controls = []
    for episode_id, start_frame in zip(
        np.asarray(test["episode_index"]), np.asarray(test["start_frame"])
    ):
        episode_rows = np.flatnonzero(
            raw_for_controls["episode_index"] == int(episode_id)
        )
        episode_frames = raw_for_controls["frame_index"][episode_rows]
        first = np.flatnonzero(episode_frames == int(start_frame) - 1)
        if len(first) != 1:
            raise ValueError(
                f"Could not find preceding action for episode={episode_id}, "
                f"start_frame={start_frame}"
            )
        local_start = int(first[0])
        selected = episode_rows[local_start : local_start + args.horizon]
        if len(selected) != args.horizon:
            raise ValueError("Insufficient preceding controls for test window")
        filter_controls.append(normalized_controls[selected])
    filter_controls = jnp.asarray(np.stack(filter_controls))
    suffix = f"_{args.obs_noise_type}" if args.obs_noise_type != "gaussian" else ""
    dynamics_path = output / f"particle_filter_dynamics{suffix}.eqx"
    dynamics_stats_path = output / f"particle_filter_dynamics{suffix}.npz"
    model = ControlledResidualDynamics(
        d_x, d_u, args.width, args.depth, jr.PRNGKey(1)
    )
    if args.reuse_ekf_dynamics:
        shared_dynamics = output / "controlled_dynamics.eqx"
        shared_stats = output / "ekf_results.npz"
        if not shared_dynamics.exists() or not shared_stats.exists():
            raise FileNotFoundError(
                "--reuse_ekf_dynamics requires controlled_dynamics.eqx and "
                "ekf_results.npz from run_aloha_insertion_ekf"
            )
        model = eqx.tree_deserialise_leaves(shared_dynamics, model)
        process_covariance = np.load(shared_stats)["process_covariance"]
        process_var = jnp.asarray(np.diag(process_covariance))
        print(f"Loaded shared EKF dynamics from {shared_dynamics}")
    elif dynamics_path.exists() and dynamics_stats_path.exists():
        model = eqx.tree_deserialise_leaves(dynamics_path, model)
        process_var = jnp.asarray(np.load(dynamics_stats_path)["process_variance"])
        print(f"Loaded particle-filter dynamics from {dynamics_path}")
    else:
        raw = read_aloha_insertion_parquet(args.data_path)
        state = (
            raw["observation.state"] - metadata.state_mean
        ) / metadata.state_std
        control = (raw["action"] - metadata.action_mean) / metadata.action_std
        episode = raw["episode_index"]
        consecutive = episode[:-1] == episode[1:]
        train_rows = (
            np.isin(episode[:-1], metadata.train_episodes) & consecutive
        )
        val_rows = np.isin(episode[:-1], metadata.val_episodes) & consecutive
        x_train = jnp.asarray(state[:-1][train_rows])
        u_train = jnp.asarray(control[:-1][train_rows])
        next_train = jnp.asarray(state[1:][train_rows])
        x_val = jnp.asarray(state[:-1][val_rows])
        u_val = jnp.asarray(control[:-1][val_rows])
        next_val = jnp.asarray(state[1:][val_rows])
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(args.dynamics_learning_rate, weight_decay=1e-5),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        def dynamics_loss(current, xb, ub, target):
            return jnp.mean((jax.vmap(current)(xb, ub) - target) ** 2)

        @eqx.filter_jit
        def dynamics_step(current, state_opt, xb, ub, target):
            loss, grads = eqx.filter_value_and_grad(dynamics_loss)(
                current, xb, ub, target
            )
            updates, state_opt = optimizer.update(
                grads, state_opt, eqx.filter(current, eqx.is_inexact_array)
            )
            return eqx.apply_updates(current, updates), state_opt, loss

        key = jr.PRNGKey(args.seed + 401)
        best_model, best_val, stale = model, np.inf, 0
        for epoch in range(args.dynamics_epochs):
            order = np.asarray(jr.permutation(jr.fold_in(key, epoch), len(x_train)))
            for start in range(0, len(order), args.dynamics_batch_size):
                idx = order[start : start + args.dynamics_batch_size]
                model, opt_state, _loss = dynamics_step(
                    model, opt_state, x_train[idx], u_train[idx], next_train[idx]
                )
            val = float(dynamics_loss(model, x_val, u_val, next_val))
            if val < best_val:
                best_model, best_val, stale = model, val, 0
            else:
                stale += 1
            if epoch % 20 == 0:
                print(f"dynamics epoch={epoch} val_mse={val:.7f} best={best_val:.7f}")
            if stale >= 30:
                break
        model = best_model
        residual = np.asarray(jax.vmap(model)(x_train, u_train) - next_train)
        process_var = jnp.asarray(
            np.maximum(residual.var(axis=0), args.process_floor)
        )
        eqx.tree_serialise_leaves(dynamics_path, model)
        np.savez(
            dynamics_stats_path,
            process_variance=np.asarray(process_var),
            validation_mse=best_val,
        )
        print(f"Saved particle-filter dynamics; validation MSE={best_val:.7f}")
    process_std = jnp.sqrt(jnp.maximum(process_var, 0.0))

    def filter_sequence(y, controls, x0, key):
        key, init_key = jr.split(key)
        particles = x0 + jnp.sqrt(args.initial_covariance) * jr.normal(
            init_key, (args.num_particles, d_x)
        )

        def step(carry, inputs):
            current, step_key = carry
            observation, control = inputs
            step_key, noise_key, resample_key = jr.split(step_key, 3)
            propagated = jax.vmap(lambda state: model(state, control))(current)
            propagated += process_std * jr.normal(noise_key, propagated.shape)
            residual = observation[None, :] - propagated
            log_weights = _observation_log_likelihood(
                residual,
                args.obs_noise_type,
                args.obs_std,
                args.obs_df,
                gmm_weights,
                gmm_offsets,
                gmm_stds,
            )
            log_weights -= logsumexp(log_weights)
            indices = jr.categorical(
                resample_key, log_weights, shape=(args.num_particles,)
            )
            particles = propagated[indices]
            mean = jnp.mean(particles, axis=0)
            centered = particles - mean
            covariance = centered.T @ centered / (args.num_particles - 1)
            return (particles, step_key), (mean, covariance, particles)

        _, result = jax.lax.scan(step, (particles, key), (y, controls))
        return result

    batch = int(test["x"].shape[0])
    keys = jr.split(jr.PRNGKey(args.seed + 299), batch)
    means, covs, particles = jax.jit(jax.vmap(filter_sequence))(
        test["y"], filter_controls, test["x0"], keys
    )
    means = np.asarray(means)
    covs = np.asarray(covs)
    # Convert (batch,time,particle,state) to the plotting convention.
    samples = np.asarray(particles).transpose(0, 2, 1, 3)
    truth = np.asarray(test["x"])
    rmse = float(np.mean(np.sqrt(np.mean((means - truth) ** 2, axis=(1, 2)))))
    mae = float(np.mean(np.abs(means - truth)))
    lo = np.quantile(samples, 0.05, axis=1)
    hi = np.quantile(samples, 0.95, axis=1)
    coverage = float(np.mean((truth >= lo) & (truth <= hi)))

    save_state_estimation_plots(
        output,
        f"bootstrap_particle_filter{suffix}",
        truth,
        np.asarray(test["y"]),
        samples,
        dt=0.02,
    )
    np.savez_compressed(
        output / f"particle_filter_results{suffix}.npz",
        truth=truth,
        observation=np.asarray(test["y"]),
        transition_controls=np.asarray(filter_controls),
        posterior_mean=means,
        posterior_covariance=covs,
        posterior_particles=samples,
        process_variance=np.asarray(process_var),
        observation_variance=args.obs_std**2,
        episode_index=np.asarray(test["episode_index"]),
        start_frame=np.asarray(test["start_frame"]),
    )
    with (output / f"particle_filter_metrics{suffix}.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "rmse", "mae", "coverage_90", "num_particles"))
        writer.writerow(
            (
                f"bootstrap_particle_filter{suffix}",
                rmse,
                mae,
                coverage,
                args.num_particles,
            )
        )
    print(
        f"PF RMSE={rmse:.6f} MAE={mae:.6f} "
        f"coverage90={coverage*100:.2f}% particles={args.num_particles}"
    )


if __name__ == "__main__":
    main()
