"""Train a controlled neural transition and run an EKF on ALOHA insertion."""

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

from .utils.dataset import (
    generate_pseudo_process_trajectory,
    load_aloha_insertion_dataset,
    read_aloha_insertion_parquet,
)
from bayesinf.applications.state_estimation.utils.plots import save_state_estimation_plots


class ControlledResidualDynamics(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, d_x: int, d_u: int, width: int, depth: int, key):
        self.net = eqx.nn.MLP(
            d_x + d_u,
            d_x,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, x, u):
        return x + self.net(jnp.concatenate((x, u)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/aloha_sim_insertion_human")
    parser.add_argument(
        "--output_dir",
        default="results/results_state_estimation/Aloha_sim_insertion/dataset_noisy",
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--process_floor", type=float, default=1e-5)
    parser.add_argument("--initial_covariance", type=float, default=1e-4)
    parser.add_argument("--posterior_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pseudo_process_uncertainty", action="store_true")
    parser.add_argument("--disturbance_dim", type=int, default=4)
    parser.add_argument("--rho_d", type=float, default=0.98)
    parser.add_argument("--sigma_d", type=float, default=0.05)
    parser.add_argument("--regime_gain", type=float, default=2.0)
    parser.add_argument("--mixing_scale", type=float, default=0.35)
    args = parser.parse_args()

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

    raw = read_aloha_insertion_parquet(args.data_path)
    q_raw = raw["observation.state"]
    if args.pseudo_process_uncertainty:
        q_raw, _, _, _ = generate_pseudo_process_trajectory(
            q_raw,
            raw["action"],
            raw["episode_index"],
            seed=args.seed,
            dt=0.02,
            disturbance_dim=args.disturbance_dim,
            rho_d=args.rho_d,
            sigma_d=args.sigma_d,
            regime_gain=args.regime_gain,
            mixing_scale=args.mixing_scale,
            regime_reference=metadata.disturbance_config["regime_reference"],
        )
    q = (q_raw - metadata.state_mean) / metadata.state_std
    u = (raw["action"] - metadata.action_mean) / metadata.action_std
    episode = raw["episode_index"]
    consecutive = episode[:-1] == episode[1:]
    train_row = np.isin(episode[:-1], metadata.train_episodes) & consecutive
    val_row = np.isin(episode[:-1], metadata.val_episodes) & consecutive
    x_train = jnp.asarray(q[:-1][train_row])
    u_train = jnp.asarray(u[:-1][train_row])
    next_train = jnp.asarray(q[1:][train_row])
    x_val = jnp.asarray(q[:-1][val_row])
    u_val = jnp.asarray(u[:-1][val_row])
    next_val = jnp.asarray(q[1:][val_row])

    key = jr.PRNGKey(args.seed)
    model = ControlledResidualDynamics(
        q.shape[1], u.shape[1], args.width, args.depth, jr.fold_in(key, 1)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.learning_rate, weight_decay=1e-5),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(m, xb, ub, yb):
        pred = jax.vmap(m)(xb, ub)
        return jnp.mean((pred - yb) ** 2)

    @eqx.filter_jit
    def train_step(m, state, xb, ub, yb):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, xb, ub, yb)
        updates, state = optimizer.update(
            grads, state, eqx.filter(m, eqx.is_inexact_array)
        )
        return eqx.apply_updates(m, updates), state, loss

    best_model = model
    best_val = np.inf
    patience = 30
    stale = 0
    n = len(x_train)
    for epoch in range(args.epochs):
        perm = np.asarray(jr.permutation(jr.fold_in(key, epoch + 10), n))
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            model, opt_state, _ = train_step(
                model, opt_state, x_train[idx], u_train[idx], next_train[idx]
            )
        val = float(loss_fn(model, x_val, u_val, next_val))
        if val < best_val:
            best_val = val
            best_model = model
            stale = 0
            eqx.tree_serialise_leaves(output / "controlled_dynamics.eqx", model)
        else:
            stale += 1
        if epoch % 20 == 0:
            print(f"epoch={epoch} val_mse={val:.7f} best={best_val:.7f}")
        if stale >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    model = best_model

    residual = np.asarray(jax.vmap(model)(x_train, u_train) - next_train)
    q_diag = np.maximum(residual.var(axis=0), args.process_floor)
    Q = jnp.diag(jnp.asarray(q_diag))
    observation_variance = args.obs_std**2
    if args.obs_noise_type == "gmm":
        weights = np.asarray(gmm_weights) / np.sum(gmm_weights)
        offsets = args.obs_std * np.asarray(gmm_offsets)
        scales = args.obs_std * np.asarray(gmm_stds)
        mixture_mean = np.sum(weights * offsets)
        observation_variance = float(
            np.sum(weights * (scales**2 + (offsets - mixture_mean) ** 2))
        )
    elif args.obs_noise_type == "student_t":
        observation_variance = (
            args.obs_std**2 * args.obs_df / (args.obs_df - 2)
            if args.obs_df > 2
            else args.obs_std**2
        )
    R = jnp.eye(q.shape[1]) * observation_variance
    I = jnp.eye(q.shape[1])

    def ekf_sequence(y, controls, x0):
        def step(carry, inputs):
            mean, cov = carry
            observation, control = inputs
            F = jax.jacfwd(model, argnums=0)(mean, control)
            pred_mean = model(mean, control)
            pred_cov = F @ cov @ F.T + Q
            innovation_cov = pred_cov + R
            gain = jnp.linalg.solve(innovation_cov, pred_cov.T).T
            filt_mean = pred_mean + gain @ (observation - pred_mean)
            left = I - gain
            filt_cov = left @ pred_cov @ left.T + gain @ R @ gain.T
            filt_cov = 0.5 * (filt_cov + filt_cov.T)
            return (filt_mean, filt_cov), (filt_mean, filt_cov)

        init = (x0, I * args.initial_covariance)
        _, result = jax.lax.scan(step, init, (y, controls))
        return result

    test = splits["test"]
    means, covs = jax.jit(jax.vmap(ekf_sequence))(
        test["y"], test["u"], test["x0"]
    )
    means = np.asarray(means)
    covs = np.asarray(covs)
    truth = np.asarray(test["x"])
    rmse = float(np.mean(np.sqrt(np.mean((means - truth) ** 2, axis=(1, 2)))))
    mae = float(np.mean(np.abs(means - truth)))
    std = np.sqrt(np.maximum(np.diagonal(covs, axis1=-2, axis2=-1), 0))
    lo, hi = means - 1.644854 * std, means + 1.644854 * std
    coverage = float(np.mean((truth >= lo) & (truth <= hi)))

    rng = np.random.default_rng(args.seed + 99)
    samples = means[:, None] + std[:, None] * rng.normal(
        size=(len(means), args.posterior_samples, means.shape[1], means.shape[2])
    )
    save_state_estimation_plots(
        output,
        "extended_kalman_filter",
        truth,
        np.asarray(test["y"]),
        samples,
        dt=0.02,
    )
    np.savez(
        output / "ekf_results.npz",
        truth=truth,
        observation=np.asarray(test["y"]),
        posterior_mean=means,
        posterior_covariance=covs,
        process_covariance=np.asarray(Q),
        observation_covariance=np.asarray(R),
        episode_index=np.asarray(test["episode_index"]),
        start_frame=np.asarray(test["start_frame"]),
    )
    with (output / "ekf_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "rmse", "mae", "coverage_90", "val_dynamics_mse"))
        writer.writerow(("extended_kalman_filter", rmse, mae, coverage, best_val))
    print(f"EKF RMSE={rmse:.6f} MAE={mae:.6f} coverage90={coverage*100:.2f}%")


if __name__ == "__main__":
    main()
