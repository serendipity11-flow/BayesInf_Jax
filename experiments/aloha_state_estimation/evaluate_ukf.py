"""Run a UKF using the controlled dynamics fitted by the ALOHA EKF job."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .utils.dataset import load_aloha_insertion_dataset
from bayesinf.applications.state_estimation.utils.plots import save_state_estimation_plots
from .evaluate_ekf import ControlledResidualDynamics


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
    parser.add_argument("--initial_covariance", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=0.0)
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
    dynamics_path = output / "controlled_dynamics.eqx"
    ekf_results_path = output / "ekf_results.npz"
    if not dynamics_path.exists() or not ekf_results_path.exists():
        raise FileNotFoundError(
            "UKF requires controlled_dynamics.eqx and ekf_results.npz from the EKF job"
        )

    splits, _ = load_aloha_insertion_dataset(
        args.data_path,
        horizon=args.horizon,
        stride=args.window_stride,
        obs_std=args.obs_std,
        obs_noise_type=args.obs_noise_type,
        obs_df=args.obs_df,
        gmm_weights=tuple(float(v) for v in args.gmm_weights.split(",")),
        gmm_offsets=tuple(float(v) for v in args.gmm_offsets.split(",")),
        gmm_stds=tuple(float(v) for v in args.gmm_stds.split(",")),
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
    d_x = int(splits["test"]["x"].shape[-1])
    d_u = int(splits["test"]["u"].shape[-1])
    model = ControlledResidualDynamics(
        d_x, d_u, args.width, args.depth, jax.random.PRNGKey(1)
    )
    model = eqx.tree_deserialise_leaves(dynamics_path, model)
    saved = np.load(ekf_results_path)
    Q = jnp.asarray(saved["process_covariance"])
    R = jnp.asarray(saved["observation_covariance"])
    I = jnp.eye(d_x)

    lam = args.alpha**2 * (d_x + args.kappa) - d_x
    scale = d_x + lam
    wm = np.full(2 * d_x + 1, 1.0 / (2.0 * scale), dtype=np.float32)
    wc = wm.copy()
    wm[0] = lam / scale
    wc[0] = wm[0] + (1.0 - args.alpha**2 + args.beta)
    wm, wc = jnp.asarray(wm), jnp.asarray(wc)

    def sigma_points(mean, cov):
        chol = jnp.linalg.cholesky(scale * (cov + 1e-7 * I))
        offsets = chol.T
        return jnp.concatenate((mean[None], mean + offsets, mean - offsets), axis=0)

    def weighted_cov(points, mean, weights):
        centered = points - mean
        return jnp.einsum("s,si,sj->ij", weights, centered, centered)

    def ukf_sequence(y, controls, x0):
        def step(carry, inputs):
            mean, cov = carry
            observation, control = inputs
            sigma = sigma_points(mean, cov)
            propagated = jax.vmap(lambda point: model(point, control))(sigma)
            pred_mean = jnp.einsum("s,si->i", wm, propagated)
            pred_cov = weighted_cov(propagated, pred_mean, wc) + Q

            # For h(x)=x and additive process noise, the measurement and
            # cross-covariances must include Q through the complete P^-.
            obs_mean = pred_mean
            obs_cov = pred_cov + R
            cross_cov = pred_cov
            gain = jnp.linalg.solve(obs_cov, cross_cov.T).T
            filt_mean = pred_mean + gain @ (observation - obs_mean)
            filt_cov = pred_cov - gain @ obs_cov @ gain.T
            filt_cov = 0.5 * (filt_cov + filt_cov.T) + 1e-8 * I
            return (filt_mean, filt_cov), (filt_mean, filt_cov)

        init = (x0, I * args.initial_covariance)
        _, result = jax.lax.scan(step, init, (y, controls))
        return result

    test = splits["test"]
    means, covs = jax.jit(jax.vmap(ukf_sequence))(
        test["y"], test["u"], test["x0"]
    )
    means, covs = np.asarray(means), np.asarray(covs)
    truth = np.asarray(test["x"])
    rmse = float(np.mean(np.sqrt(np.mean((means - truth) ** 2, axis=(1, 2)))))
    mae = float(np.mean(np.abs(means - truth)))
    std = np.sqrt(np.maximum(np.diagonal(covs, axis1=-2, axis2=-1), 0))
    lo, hi = means - 1.644854 * std, means + 1.644854 * std
    coverage = float(np.mean((truth >= lo) & (truth <= hi)))

    rng = np.random.default_rng(args.seed + 199)
    samples = means[:, None] + std[:, None] * rng.normal(
        size=(len(means), args.posterior_samples, means.shape[1], means.shape[2])
    )
    save_state_estimation_plots(
        output,
        "unscented_kalman_filter",
        truth,
        np.asarray(test["y"]),
        samples,
        dt=0.02,
    )
    np.savez(
        output / "ukf_results.npz",
        truth=truth,
        observation=np.asarray(test["y"]),
        posterior_mean=means,
        posterior_covariance=covs,
        process_covariance=np.asarray(Q),
        observation_covariance=np.asarray(R),
        episode_index=np.asarray(test["episode_index"]),
        start_frame=np.asarray(test["start_frame"]),
        alpha=args.alpha,
        beta=args.beta,
        kappa=args.kappa,
    )
    with (output / "ukf_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "rmse", "mae", "coverage_90", "alpha", "beta", "kappa"))
        writer.writerow(
            ("unscented_kalman_filter", rmse, mae, coverage, args.alpha, args.beta, args.kappa)
        )
    print(f"UKF RMSE={rmse:.6f} MAE={mae:.6f} coverage90={coverage*100:.2f}%")


if __name__ == "__main__":
    main()
