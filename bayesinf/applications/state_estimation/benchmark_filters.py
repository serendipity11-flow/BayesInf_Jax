"""Benchmark Lorenz-96 state estimation against EKF or particle filtering."""

from __future__ import annotations

import argparse
import csv
import os

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .filtering_baselines import bootstrap_particle_filter, extended_kalman_filter
from .lorenz96_dataset import generate_lorenz96_dataset, train_val_test_split


USE_CASES = {
    "amortized_state_flow": ("ekf", "gaussian"),
    "latent_ssm_flow": ("particle_filter", "student_t"),
    "flow_sde": ("particle_filter", "gmm"),
    "cond_stoch_interpolant": ("particle_filter", "gmm"),
}


def _tuple(text):
    return tuple(float(x) for x in text.split(","))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--use-case", choices=USE_CASES, required=True)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--obs-interval", type=int, default=2)
    p.add_argument("--obs-std", type=float, default=0.5)
    p.add_argument("--obs-df", type=float, default=5.0)
    p.add_argument("--noise-type", choices=("gaussian", "student_t", "gmm"))
    p.add_argument("--gmm-weights", type=_tuple, default=(0.70, 0.15, 0.15))
    p.add_argument("--gmm-offsets", type=_tuple, default=(0.0, 2.5, -2.5))
    p.add_argument("--gmm-stds", type=_tuple, default=(0.50, 1.50, 1.50))
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--F", type=float, default=8.0)
    p.add_argument("--spinup-steps", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--num-particles", type=int, default=1000)
    p.add_argument("--process-std", type=float, default=0.05)
    p.add_argument("--initial-std", type=float, default=0.0)
    p.add_argument("--ci", type=float, default=90.0)
    p.add_argument("--n-windows", type=int, default=6)
    p.add_argument("--segment-length", type=int, default=5,
                   help="Only controls the boundary markers used in comparison plots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/results_state_estimation/filter_benchmark")
    args = p.parse_args()

    method, default_noise = USE_CASES[args.use_case]
    noise = args.noise_type or default_noise
    if method == "ekf" and noise != "gaussian":
        p.error("The EKF benchmark requires Gaussian observation noise")

    key = jr.PRNGKey(args.seed)
    data = generate_lorenz96_dataset(
        key, args.n_samples, args.horizon, N=args.N,
        obs_interval=args.obs_interval, obs_std=args.obs_std,
        obs_df=args.obs_df if noise == "student_t" else None,
        obs_noise_type=noise, gmm_weights=args.gmm_weights,
        gmm_offsets=args.gmm_offsets, gmm_stds=args.gmm_stds,
        dt=args.dt, F=args.F, spinup_steps=args.spinup_steps,
    )
    _, _, test = train_val_test_split(
        data, jr.fold_in(key, 1), args.train_frac, args.val_frac
    )
    count = min(args.n_eval, test["x"].shape[0])
    y, x0, truth = test["y"][:count], test["x0"][:count], test["x"][:count]

    if method == "ekf":
        result = extended_kalman_filter(
            y, x0, test["H"], dt=args.dt, F=args.F,
            obs_std=args.obs_std, process_std=args.process_std,
            initial_std=args.initial_std,
        )
        # Gaussian marginal intervals.
        from jax.scipy.special import ndtri
        z = ndtri(jnp.asarray(0.5 + args.ci / 200.0))
        sd = jnp.sqrt(jnp.maximum(jnp.diagonal(result.covariance, axis1=-2, axis2=-1), 0))
        lo, hi = result.mean - z * sd, result.mean + z * sd
    else:
        result = bootstrap_particle_filter(
            y, x0, test["H"], jr.fold_in(key, 2), dt=args.dt, F=args.F,
            obs_std=args.obs_std, process_std=args.process_std,
            initial_std=args.initial_std, num_particles=args.num_particles,
            noise_type=noise, obs_df=args.obs_df,
            gmm_weights=args.gmm_weights, gmm_offsets=args.gmm_offsets,
            gmm_stds=args.gmm_stds,
        )
        lo = jnp.percentile(result.particles, (100 - args.ci) / 2, axis=-2)
        hi = jnp.percentile(result.particles, 100 - (100 - args.ci) / 2, axis=-2)

    mean, truth, lo, hi = map(np.asarray, (result.mean, truth, lo, hi))
    obs_idx = np.arange(0, args.N, args.obs_interval)
    unobs_idx = np.setdiff1d(np.arange(args.N), obs_idx)
    metrics = {
        "use_case": args.use_case, "baseline": method, "noise_type": noise,
        "n_eval": count,
        "rmse_all": float(np.mean(np.sqrt(np.mean((truth - mean) ** 2, axis=(1, 2))))),
        "rmse_observed": float(np.mean(np.sqrt(np.mean((truth[..., obs_idx] - mean[..., obs_idx]) ** 2, axis=(1, 2))))),
        "rmse_unobserved": float(np.mean(np.sqrt(np.mean((truth[..., unobs_idx] - mean[..., unobs_idx]) ** 2, axis=(1, 2))))) if len(unobs_idx) else np.nan,
        "mae_all": float(np.mean(np.abs(truth - mean))),
        "coverage": float(np.mean((truth >= lo) & (truth <= hi))),
    }

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics)
        writer.writeheader(); writer.writerow(metrics)
    np.savez_compressed(args.out + ".npz", mean=mean, lower=lo, upper=hi, truth=truth,
                        observations=np.asarray(y), **metrics)

    t = np.arange(1, args.horizon + 1) * args.dt
    n_win = min(args.n_windows, count)
    n_rows = (n_win + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), squeeze=False)
    display_method = "Extended Kalman Filter" if method == "ekf" else "Particle Filter"
    short_method = "EKF" if method == "ekf" else "PF"
    fig.suptitle(
        f"{display_method}  –  Lorenz-96 N={args.N}  –  {noise} noise  –  {args.ci:.0f}% CI\n"
        f"RMSE all={metrics['rmse_all']:.3f}  obs={metrics['rmse_observed']:.3f}  "
        f"unobs={metrics['rmse_unobserved']:.3f}  coverage={metrics['coverage']*100:.1f}%",
        fontsize=11,
    )
    n_show = min(4, args.N)
    colors_obs = plt.cm.Blues(np.linspace(.5, .9, len(obs_idx[:n_show])))
    colors_unobs = plt.cm.Oranges(np.linspace(.5, .9, len(unobs_idx[:n_show])))
    boundaries = np.arange(args.segment_length, args.horizon, args.segment_length) * args.dt
    for wi, ax in enumerate(axes.flat):
        if wi >= n_win:
            ax.set_visible(False); continue
        for ci, dim in enumerate(obs_idx[:n_show]):
            color = colors_obs[ci]
            ax.fill_between(t, lo[wi, :, dim], hi[wi, :, dim], color=color, alpha=.25)
            ax.plot(t, mean[wi, :, dim], color=color, lw=1.5, ls="--")
            ax.plot(t, truth[wi, :, dim], color=color, lw=1.5)
            oi = int(np.where(obs_idx == dim)[0][0])
            ax.scatter(t, np.asarray(y)[wi, :, oi], color=color, s=6, zorder=5)
        for ci, dim in enumerate(unobs_idx[:n_show]):
            color = colors_unobs[ci]
            ax.fill_between(t, lo[wi, :, dim], hi[wi, :, dim], color=color, alpha=.25)
            ax.plot(t, mean[wi, :, dim], color=color, lw=1.5, ls="--")
            ax.plot(t, truth[wi, :, dim], color=color, lw=1.5)
        for boundary in boundaries:
            ax.axvline(boundary, color="grey", lw=.8, ls=":", alpha=.7)
        ax.set_title(f"Window {wi}  (H={args.horizon}, {args.horizon // args.segment_length}×{args.segment_length} segments)", fontsize=9)
        ax.set_xlabel("Time"); ax.set_ylabel("State value")
    fig.legend(handles=[
        Line2D([0], [0], color="steelblue", label="Truth (observed dim)"),
        Line2D([0], [0], color="steelblue", ls="--", label=f"{short_method} mean (observed)"),
        Line2D([0], [0], color="steelblue", marker="o", ls="none", ms=5, label="Observation y"),
        Line2D([0], [0], color="darkorange", label="Truth (unobserved dim)"),
        Line2D([0], [0], color="darkorange", ls="--", label=f"{short_method} mean (unobserved)"),
        Patch(facecolor="grey", alpha=.3, label=f"{args.ci:.0f}% CI"),
        Line2D([0], [0], color="grey", ls=":", label="Segment boundary"),
    ], loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(args.out + "_windows.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    rmse_dim = np.sqrt(np.mean((truth - mean) ** 2, axis=(0, 1)))
    coverage_dim = np.mean((truth >= lo) & (truth <= hi), axis=(0, 1))
    posterior_std = np.sqrt(np.maximum(np.diagonal(np.asarray(result.covariance), axis1=-2, axis2=-1), 0))
    error = np.abs(truth - mean)
    bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(args.N)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{short_method} diagnostics  –  lorenz96_N{args.N}_seg{args.segment_length}_obs{args.obs_interval}_{noise}_std{args.obs_std}", fontsize=11)
    axes[0].bar(np.arange(args.N), rmse_dim, color=bar_colors)
    axes[0].axhline(args.obs_std, color="red", ls="--", lw=1)
    axes[0].set(xlabel="State dimension $i$", ylabel="RMSE", title="RMSE per dimension")
    axes[0].legend(handles=[Patch(color="steelblue", label="Observed"), Patch(color="darkorange", label="Unobserved"), Line2D([0], [0], color="red", ls="--", label=f"obs_std={args.obs_std}")], fontsize=8)
    axes[1].bar(np.arange(args.N), coverage_dim * 100, color=bar_colors)
    axes[1].axhline(args.ci, color="red", ls="--", lw=1, label=f"nominal {args.ci:.0f}%")
    axes[1].set_ylim(0, 105); axes[1].set(xlabel="State dimension $i$", ylabel="Coverage (%)", title=f"Empirical {args.ci:.0f}% CI coverage per dimension")
    axes[1].legend(fontsize=8)
    rng = np.random.default_rng(0)
    for label, indices, color in (("Observed", obs_idx, "steelblue"), ("Unobserved", unobs_idx, "darkorange")):
        std_flat, err_flat = posterior_std[..., indices].ravel(), error[..., indices].ravel()
        chosen = rng.choice(len(std_flat), min(2000, len(std_flat)), replace=False)
        axes[2].scatter(std_flat[chosen], err_flat[chosen], alpha=.15, s=6, color=color, label=label)
    limit = max(float(posterior_std.max()), float(error.max())) * 1.05
    axes[2].plot([0, limit], [0, limit], "k--", lw=1, label="y = x")
    axes[2].set_xlim(0, limit); axes[2].set_ylim(0, limit)
    axes[2].set(xlabel="Posterior std", ylabel="|Error|", title="Calibration: std vs |error|")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out + "_diagnostics.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(metrics)
    print(f"Saved {args.out}.csv, {args.out}.npz, {args.out}_windows.png, and {args.out}_diagnostics.png")


if __name__ == "__main__":
    main()
