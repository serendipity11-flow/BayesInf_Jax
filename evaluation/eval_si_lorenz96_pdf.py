#!/usr/bin/env python3
"""Evaluate and save Lorenz-96 SI figures in PDF with larger fonts.

This script generates three figures:
  1. Trajectories: first row only, truth / posterior mean / CI / observations
  2. Diagnostics: RMSE per dim, coverage per dim, calibration scatter
  3. Multimodal: particle histogram, marginal predictive, trajectory spread

Example:
  python3 evaluation/eval_si_lorenz96_pdf.py \
      --ckpt_path results/results_state_estimation/SI_N12_h101_obs2_gmm3c_std0.5_..._ms100.pt \
      --output_dir evaluation --font_scale 1.8 --n_plot_seqs 2
"""

import argparse
import os
import sys

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
)
from bayesinf.applications.state_estimation.stochastic_interpolant import StochasticInterpolantSE


class CalibratedStochasticInterpolantSE(StochasticInterpolantSE):
    obs_loss_weight: float = eqx.field(static=True)
    unobs_loss_weight: float = eqx.field(static=True)
    obs_source_noise_std: float = eqx.field(static=True)
    unobs_source_scale: float = eqx.field(static=True)
    obs_g0_scale: float = eqx.field(static=True)
    unobs_g0_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *args,
        obs_loss_weight: float = 1.0,
        unobs_loss_weight: float = 1.0,
        obs_source_noise_std: float = 0.0,
        unobs_source_scale: float = 1.0,
        obs_g0_scale: float = 1.0,
        unobs_g0_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obs_loss_weight = obs_loss_weight
        self.unobs_loss_weight = unobs_loss_weight
        self.obs_source_noise_std = obs_source_noise_std
        self.unobs_source_scale = unobs_source_scale
        self.obs_g0_scale = obs_g0_scale
        self.unobs_g0_scale = unobs_g0_scale


def _parse_tuple(s):
    return tuple(float(x) for x in s.split(","))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SI on Lorenz-96 and save PDF figures")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Checkpoint path to the trained SI model (.pt)")
    parser.add_argument("--output_dir", type=str, default="evaluation",
                        help="Directory where PDF files are saved")
    parser.add_argument("--font_scale", type=float, default=1.8,
                        help="Global font size scale for all figures")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of evaluation sequences to sample")
    parser.add_argument("--n_plot_seqs", type=int, default=2,
                        help="Number of trajectory sequences to show in the first row (max 2)")
    parser.add_argument("--n_particles", type=int, default=200,
                        help="Number of posterior particles")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Number of flow time steps for sampling")
    parser.add_argument("--ci", type=float, default=90.0,
                        help="Credible interval percent")
    parser.add_argument("--sampler", type=str, default="segmented",
                        choices=("sde", "ode", "segmented"))
    parser.add_argument("--seg_len_eval", type=int, default=None,
                        help="Segment length used during segmented sampling")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for data and posterior sampling")
    parser.add_argument("--out_tag", type=str, default="SI",
                        help="Output figure basename prefix")
    parser.add_argument("--file_ext", type=str, default="pdf", choices=("pdf",),
                        help="Output figure extension")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress messages")
    parser.add_argument("--obs_noise_type", type=str, default="gmm",
                        choices=("gaussian", "student_t", "gmm"))
    parser.add_argument("--obs_df", type=float, default=5.0,
                        help="Degrees of freedom for student_t observation noise")
    parser.add_argument("--gmm_weights", type=str, default="0.70,0.15,0.15")
    parser.add_argument("--gmm_offsets", type=str, default="0.0,2.5,-2.5")
    parser.add_argument("--gmm_stds", type=str, default="0.50,1.50,1.50")
    parser.add_argument("--obs_std", type=float, default=0.5)
    parser.add_argument("--obs_interval", type=int, default=2)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=101)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--F_lorenz", type=float, default=8.0)
    parser.add_argument("--spinup_steps", type=int, default=500)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--d_c", type=int, default=128)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time_emb_dim", type=int, default=32)
    parser.add_argument("--sigma_0", type=float, default=1.0)
    parser.add_argument("--g0", type=float, default=0.5)
    parser.add_argument("--score_weight", type=float, default=0.25)
    parser.add_argument("--lifter_loss_weight", type=float, default=0.1)
    parser.add_argument("--score_sigma_floor", type=float, default=0.05)
    parser.add_argument("--s_min", type=float, default=0.02)
    parser.add_argument("--interpolant_type", type=str, default="linear",
                        choices=("linear", "cosine"))
    parser.add_argument("--ms", action="store_true",
                        help="Indicate multiple-shooting training tag")
    parser.add_argument("--seg_len", type=int, default=100,
                        help="Segment length used during training")
    parser.add_argument("--obs_loss_weight", type=float, default=1.0)
    parser.add_argument("--unobs_loss_weight", type=float, default=1.0)
    parser.add_argument("--obs_source_noise_std", type=float, default=0.0)
    parser.add_argument("--unobs_source_scale", type=float, default=1.0)
    parser.add_argument("--obs_g0_scale", type=float, default=1.0)
    parser.add_argument("--unobs_g0_scale", type=float, default=1.0)
    return parser.parse_args()


def build_model(args):
    return CalibratedStochasticInterpolantSE(
        d_x=args.N,
        d_y=args.N // args.obs_interval,
        d_c=args.d_c,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        depth=args.depth,
        time_emb_dim=args.time_emb_dim,
        sigma_0=args.sigma_0,
        g0=args.g0,
        score_weight=args.score_weight,
        lifter_loss_weight=args.lifter_loss_weight,
        score_sigma_floor=args.score_sigma_floor,
        s_min=args.s_min,
        interpolant_type=args.interpolant_type,
        obs_loss_weight=args.obs_loss_weight,
        unobs_loss_weight=args.unobs_loss_weight,
        obs_source_noise_std=args.obs_source_noise_std,
        unobs_source_scale=args.unobs_source_scale,
        obs_g0_scale=args.obs_g0_scale,
        unobs_g0_scale=args.unobs_g0_scale,
        key=jr.PRNGKey(42),
    )


def configure_matplotlib(font_scale: float):
    plt.rcParams.update({
        "font.size": 12 * font_scale,
        "axes.titlesize": 14 * font_scale,
        "axes.labelsize": 14 * font_scale,
        "xtick.labelsize": 12 * font_scale,
        "ytick.labelsize": 12 * font_scale,
        "legend.fontsize": 14 * font_scale,
        "legend.title_fontsize": 15 * font_scale,
    })


def trajectory_font_sizes(font_scale: float):
    return {
        "label": 9 * font_scale,
        "title": 9 * font_scale,
        "tick": 9 * font_scale,
    }


def save_pdf(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seg_len_eval is None:
        args.seg_len_eval = args.seg_len

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    if args.verbose:
        print(f"Loading checkpoint from {args.ckpt_path}")

    model = build_model(args)

    def _load_array_leaf(f, x):
        is_jax_arr = isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
        is_np_arr = isinstance(x, np.ndarray)
        if is_jax_arr or is_np_arr:
            for _ in range(8):
                value = np.load(f, allow_pickle=True)
                if np.shape(value) == np.shape(x):
                    return jnp.asarray(value) if is_jax_arr else value
                if np.size(value) == 1:
                    continue
                return jnp.asarray(value) if is_jax_arr else value
            return jnp.asarray(value) if is_jax_arr else value

        if isinstance(x, (bool, int, float, complex, np.generic)):
            pos = f.tell()
            value = np.load(f, allow_pickle=True)
            if np.size(value) == 1:
                return type(x)(value.item())
            f.seek(pos)
            return x

        return x

    model = eqx.tree_deserialise_leaves(args.ckpt_path, model, filter_spec=_load_array_leaf)
    model = eqx.nn.inference_mode(model)

    if args.verbose:
        print("Preparing test data...")

    k_data = jr.PRNGKey(args.seed)
    gmm_weights = _parse_tuple(args.gmm_weights)
    gmm_offsets = _parse_tuple(args.gmm_offsets)
    gmm_stds = _parse_tuple(args.gmm_stds)

    norm_stats_path = args.ckpt_path.replace('.pt', '_norm_stats.npz')
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            f"Norm stats file not found: {norm_stats_path}\n"
            "Please provide a checkpoint path with a companion _norm_stats.npz file."
        )
    ns = np.load(norm_stats_path)
    x_mean = ns['x_mean']
    x_std = ns['x_std']
    y_mean = ns['y_mean']
    y_std = ns['y_std']

    data = generate_lorenz96_dataset(
        k_data,
        n_samples=max(args.n_eval, 2000),
        horizon=args.horizon,
        N=args.N,
        obs_interval=args.obs_interval,
        obs_std=args.obs_std,
        obs_noise_type=args.obs_noise_type,
        obs_df=args.obs_df if args.obs_noise_type == "student_t" else None,
        gmm_weights=gmm_weights if args.obs_noise_type == "gmm" else None,
        gmm_offsets=gmm_offsets if args.obs_noise_type == "gmm" else None,
        gmm_stds=gmm_stds if args.obs_noise_type == "gmm" else None,
        dt=args.dt,
        F=args.F_lorenz,
        spinup_steps=args.spinup_steps,
        d_u=1,
    )

    _, _, test_data = train_val_test_split(
        data, jr.fold_in(k_data, 1), train_frac=args.train_frac, val_frac=args.val_frac
    )

    x_test_raw = np.array(test_data["x"])
    y_test_raw = np.array(test_data["y"])
    x_test = jnp.array((x_test_raw - x_mean) / x_std)
    y_test = jnp.array((y_test_raw - y_mean) / y_std)

    n_eval = min(args.n_eval, x_test_raw.shape[0])
    n_plot = min(args.n_plot_seqs, n_eval, 2)
    T = args.horizon
    t_axis = np.arange(T) * args.dt
    obs_idx = np.arange(0, args.N, args.obs_interval)
    unobs_idx = np.array([i for i in range(args.N) if i not in obs_idx])

    if args.verbose:
        print(f"Sampling posteriors for {n_eval} sequences")

    @eqx.filter_jit
    def sample_batch(model, y_batch, key):
        B = y_batch.shape[0]
        keys = jr.split(key, B)
        if args.sampler == "segmented":
            fn = lambda yi, ki: model.sample_posterior_segmented(
                yi, ki, args.n_particles, args.n_steps, args.seg_len_eval
            )
        elif args.sampler == "ode":
            fn = lambda yi, ki: model.sample_posterior_ode(yi, ki, args.n_particles, args.n_steps)
        else:
            fn = lambda yi, ki: model.sample_posterior(yi, ki, args.n_particles, args.n_steps)
        return jax.vmap(fn)(y_batch, keys)

    chunk = 10
    all_samps = []
    k_eval = jr.PRNGKey(999)
    for start in range(0, n_eval, chunk):
        end = min(start + chunk, n_eval)
        y_batch = y_test[start:end]
        key_batch = jr.fold_in(k_eval, start)
        all_samps.append(np.array(sample_batch(model, y_batch, key_batch)))
    all_samps = np.concatenate(all_samps, axis=0)

    if args.verbose:
        print("Computing metrics and plotting...")

    post_mean = all_samps.mean(axis=1)
    lo = np.percentile(all_samps, (100 - args.ci) / 2, axis=1)
    hi = np.percentile(all_samps, 100 - (100 - args.ci) / 2, axis=1)
    x_true = x_test_raw[:n_eval]

    rmse_all = float(np.mean(np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(1, 2)))))
    rmse_obs = float(np.sqrt(np.mean((x_true[:, :, obs_idx] - post_mean[:, :, obs_idx]) ** 2)))
    rmse_unobs = float(np.sqrt(np.mean((x_true[:, :, unobs_idx] - post_mean[:, :, unobs_idx]) ** 2))) if len(unobs_idx) else float("nan")
    coverage = float(np.mean((x_true >= lo) & (x_true <= hi)))

    configure_matplotlib(args.font_scale)
    traj_fs = trajectory_font_sizes(args.font_scale)

    t_axis_eval = np.arange(T) * args.dt
    fig1, axes1 = plt.subplots(1, n_plot, figsize=(6.5 * n_plot, 4.5), squeeze=False)
    c_obs = plt.cm.Blues(np.linspace(0.45, 0.85, max(1, len(obs_idx[:4]))))
    c_unobs = plt.cm.Oranges(np.linspace(0.45, 0.85, max(1, len(unobs_idx[:4]))))

    for wi in range(n_plot):
        ax = axes1[0, wi]
        for ci_i, dim in enumerate(obs_idx[:4]):
            c = c_obs[ci_i]
            ax.fill_between(t_axis_eval, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.20)
            ax.plot(t_axis_eval, post_mean[wi, :, dim], color=c, lw=1.8, ls="--")
            ax.plot(t_axis_eval, x_true[wi, :, dim], color=c, lw=1.5)
            oi = int(np.where(obs_idx == dim)[0][0])
            ax.scatter(t_axis_eval, y_test_raw[wi, :T, oi], color=c, s=16, zorder=5, alpha=0.6)
        for ci_i, dim in enumerate(unobs_idx[:4]):
            c = c_unobs[ci_i]
            ax.fill_between(t_axis_eval, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.20)
            ax.plot(t_axis_eval, post_mean[wi, :, dim], color=c, lw=1.8, ls="--")
            ax.plot(t_axis_eval, x_true[wi, :, dim], color=c, lw=1.5)
        ax.set_xlim(t_axis_eval[0], t_axis_eval[-1])
        ax.margins(x=0)
        ax.set_title(f"Seq {wi}", fontsize=traj_fs["title"])
        ax.set_xlabel("Time", fontsize=traj_fs["label"])
        ax.set_ylabel("State", fontsize=traj_fs["label"])
        ax.tick_params(labelsize=traj_fs["tick"])

    legend_elems = [
        Line2D([0], [0], color="steelblue", lw=2.0, ls="-", label="Truth (obs dim)"),
        Line2D([0], [0], color="steelblue", lw=2.2, ls="--", label="Post. mean (obs dim)"),
        Line2D([0], [0], color="steelblue", marker="o", ls="", ms=6, label="Observation y"),
        Line2D([0], [0], color="darkorange", lw=2.0, ls="-", label="Truth (unobs dim)"),
        Line2D([0], [0], color="darkorange", lw=2.2, ls="--", label="Post. mean (unobs dim)"),
        Patch(facecolor="grey", alpha=0.3, label=f"{args.ci:.0f}% CI"),
    ]
    fig1.legend(handles=legend_elems, loc="lower center", ncol=3,
                fontsize=16 * args.font_scale, bbox_to_anchor=(0.5, -0.05))
    fig1.tight_layout(rect=[0, 0.05, 1, 1])

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(f"SI diagnostics — {args.out_tag}  n_eval={n_eval}  sampler={args.sampler}",
                  fontsize=14 * args.font_scale)

    ax = axes2[0]
    rmse_per_dim = np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(0, 1)))
    bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(args.N)]
    ax.bar(np.arange(args.N), rmse_per_dim, color=bar_colors)
    ax.axhline(args.obs_std, color="red", ls="--", lw=1.8)
    ax.set_xlabel("State dimension")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE per state dim")
    ax.legend(handles=[
        Patch(color="steelblue", label="Observed"),
        Patch(color="darkorange", label="Unobserved"),
        Line2D([0], [0], color="red", ls="--", label=f"obs_std={args.obs_std}"),
    ], fontsize=12 * args.font_scale)

    ax = axes2[1]
    ax.bar(np.arange(args.N), np.mean((x_true >= lo) & (x_true <= hi), axis=(0, 1)) * 100,
           color=bar_colors)
    ax.axhline(args.ci, color="red", ls="--", lw=1.8)
    ax.set_ylim(0, 105)
    ax.set_xlabel("State dimension")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Empirical {args.ci:.0f}% coverage")
    ax.legend(fontsize=12 * args.font_scale)

    ax = axes2[2]
    post_std = all_samps[:n_plot].std(axis=1)
    abs_err = np.abs(x_true[:n_plot] - post_mean[:n_plot])
    rng_np = np.random.default_rng(0)
    for label, idx, color in [("Observed", obs_idx, "steelblue"),
                              ("Unobserved", unobs_idx, "darkorange")]:
        if len(idx) == 0:
            continue
        s_f = post_std[:, :, idx].ravel()
        e_f = abs_err[:, :, idx].ravel()
        sel = rng_np.choice(len(s_f), size=min(5000, len(s_f)), replace=False)
        ax.scatter(s_f[sel], e_f[sel], alpha=0.20, s=18, color=color, label=label)
    lim = max(post_std.max(), abs_err.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.8, label="ideal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Posterior std")
    ax.set_ylabel("|Error|")
    ax.set_title("Calibration: posterior std vs |error|")
    ax.legend(fontsize=12 * args.font_scale)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    t_idx = int(round(0.5 * (T - 1)))
    t_val = t_idx * args.dt
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig3.suptitle(f"SI multi-modal analysis — {args.out_tag}", fontsize=14 * args.font_scale)

    ax = axes3[0]
    particles_t = all_samps[:, :, t_idx, :]
    true_t = x_true[:, t_idx, :]
    dim_pairs = [("steelblue", int(obs_idx[0]), f"dim {int(obs_idx[0])} (obs)")]
    if len(unobs_idx):
        dim_pairs.append(("darkorange", int(unobs_idx[0]), f"dim {int(unobs_idx[0])} (unobs)"))
    for color, dim, label in dim_pairs:
        p_flat = particles_t[:, :, dim].ravel()
        t_flat = true_t[:, dim].ravel()
        lo_p, hi_p = np.percentile(np.concatenate([p_flat, t_flat]), [0.5, 99.5])
        bins = np.linspace(lo_p, hi_p, 50)
        ax.hist(p_flat, bins=bins, density=True, alpha=0.5, color=color,
                label=f"particles – {label}")
        ax.hist(t_flat, bins=bins, density=True, alpha=0.0, edgecolor=color,
                histtype="step", lw=2, label=f"truth – {label}")
    ax.set_xlabel(f"State value at t={t_val:.2f}")
    ax.set_ylabel("Density")
    ax.set_title("Particle distribution at a fixed time")
    ax.legend(fontsize=12 * args.font_scale)

    ax = axes3[1]
    rng_pred = np.random.default_rng(42)
    state_at_obs = all_samps[:, :, :T, obs_idx]
    if args.obs_noise_type == "gmm":
        w_arr = np.array(gmm_weights)
        mu_arr = np.array(gmm_offsets) * args.obs_std
        s_arr = np.array(gmm_stds) * args.obs_std
        cumw = np.cumsum(w_arr[:-1])
        u_c = rng_pred.uniform(size=state_at_obs.shape)
        comp = (u_c[..., None] >= cumw).sum(axis=-1)
        eps = rng_pred.standard_normal(state_at_obs.shape)
        pred_noise = mu_arr[comp] + s_arr[comp] * eps
    elif args.obs_noise_type == "student_t" and args.obs_df < 1e6:
        a = args.obs_df / 2.0
        tau = rng_pred.gamma(a, 1.0 / a, size=state_at_obs.shape)
        pred_noise = args.obs_std * rng_pred.standard_normal(state_at_obs.shape) / np.sqrt(tau)
    else:
        pred_noise = args.obs_std * rng_pred.standard_normal(state_at_obs.shape)
    y_pred = (state_at_obs + pred_noise).ravel()
    y_data = y_test_raw[:n_eval, :T, :].ravel()
    lo_m, hi_m = np.percentile(np.concatenate([y_data, y_pred]), [0.5, 99.5])
    bins_m = np.linspace(lo_m, hi_m, 70)
    ax.hist(y_data, bins=bins_m, density=True, alpha=0.55, color="steelblue",
            label="Data $y_{\rm test}$")
    ax.hist(y_pred, bins=bins_m, density=True, alpha=0.55, color="darkorange",
            label="Posterior predictive $\hat{y}$")
    ax.set_xlabel("Observation value")
    ax.set_ylabel("Density")
    ax.set_title("Marginal data vs posterior predictive")
    ax.legend(fontsize=12 * args.font_scale)

    ax = axes3[2]
    seq_i = 0
    n_show_traj = min(30, args.n_particles)
    dim_show = int(obs_idx[0]) if len(obs_idx) else 0
    colors_traj = plt.cm.viridis(np.linspace(0.1, 0.9, n_show_traj))
    for pi in range(n_show_traj):
        ax.plot(t_axis_eval, all_samps[seq_i, pi, :, dim_show], color=colors_traj[pi], alpha=0.35, lw=0.8)
    ax.plot(t_axis_eval, x_true[seq_i, :, dim_show], "k-", lw=2.0, label="Truth")
    ax.plot(t_axis_eval, post_mean[seq_i, :, dim_show], "r--", lw=1.8, label="Post. mean")
    ax.set_xlim(t_axis_eval[0], t_axis_eval[-1])
    ax.margins(x=0)
    ax.set_xlabel("Time", fontsize=traj_fs["label"])
    ax.set_ylabel(f"State dim {dim_show}", fontsize=traj_fs["label"])
    ax.set_title(f"Particle trajectories (seq 0, dim {dim_show})", fontsize=traj_fs["title"])
    ax.tick_params(labelsize=traj_fs["tick"])
    ax.legend(fontsize=12 * args.font_scale)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    base = os.path.join(args.output_dir, f"{args.out_tag}_segev{args.seg_len_eval}")
    save_pdf(fig1, base + f"_trajectories.{args.file_ext}")
    save_pdf(fig2, base + f"_diagnostics.{args.file_ext}")
    save_pdf(fig3, base + f"_multimodal.{args.file_ext}")

    if args.verbose:
        print("Saved figures:")
        print("  ", base + f"_trajectories.{args.file_ext}")
        print("  ", base + f"_diagnostics.{args.file_ext}")
        print("  ", base + f"_multimodal.{args.file_ext}")


if __name__ == "__main__":
    main()
