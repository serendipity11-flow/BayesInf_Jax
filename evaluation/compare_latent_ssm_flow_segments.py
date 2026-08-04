"""
Compare LatentSSMFlow checkpoints across segment lengths and flow types.

Produces the same two visualization styles as the AmortizedStateFlow segment
comparison:
  1. Metrics grid over segment length.
  2. Trajectory grid with FM and RealNVP overlaid for each segment length.

Example
-------
python3 evaluation/compare_latent_ssm_flow_segments.py \\
    --flow_types fm realnvp --seg_lens 5 10 20 25 \\
    --n_eval 50 --num_samples 50
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bayesinf.applications.state_estimation.latent_ssm_flow import (  # noqa: E402
    build_latent_ssm_flow,
)
from experiments.lorenz96_state_estimation.utils.dataset import (  # noqa: E402
    generate_lorenz96_dataset,
    train_val_test_split,
)


@dataclass
class RunResult:
    flow_type: str
    seg_len: int
    checkpoint: str
    rmse_all: float
    rmse_obs: float
    rmse_unobs: float
    coverage: float
    coverage_per_dim: np.ndarray
    mean_ci_width: float
    calibration_kl: float
    samples: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


FLOW_LABELS = {
    "fm": "FM",
    "realnvp": "RealNVP",
    "nsf": "NSF",
    "maf": "MAF",
}


def observed_status(dim, obs_interval):
    return "observed" if dim % obs_interval == 0 else "unobserved"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual comparison of LatentSSMFlow segment lengths."
    )
    parser.add_argument("--flow_types", nargs="+", default=["fm", "realnvp"],
                        choices=("fm", "realnvp", "nsf", "maf"))
    parser.add_argument("--seg_lens", nargs="+", type=int, default=[5, 10, 20, 25])

    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--obs_interval", type=int, default=2)
    parser.add_argument("--obs_std", type=float, default=0.5)
    parser.add_argument("--obs_df", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--F_lorenz", type=float, default=8.0)
    parser.add_argument("--spinup_steps", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)

    parser.add_argument("--d_enc", type=int, default=64)
    parser.add_argument("--d_out", type=int, default=96)
    parser.add_argument("--encoder_type", type=str, default="gru",
                        choices=("gru", "transformer", "pointwise"))
    parser.add_argument("--local_window", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--local_window_prev_only", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_coupling", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--max_rho", type=float, default=0.80)
    parser.add_argument("--fix_ssm", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--lambda_ent", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=999)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--ci", type=float, default=90.0)
    parser.add_argument("--full_horizon_rollout", action="store_true",
                        help="Sample one full 100-step posterior rollout instead of restarting at segment boundaries.")

    parser.add_argument("--plot_window", type=int, default=0)
    parser.add_argument("--plot_dims", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--plots", nargs="+",
                        choices=("metrics", "trajectories", "uncertainty"),
                        default=["metrics", "trajectories", "uncertainty"],
                        help="Which plot groups to generate.")
    parser.add_argument("--plot_formats", nargs="+", choices=("png", "pdf"),
                        default=["png", "pdf"],
                        help="Output formats for generated plots.")
    parser.add_argument("--skip_metrics_csv", action="store_true",
                        help="Do not write the metrics CSV file.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any requested checkpoint is missing.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "results_state_estimation"),
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            "evaluation",
            "latent_ssm_flow_segment_comparison",
        ),
    )
    return parser.parse_args()


def data_tag(args, seg_len):
    seg_label = f"seg{seg_len}" if seg_len > 0 and args.horizon > seg_len else f"h{args.horizon}"
    return (
        f"lorenz96_N{args.N}_{seg_label}_obs{args.obs_interval}"
        f"_studt{args.obs_df}_std{args.obs_std}"
    )


def model_tag(args, flow_type, seg_len):
    ssm_tag = "_fixedSSM" if args.fix_ssm else ""
    ent_tag = f"_ent{args.lambda_ent}" if args.lambda_ent > 0 else ""
    enc_tag = f"_enc{args.encoder_type}" if args.encoder_type != "transformer" else ""
    lw_tag = (
        "_lwp"
        if args.local_window and args.local_window_prev_only
        else ("_lw" if args.local_window else "")
    )
    return f"LatentSSMFlow_{flow_type}{ssm_tag}{enc_tag}{lw_tag}{ent_tag}_{data_tag(args, seg_len)}"


def checkpoint_path(args, flow_type, seg_len):
    return os.path.join(args.results_dir, f"{model_tag(args, flow_type, seg_len)}.pt")


def calibration_kl_divergence(x_true, post_mean, post_std, bins=80, clip=5.0):
    """KL(empirical standardized truth error || N(0, 1)); lower is better."""
    z = ((x_true - post_mean) / (post_std + 1e-6)).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")

    counts, edges = np.histogram(z, bins=bins, range=(-clip, clip))
    p = counts.astype(float) + 1e-12
    p /= p.sum()

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    q = np.exp(-0.5 * centers**2) / np.sqrt(2.0 * np.pi) * widths
    q += 1e-12
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def load_test_data(args):
    key = jr.PRNGKey(args.seed)
    data = generate_lorenz96_dataset(
        key,
        n_samples=args.n_samples,
        horizon=args.horizon,
        N=args.N,
        obs_interval=args.obs_interval,
        obs_std=args.obs_std,
        obs_df=args.obs_df,
        dt=args.dt,
        F=args.F_lorenz,
        spinup_steps=args.spinup_steps,
        d_u=1,
    )
    _, _, test_data = train_val_test_split(
        data,
        jr.fold_in(key, 1),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    return test_data


def build_model(args, flow_type, seg_len):
    return build_latent_ssm_flow(
        d_x=args.N,
        d_y=args.N // args.obs_interval,
        d_u=1,
        d_enc=args.d_enc,
        d_out=args.d_out,
        encoder_type=args.encoder_type,
        local_window=args.local_window,
        local_window_prev_only=args.local_window_prev_only,
        gru_hidden=args.gru_hidden,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        flow_type=flow_type,
        seg_len=seg_len,
        n_coupling=args.n_coupling,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        num_bins=args.num_bins,
        max_rho=args.max_rho,
        key=jr.PRNGKey(42),
    )


def segment_eval_arrays(x, y, u, x0, seg_len):
    if seg_len <= 0 or x.shape[1] <= seg_len:
        return x, y, u, x0, 1, x.shape[1]

    M, H, D_x = x.shape
    D_y = y.shape[-1]
    D_u = u.shape[-1]
    n_segs = H // seg_len
    H_eff = n_segs * seg_len

    x_eff = x[:, :H_eff]
    y_eff = y[:, :H_eff]
    u_eff = u[:, :H_eff]
    if n_segs > 1:
        boundaries = x_eff[:, seg_len - 1 : H_eff - 1 : seg_len, :]
        x0_segs = np.concatenate([x0[:, None, :], boundaries], axis=1)
    else:
        x0_segs = x0[:, None, :]

    x_seg = x_eff.reshape(M, n_segs, seg_len, D_x).reshape(M * n_segs, seg_len, D_x)
    y_seg = y_eff.reshape(M, n_segs, seg_len, D_y).reshape(M * n_segs, seg_len, D_y)
    u_seg = u_eff.reshape(M, n_segs, seg_len, D_u).reshape(M * n_segs, seg_len, D_u)
    x0_seg = x0_segs.reshape(M * n_segs, D_x)
    return x_seg, y_seg, u_seg, x0_seg, n_segs, H_eff


def unsegment_samples(samples_seg, n_eval, n_segs, seg_len):
    if n_segs == 1:
        return samples_seg
    _, n_samples, _, D_x = samples_seg.shape
    return (
        samples_seg.reshape(n_eval, n_segs, n_samples, seg_len, D_x)
        .transpose(0, 2, 1, 3, 4)
        .reshape(n_eval, n_samples, n_segs * seg_len, D_x)
    )


def run_posterior_batch(model, y, u, x0, key, n_samples, full_horizon_rollout=False):
    keys = jr.split(key, y.shape[0])

    if full_horizon_rollout:
        def one_seq(yi, ui, x0i, ki):
            return model.sample_posterior(
                yi, ui, x0i, ki, n_samples, restart_segments=False,
            )
    else:
        def one_seq(yi, ui, x0i, ki):
            return model.sample_posterior(yi, ui, x0i, ki, n_samples)

    return jax.vmap(one_seq)(y, u, x0, keys)


def evaluate_checkpoint(args, test_data, flow_type, seg_len):
    ckpt = checkpoint_path(args, flow_type, seg_len)
    if not os.path.exists(ckpt):
        message = f"Missing checkpoint, skipping: {ckpt}"
        if args.strict:
            raise FileNotFoundError(message)
        print(f"[warn] {message}")
        return None

    model = build_model(args, flow_type, seg_len)
    model = eqx.tree_deserialise_leaves(ckpt, model)
    model = eqx.nn.inference_mode(model)

    n_eval = min(args.n_eval, test_data["x"].shape[0])
    print(
        f"Sampling {flow_type} seg{seg_len}: {args.num_samples} draws, "
        f"{n_eval} sequences"
    )
    if args.full_horizon_rollout:
        x_seg = np.array(test_data["x"][:n_eval])
        y_seg = np.array(test_data["y"][:n_eval])
        u_seg = np.array(test_data["u"][:n_eval])
        x0_seg = np.array(test_data["x0"][:n_eval])
        n_segs = 1
        h_eff = args.horizon
    else:
        x_seg, y_seg, u_seg, x0_seg, n_segs, h_eff = segment_eval_arrays(
            np.array(test_data["x"][:n_eval]),
            np.array(test_data["y"][:n_eval]),
            np.array(test_data["u"][:n_eval]),
            np.array(test_data["x0"][:n_eval]),
            seg_len,
        )
    samples_mst = np.array(
        run_posterior_batch(
            model,
            jnp.array(y_seg),
            jnp.array(u_seg),
            jnp.array(x0_seg),
            key=jr.fold_in(jr.PRNGKey(args.sample_seed), 1000 * seg_len + len(flow_type)),
            n_samples=args.num_samples,
            full_horizon_rollout=args.full_horizon_rollout,
        )
    )
    if not args.full_horizon_rollout:
        samples_mst = unsegment_samples(samples_mst, n_eval, n_segs, seg_len)
    samples = np.transpose(samples_mst, (1, 0, 2, 3))  # (S, M, T_eff, N)

    mean = samples.mean(axis=0)
    lo_q = (100.0 - args.ci) / 2.0
    hi_q = 100.0 - lo_q
    lo = np.percentile(samples, lo_q, axis=0)
    hi = np.percentile(samples, hi_q, axis=0)

    x_true = np.array(test_data["x"][:n_eval, :h_eff])
    obs_idx = np.arange(0, args.N, args.obs_interval)
    unobs_idx = np.array([i for i in range(args.N) if i not in obs_idx])

    err = x_true - mean
    rmse_all = float(np.sqrt(np.mean(err**2)))
    rmse_obs = float(np.sqrt(np.mean(err[:, :, obs_idx] ** 2)))
    rmse_unobs = float(np.sqrt(np.mean(err[:, :, unobs_idx] ** 2)))
    coverage = float(np.mean((x_true >= lo) & (x_true <= hi)))
    coverage_per_dim = np.mean((x_true >= lo) & (x_true <= hi), axis=(0, 1))
    mean_ci_width = float(np.mean(hi - lo))
    calibration_kl = calibration_kl_divergence(x_true, mean, samples.std(axis=0))

    return RunResult(
        flow_type=flow_type,
        seg_len=seg_len,
        checkpoint=ckpt,
        rmse_all=rmse_all,
        rmse_obs=rmse_obs,
        rmse_unobs=rmse_unobs,
        coverage=coverage,
        coverage_per_dim=coverage_per_dim,
        mean_ci_width=mean_ci_width,
        calibration_kl=calibration_kl,
        samples=samples,
        mean=mean,
        lo=lo,
        hi=hi,
    )


def save_metrics_csv(path, results):
    fieldnames = [
        "flow_type",
        "seg_len",
        "rmse_all",
        "rmse_obs",
        "rmse_unobs",
        "coverage",
        "mean_ci_width",
        "calibration_kl",
        "checkpoint",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({name: getattr(r, name) for name in fieldnames})


def plot_metrics(args, results):
    colors = {"fm": "tab:blue", "realnvp": "tab:orange", "nsf": "tab:green",
              "maf": "tab:red"}
    markers = {"fm": "o", "realnvp": "s", "nsf": "^", "maf": "D"}
    metrics = [
        ("rmse_all", "RMSE"),
        ("rmse_obs", "Observed RMSE"),
        ("rmse_unobs", "Unobserved RMSE"),
        ("coverage", f"{args.ci:.0f}% CI coverage"),
        ("mean_ci_width", "Mean CI width"),
        ("calibration_kl", "Calibration KL"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))
    axes_flat = axes.ravel()
    handles = []
    labels = []
    for ax, (attr, title) in zip(axes_flat, metrics):
        for flow_type in args.flow_types:
            rows = sorted([r for r in results if r.flow_type == flow_type], key=lambda r: r.seg_len)
            if not rows:
                continue
            xs = [r.seg_len for r in rows]
            ys = [getattr(r, attr) for r in rows]
            line, = ax.plot(
                xs,
                ys,
                marker=markers.get(flow_type, "o"),
                lw=2.4,
                ms=7,
                color=colors.get(flow_type),
                label=FLOW_LABELS.get(flow_type, flow_type.upper()),
            )
            if attr == metrics[0][0]:
                handles.append(line)
                labels.append(FLOW_LABELS.get(flow_type, flow_type.upper()))
        if attr == "coverage":
            ax.axhline(args.ci / 100.0, color="0.35", ls="--", lw=1)
            ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Segment length", fontsize=18)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=16)
    axes_flat[0].set_ylabel("Metric value", fontsize=18)
    axes_flat[3].set_ylabel("Metric value", fontsize=18)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=18)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in args.plot_formats:
        path = args.out_prefix + f"_metrics.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")


def plot_trajectory_grid(args, test_data, results):
    dims = [d for d in args.plot_dims if 0 <= d < args.N]
    if not dims:
        dims = [0]
    dims = dims[:4]

    colors = {"fm": "tab:blue", "realnvp": "tab:orange", "nsf": "tab:green",
              "maf": "tab:red"}
    by_seg = {
        seg_len: sorted(
            [r for r in results if r.seg_len == seg_len],
            key=lambda r: args.flow_types.index(r.flow_type)
            if r.flow_type in args.flow_types else len(args.flow_types),
        )
        for seg_len in args.seg_lens
    }
    seg_rows = [seg_len for seg_len in args.seg_lens if by_seg.get(seg_len)]
    n_rows = len(seg_rows)
    n_cols = len(dims)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16.5, 12.0),
        sharex=True,
        squeeze=False,
    )

    win = min(args.plot_window, min(r.mean.shape[0] for r in results) - 1)
    legend_handles = [Line2D([0], [0], color="black", lw=1.6, label="Truth")]
    seen_flows = set()
    for row_i, seg_len in enumerate(seg_rows):
        row_results = by_seg[seg_len]
        h_eff = min(r.mean.shape[1] for r in row_results)
        t = np.arange(1, h_eff + 1) * args.dt
        x_true = np.array(test_data["x"][win, :h_eff])
        for col_i, dim in enumerate(dims):
            ax = axes[row_i, col_i]
            ax.plot(t, x_true[:, dim], color="black", lw=1.6)
            for r in row_results:
                color = colors.get(r.flow_type, "tab:gray")
                label = FLOW_LABELS.get(r.flow_type, r.flow_type.upper())
                ax.fill_between(
                    t,
                    r.lo[win, :h_eff, dim],
                    r.hi[win, :h_eff, dim],
                    color=color,
                    alpha=0.14,
                )
                ax.plot(t, r.mean[win, :h_eff, dim], color=color, lw=1.8, ls="--")
                if r.flow_type not in seen_flows:
                    legend_handles.append(
                        Line2D([0], [0], color=color, lw=1.8, ls="--", label=label)
                    )
                    seen_flows.add(r.flow_type)
            if seg_len > 0 and h_eff > seg_len:
                for boundary in np.arange(seg_len, h_eff, seg_len) * args.dt:
                    ax.axvline(boundary, color="0.7", lw=0.8, ls=":")
            if row_i == 0:
                ax.set_title(
                    f"x[{dim}] ({observed_status(dim, args.obs_interval)})",
                    fontsize=20,
                )
            if col_i == 0:
                ax.set_ylabel(f"seg={seg_len}", fontsize=18)
            if row_i == n_rows - 1:
                ax.set_xlabel("Time", fontsize=18)
            ax.tick_params(axis="both", labelsize=16)

    legend_handles.extend([
        Patch(facecolor="0.5", alpha=0.14, label=f"{args.ci:.0f}% CI"),
        Line2D([0], [0], color="0.7", lw=0.8, ls=":", label="Segment boundary"),
    ])
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=18)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in args.plot_formats:
        path = args.out_prefix + f"_trajectories.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")


def plot_uncertainty_distribution(args, results):
    colors = {"fm": "tab:blue", "realnvp": "tab:orange", "nsf": "tab:green",
              "maf": "tab:red"}
    by_seg = {
        seg_len: sorted(
            [r for r in results if r.seg_len == seg_len],
            key=lambda r: args.flow_types.index(r.flow_type)
            if r.flow_type in args.flow_types else len(args.flow_types),
        )
        for seg_len in args.seg_lens
    }
    seg_rows = [seg_len for seg_len in args.seg_lens if by_seg.get(seg_len)]
    if not seg_rows:
        return

    obs_idx = np.arange(0, args.N, args.obs_interval)
    x_pdf = np.linspace(-4.0, 4.0, 500)
    if args.obs_df < 1e6:
        noise_pdf = stats.t.pdf(x_pdf, df=args.obs_df)
        noise_label = f"obs noise Student-t(df={args.obs_df})"
        noise_std = (
            args.obs_std * np.sqrt(args.obs_df / (args.obs_df - 2.0))
            if args.obs_df > 2.0
            else np.inf
        )
    else:
        noise_pdf = stats.norm.pdf(x_pdf)
        noise_label = "obs noise Gaussian"
        noise_std = args.obs_std

    fig = plt.figure(figsize=(12.0, 8.8 * len(seg_rows)), constrained_layout=False)
    gs = fig.add_gridspec(
        2 * len(seg_rows),
        2,
        height_ratios=[ratio for _ in seg_rows for ratio in (1.0, 1.35)],
        hspace=0.50,
        wspace=0.25,
    )

    rng = np.random.default_rng(0)
    for row_i, seg_len in enumerate(seg_rows):
        grid_row = 2 * row_i
        ax_dist = fig.add_subplot(gs[grid_row, 0])
        ax_scale = fig.add_subplot(gs[grid_row, 1])
        ax_cov = fig.add_subplot(gs[grid_row + 1, :])

        ax_dist.plot(x_pdf, noise_pdf, color="black", lw=2.4, label=noise_label)
        for r in by_seg[seg_len]:
            color = colors.get(r.flow_type, "tab:gray")
            label = FLOW_LABELS.get(r.flow_type, r.flow_type.upper())
            centered = r.samples[:, :, :, obs_idx] - r.mean[None, :, :, obs_idx]
            z_unc = (centered / args.obs_std).ravel()
            z_unc = z_unc[np.isfinite(z_unc)]
            if z_unc.size > 200_000:
                z_unc = rng.choice(z_unc, size=200_000, replace=False)
            z_unc = z_unc[np.abs(z_unc) <= 4.0]
            ax_dist.hist(
                z_unc,
                bins=np.linspace(-4.0, 4.0, 80),
                density=True,
                histtype="step",
                lw=2.4,
                color=color,
                label=f"{label} posterior samples",
            )

            post_std = r.samples[:, :, :, obs_idx].std(axis=0).ravel()
            post_std = post_std[np.isfinite(post_std)]
            ax_scale.hist(
                post_std,
                bins=60,
                density=True,
                histtype="stepfilled",
                alpha=0.20,
                color=color,
                label=f"{label} posterior std",
            )
            ax_scale.hist(
                post_std,
                bins=60,
                density=True,
                histtype="step",
                lw=2.2,
                color=color,
            )

        ax_dist.set_title("Centered posterior uncertainty", fontsize=20)
        ax_dist.set_xlabel(r"$(x^{(s)} - \bar{x}) / \sigma_{\rm obs}$", fontsize=18)
        ax_dist.set_ylabel("Density", fontsize=18)
        ax_dist.grid(alpha=0.22)
        ax_dist.tick_params(axis="both", labelsize=16)
        ax_dist.legend(fontsize=15)

        x_scale_max = ax_scale.get_xlim()[1]
        x_noise = np.linspace(0.0, x_scale_max, 500)
        if args.obs_df < 1e6:
            abs_noise_pdf = 2.0 * stats.t.pdf(x_noise / args.obs_std, df=args.obs_df) / args.obs_std
        else:
            abs_noise_pdf = 2.0 * stats.norm.pdf(x_noise / args.obs_std) / args.obs_std
        ax_scale.plot(
            x_noise,
            abs_noise_pdf,
            color="black",
            lw=2.4,
            label=f"|{noise_label}|",
        )
        ax_scale.set_xlim(left=0.0, right=x_scale_max)
        ax_scale.axvline(
            args.obs_std,
            color="black",
            ls="--",
            lw=2.0,
            label=rf"obs scale $\sigma$={args.obs_std}",
        )
        if np.isfinite(noise_std):
            ax_scale.axvline(
                noise_std,
                color="0.35",
                ls=":",
                lw=2.2,
                label=f"obs marginal std={noise_std:.3f}",
            )
        ax_scale.set_title("Posterior std distribution", fontsize=20)
        ax_scale.set_xlabel("Posterior std in state units", fontsize=18)
        ax_scale.set_ylabel("Density", fontsize=18)
        ax_scale.grid(alpha=0.22)
        ax_scale.tick_params(axis="both", labelsize=16)
        ax_scale.legend(fontsize=15)

        row_results = by_seg[seg_len]
        dims = np.arange(args.N)
        bar_colors = [
            "steelblue" if dim % args.obs_interval == 0 else "darkorange"
            for dim in dims
        ]
        n_flows = len(row_results)
        width = min(0.72 / max(n_flows, 1), 0.55)
        offsets = (np.arange(n_flows) - (n_flows - 1) / 2.0) * width
        for flow_i, r in enumerate(row_results):
            label = FLOW_LABELS.get(r.flow_type, r.flow_type.upper())
            bars = ax_cov.bar(
                dims + offsets[flow_i],
                r.coverage_per_dim * 100.0,
                color=bar_colors,
                alpha=0.82,
                width=width,
                edgecolor=colors.get(r.flow_type, "0.25") if n_flows > 1 else None,
                linewidth=1.2 if n_flows > 1 else 0.0,
                label=label if n_flows > 1 else None,
            )
            if n_flows == 1:
                for bar, value in zip(bars, r.coverage_per_dim):
                    ax_cov.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(100.0 * value + 2.0, 102.0),
                        f"{100.0 * value:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                    )
        ax_cov.axhline(
            args.ci,
            color="red",
            ls="--",
            lw=1.8,
        )
        ax_cov.set_title("CI coverage per state", fontsize=22)
        ax_cov.set_xticks(dims)
        ax_cov.set_xlabel("State dimension", fontsize=20)
        ax_cov.set_ylim(0.0, 105.0)
        ax_cov.set_ylabel("Coverage (%)", fontsize=20)
        ax_cov.grid(axis="y", alpha=0.22)
        ax_cov.tick_params(axis="both", labelsize=18)
        legend_handles = [
            Patch(color="steelblue", label="Observed"),
            Patch(color="darkorange", label="Unobserved"),
        ]
        if n_flows > 1:
            legend_handles.extend(
                Line2D([0], [0], color=colors.get(r.flow_type, "0.25"), lw=2,
                       label=FLOW_LABELS.get(r.flow_type, r.flow_type.upper()))
                for r in row_results
            )
        ax_cov.legend(
            handles=legend_handles,
            fontsize=16,
            loc="upper right",
            ncol=2,
        )

    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.085)
    for ext in args.plot_formats:
        path = args.out_prefix + f"_uncertainty_distribution.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    test_data = load_test_data(args)
    results = []
    for flow_type in args.flow_types:
        for seg_len in args.seg_lens:
            result = evaluate_checkpoint(args, test_data, flow_type, seg_len)
            if result is not None:
                results.append(result)

    if not results:
        raise RuntimeError("No checkpoints were evaluated. Check --results_dir, --flow_types, and --seg_lens.")

    if not args.skip_metrics_csv:
        csv_path = args.out_prefix + "_metrics.csv"
        save_metrics_csv(csv_path, results)
        print(f"Saved {csv_path}")
    if "metrics" in args.plots:
        plot_metrics(args, results)
    if "trajectories" in args.plots:
        plot_trajectory_grid(args, test_data, results)
    if "uncertainty" in args.plots:
        plot_uncertainty_distribution(args, results)

    print("\nSummary")
    for r in sorted(results, key=lambda item: (item.flow_type, item.seg_len)):
        print(
            f"{r.flow_type:7s} seg={r.seg_len:>3d} "
            f"rmse={r.rmse_all:.4f} obs={r.rmse_obs:.4f} "
            f"unobs={r.rmse_unobs:.4f} coverage={100*r.coverage:.1f}% "
            f"cal_kl={r.calibration_kl:.4f}"
        )


if __name__ == "__main__":
    main()
