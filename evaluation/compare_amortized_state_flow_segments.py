"""
Compare AmortizedStateFlow checkpoints across segment lengths and flow types.

The script loads trained AmortizedStateFlow models, samples posterior state
trajectories on a shared Lorenz-96 test set, and visualizes how metrics vary
with segment length for FM and RealNVP.

Example
-------
python3 evaluation/compare_amortized_state_flow_segments.py \\
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
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bayesinf.applications.state_estimation.amortized_state_flow import (  # noqa: E402
    AmortizedStateFlow,
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
    "sbm": "SBM",
}


def observed_status(dim, obs_interval):
    return "observed" if dim % obs_interval == 0 else "unobserved"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual comparison of AmortizedStateFlow segment lengths."
    )
    parser.add_argument("--flow_types", nargs="+", default=["fm", "realnvp"],
                        choices=("fm", "realnvp", "nsf", "maf", "sbm"))
    parser.add_argument("--seg_lens", nargs="+", type=int, default=[5, 10, 20, 25])

    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--obs_interval", type=int, default=2)
    parser.add_argument("--obs_std", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--F", type=float, default=8.0)
    parser.add_argument("--spinup_steps", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)

    parser.add_argument("--n_coupling", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--d_enc", type=int, default=64)
    parser.add_argument("--d_out", type=int, default=96)
    parser.add_argument("--encoder_type", type=str, default="transformer",
                        choices=("gru", "transformer"))
    parser.add_argument("--gru_hidden", type=int, default=32)
    parser.add_argument("--prior", type=str, default="iid",
                        choices=("wiener_scaled_t", "wiener_increment", "iid"))

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--ci", type=float, default=90.0)
    parser.add_argument("--fm_steps", type=int, default=100)
    parser.add_argument("--sbm_steps", type=int, default=1000)

    parser.add_argument("--plot_window", type=int, default=0)
    parser.add_argument("--plot_dims", nargs="+", type=int, default=[0, 1, 2, 3])
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
            "amortized_state_flow_segment_comparison",
        ),
    )
    return parser.parse_args()


def data_set_name(args, seg_len):
    seg_label = f"seg{seg_len}" if seg_len > 0 and args.horizon > seg_len else f"h{args.horizon}"
    data_set = f"lorenz96_N{args.N}_{seg_label}_obs{args.obs_interval}"
    if args.obs_std > 0:
        data_set += f"_noise_{args.obs_std}"
    return data_set


def checkpoint_path(args, flow_type, seg_len):
    return os.path.join(
        args.results_dir,
        f"AmortizedStateFlow_{flow_type}_{data_set_name(args, seg_len)}.pt",
    )


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
    d_u = 1
    data = generate_lorenz96_dataset(
        key,
        n_samples=args.n_samples,
        horizon=args.horizon,
        N=args.N,
        obs_interval=args.obs_interval,
        obs_std=args.obs_std,
        dt=args.dt,
        F=args.F,
        spinup_steps=args.spinup_steps,
        d_u=d_u,
    )
    _, _, test_data = train_val_test_split(
        data,
        jr.fold_in(key, 1),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    return test_data


def segment_inputs(test_data, n_eval, seg_len, horizon, state_dim):
    use_seg = seg_len > 0 and horizon > seg_len
    if not use_seg:
        return (
            test_data["x"][:n_eval],
            test_data["y"][:n_eval],
            test_data["u"][:n_eval],
            test_data["x0"][:n_eval],
            horizon,
            1,
        )

    n_segs = horizon // seg_len
    h_eff = n_segs * seg_len
    x = test_data["x"][:n_eval, :h_eff]
    y = test_data["y"][:n_eval, :h_eff]
    u = test_data["u"][:n_eval, :h_eff]
    x0 = test_data["x0"][:n_eval]

    if n_segs > 1:
        boundaries = x[:, seg_len - 1 : h_eff - 1 : seg_len, :]
        x0_segs = jnp.concatenate([x0[:, None, :], boundaries], axis=1)
    else:
        x0_segs = x0[:, None, :]

    y_seg = y.reshape(n_eval, n_segs, seg_len, -1).reshape(n_eval * n_segs, seg_len, -1)
    u_seg = u.reshape(n_eval, n_segs, seg_len, -1).reshape(n_eval * n_segs, seg_len, -1)
    x0_seg = x0_segs.reshape(n_eval * n_segs, state_dim)
    return x, y_seg, u_seg, x0_seg, h_eff, n_segs


def evaluate_checkpoint(args, test_data, flow_type, seg_len):
    ckpt = checkpoint_path(args, flow_type, seg_len)
    if not os.path.exists(ckpt):
        message = f"Missing checkpoint, skipping: {ckpt}"
        if args.strict:
            raise FileNotFoundError(message)
        print(f"[warn] {message}")
        return None

    d_y = args.N // args.obs_interval
    d_u = 1
    model = AmortizedStateFlow(
        d_x=args.N,
        d_y=d_y,
        d_u=d_u,
        d_enc=args.d_enc,
        d_out=args.d_out,
        key=jr.PRNGKey(42),
        flow_type=flow_type,
        n_coupling=args.n_coupling,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        encoder_type=args.encoder_type,
        gru_hidden=args.gru_hidden,
    )
    model = eqx.tree_deserialise_leaves(ckpt, model)
    model = eqx.nn.inference_mode(model)

    n_eval = min(args.n_eval, test_data["x"].shape[0])
    x_eval, y_in, u_in, x0_in, h_eff, n_segs = segment_inputs(
        test_data, n_eval, seg_len, args.horizon, args.N
    )

    print(
        f"Sampling {flow_type} seg{seg_len}: {args.num_samples} draws, "
        f"{n_eval} windows, effective horizon {h_eff}"
    )
    raw_samples = np.array(
        model.sample(
            y_in,
            u_in,
            x0_in,
            key=jr.fold_in(jr.PRNGKey(args.sample_seed), 1000 * seg_len + len(flow_type)),
            num_samples=args.num_samples,
            prior=args.prior,
            fm_steps=args.fm_steps,
            sbm_steps=args.sbm_steps,
        )
    )

    if n_segs > 1:
        samples = raw_samples.reshape(args.num_samples, n_eval, n_segs, seg_len, args.N)
        samples = samples.reshape(args.num_samples, n_eval, h_eff, args.N)
    else:
        samples = raw_samples

    x_eval_np = np.array(x_eval)
    mean = samples.mean(axis=0)
    lo_q = (100.0 - args.ci) / 2.0
    hi_q = 100.0 - lo_q
    lo = np.percentile(samples, lo_q, axis=0)
    hi = np.percentile(samples, hi_q, axis=0)

    obs_idx = np.arange(0, args.N, args.obs_interval)
    unobs_idx = np.array([i for i in range(args.N) if i not in obs_idx])
    err = x_eval_np - mean
    rmse_all = float(np.sqrt(np.mean(err**2)))
    rmse_obs = float(np.sqrt(np.mean(err[:, :, obs_idx] ** 2)))
    rmse_unobs = float(np.sqrt(np.mean(err[:, :, unobs_idx] ** 2)))
    coverage = float(np.mean((x_eval_np >= lo) & (x_eval_np <= hi)))
    mean_ci_width = float(np.mean(hi - lo))
    post_std = samples.std(axis=0)
    calibration_kl = calibration_kl_divergence(x_eval_np, mean, post_std)

    return RunResult(
        flow_type=flow_type,
        seg_len=seg_len,
        checkpoint=ckpt,
        rmse_all=rmse_all,
        rmse_obs=rmse_obs,
        rmse_unobs=rmse_unobs,
        coverage=coverage,
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
              "maf": "tab:red", "sbm": "tab:purple"}
    markers = {"fm": "o", "realnvp": "s", "nsf": "^", "maf": "D", "sbm": "P"}
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
    for ext in ("png", "pdf"):
        path = args.out_prefix + f"_metrics.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")


def plot_trajectory_grid(args, test_data, results):
    dims = [d for d in args.plot_dims if 0 <= d < args.N]
    if not dims:
        dims = [0]
    dims = dims[:4]

    colors = {"fm": "tab:blue", "realnvp": "tab:orange", "nsf": "tab:green",
              "maf": "tab:red", "sbm": "tab:purple"}
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
            ax.grid(alpha=0.18)

    legend_handles.extend([
        Patch(facecolor="0.5", alpha=0.14, label=f"{args.ci:.0f}% CI"),
        Line2D([0], [0], color="0.7", lw=0.8, ls=":", label="Segment boundary"),
    ])
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=18)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("png", "pdf"):
        path = args.out_prefix + f"_trajectories.{ext}"
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

    csv_path = args.out_prefix + "_metrics.csv"
    save_metrics_csv(csv_path, results)
    print(f"Saved {csv_path}")
    plot_metrics(args, results)
    plot_trajectory_grid(args, test_data, results)

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
