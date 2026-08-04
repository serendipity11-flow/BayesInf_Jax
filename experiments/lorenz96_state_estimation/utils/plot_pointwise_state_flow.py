"""
Evaluate and plot PointwiseStateFlow results.

Produces two figures per checkpoint:
  Figure 1 — per-window trajectory panels (truth vs posterior mean ± CI)
  Figure 2 — 3-panel diagnostics: RMSE/dim, coverage/dim, calibration scatter

Additionally, if multiple checkpoints (different seg_len values) exist for the
same flow_type + encoder_type, produces a Figure 3 comparing RMSE vs window size.

Usage:
    python3 -m experiments.lorenz96_state_estimation.utils.plot_pointwise_state_flow \\
        --flow_type realnvp --encoder_type transformer --seg_len 5 --num_samples 50
"""
import sys, os, argparse
import numpy as np
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.pointwise_state_flow import PointwiseStateFlow
from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset, train_val_test_split,
)

# ---------------------------------------------------------------------------
# Arguments  (mirror training script for checkpoint path resolution)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--flow_type",    type=str,   default="realnvp",
                    choices=("realnvp", "nsf", "maf", "fm"))
parser.add_argument("--horizon",      type=int,   default=100)
parser.add_argument("--N",            type=int,   default=12)
parser.add_argument("--obs_interval", type=int,   default=2)
parser.add_argument("--obs_std",      type=float, default=0.5)
parser.add_argument("--dt",           type=float, default=0.02)
parser.add_argument("--F",            type=float, default=8.0)
parser.add_argument("--spinup_steps", type=int,   default=500)
parser.add_argument("--n_samples",    type=int,   default=2000)
parser.add_argument("--train_frac",   type=float, default=0.8)
parser.add_argument("--val_frac",     type=float, default=0.1)
# Encoder / flow architecture (must match training)
parser.add_argument("--encoder_type", type=str,   default="transformer",
                    choices=("gru", "transformer"))
parser.add_argument("--seg_len",      type=int,   default=5)
parser.add_argument("--d_enc",        type=int,   default=64)
parser.add_argument("--d_out",        type=int,   default=96)
parser.add_argument("--gru_hidden",   type=int,   default=64)
parser.add_argument("--d_model",      type=int,   default=64)
parser.add_argument("--n_heads",      type=int,   default=4)
parser.add_argument("--d_ff",         type=int,   default=256)
parser.add_argument("--n_layers",     type=int,   default=3)
parser.add_argument("--n_coupling",   type=int,   default=8)
parser.add_argument("--hidden_size",  type=int,   default=128)
parser.add_argument("--n_hidden",     type=int,   default=2)
parser.add_argument("--num_bins",     type=int,   default=8)
parser.add_argument("--seed",         type=int,   default=0)
# Plot-specific
parser.add_argument("--num_samples",  type=int,   default=50)
parser.add_argument("--n_windows",    type=int,   default=6)
parser.add_argument("--ci",           type=float, default=90.0)
parser.add_argument("--out",          type=str,   default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
results_dir = "/home/zhangyi/bayesinf_jax/results/results_state_estimation"

def _data_set_label(seg_len):
    _seg = f"seg{seg_len}" if seg_len > 0 and args.horizon > seg_len else f"h{args.horizon}"
    ds = f"lorenz96_N{args.N}_{_seg}_obs{args.obs_interval}"
    if args.obs_std > 0:
        ds += f"_noise_{args.obs_std}"
    return ds

def _ckpt_path(seg_len):
    return os.path.join(results_dir,
        f"PointwiseStateFlow_{args.flow_type}_{_data_set_label(seg_len)}.pt")

def _build_model(seg_len):
    return PointwiseStateFlow(
        d_x=args.N, d_y=D_y, d_u=d_u,
        d_enc=args.d_enc, d_out=args.d_out,
        window_size=seg_len,
        encoder_type=args.encoder_type,
        gru_hidden=args.gru_hidden,
        d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, n_layers=args.n_layers,
        flow_type=args.flow_type,
        n_coupling=args.n_coupling,
        hidden_size=args.hidden_size, n_hidden=args.n_hidden,
        num_bins=args.num_bins,
        key=jr.PRNGKey(42),
    )

def _load_model(seg_len):
    model = _build_model(seg_len)
    model = eqx.tree_deserialise_leaves(_ckpt_path(seg_len), model)
    return eqx.nn.inference_mode(model)

# ---------------------------------------------------------------------------
# Dataset  (generated once, shared across all seg_len evaluations)
# ---------------------------------------------------------------------------
D_y = args.N // args.obs_interval
d_u = 1

k_data = jr.PRNGKey(args.seed)
data   = generate_lorenz96_dataset(
    k_data, n_samples=args.n_samples, horizon=args.horizon,
    N=args.N, obs_interval=args.obs_interval, obs_std=args.obs_std,
    dt=args.dt, F=args.F, spinup_steps=args.spinup_steps, d_u=d_u,
)
_, _, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=args.train_frac, val_frac=args.val_frac,
)

x_test  = np.array(test_data["x"])   # (M_test, H, N)
y_test  = np.array(test_data["y"])
M_test, H, N = x_test.shape
t         = np.arange(1, H + 1) * args.dt
obs_idx   = np.arange(0, N, args.obs_interval)
unobs_idx = np.array([i for i in range(N) if i not in obs_idx])
bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(N)]

# ---------------------------------------------------------------------------
# Load the requested checkpoint and sample
# ---------------------------------------------------------------------------
ckpt = _ckpt_path(args.seg_len)
assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"

model = _load_model(args.seg_len)

print(f"Sampling {args.num_samples} draws for {M_test} test windows "
      f"(flow={args.flow_type}, encoder={args.encoder_type}, W={args.seg_len}) ...")
samples = np.array(
    model.sample(test_data["y"], test_data["u"], test_data["x0"],
                 key=jr.PRNGKey(1), num_samples=args.num_samples)
)  # (S, M_test, H, N)

post_mean = samples.mean(axis=0)
lo  = np.percentile(samples, (100 - args.ci) / 2,       axis=0)
hi  = np.percentile(samples, 100 - (100 - args.ci) / 2, axis=0)

rmse_all   = float(np.mean(np.sqrt(np.mean((x_test - post_mean)**2,                          axis=(1,2)))))
rmse_obs   = float(np.mean(np.sqrt(np.mean((x_test[:,:,obs_idx]   - post_mean[:,:,obs_idx]  )**2, axis=(1,2)))))
rmse_unobs = float(np.mean(np.sqrt(np.mean((x_test[:,:,unobs_idx] - post_mean[:,:,unobs_idx])**2, axis=(1,2)))))
coverage   = float(np.mean((x_test >= lo) & (x_test <= hi)))

print(f"Test RMSE (all):      {rmse_all:.4f}")
print(f"Test RMSE (observed): {rmse_obs:.4f}")
print(f"Test RMSE (unobs):    {rmse_unobs:.4f}")
print(f"Coverage @ {args.ci:.0f}% CI:  {coverage*100:.1f}%")

# ---------------------------------------------------------------------------
# Figure 1: per-window trajectory panels
# ---------------------------------------------------------------------------
n_win  = min(args.n_windows, M_test)
n_show = min(4, N)
n_cols = 2
n_rows = (n_win + 1) // n_cols

fig1, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
title = (f"PointwiseStateFlow ({args.flow_type}, {args.encoder_type}, W={args.seg_len})"
         f"  -  Lorenz-96 N={N}  -  {args.ci:.0f}% CI\n"
         f"RMSE all={rmse_all:.3f}  obs={rmse_obs:.3f}  unobs={rmse_unobs:.3f}"
         f"  coverage={coverage*100:.1f}%")
fig1.suptitle(title, fontsize=10)

colors_obs   = plt.cm.Blues(np.linspace(0.5, 0.9, len(obs_idx[:n_show])))
colors_unobs = plt.cm.Oranges(np.linspace(0.5, 0.9, len(unobs_idx[:n_show])))

for wi, ax in enumerate(np.array(axes).flat):
    if wi >= n_win:
        ax.set_visible(False)
        continue
    for ci_idx, dim in enumerate(obs_idx[:n_show]):
        c = colors_obs[ci_idx]
        ax.fill_between(t, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.25)
        ax.plot(t, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t, x_test[wi,   :, dim], color=c, lw=1.5, ls="-")
        oi = np.where(obs_idx == dim)[0][0]
        ax.scatter(t, y_test[wi, :, oi], color=c, s=6, zorder=5, marker="o")
    for ci_idx, dim in enumerate(unobs_idx[:n_show]):
        c = colors_unobs[ci_idx]
        ax.fill_between(t, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.25)
        ax.plot(t, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t, x_test[wi,   :, dim], color=c, lw=1.5, ls="-")
    ax.set_title(f"Window {wi}", fontsize=9)
    ax.set_xlabel("Time"); ax.set_ylabel("State value")

legend_elems = [
    Line2D([0],[0], color="steelblue",  lw=1.5, ls="-",          label="Truth (observed)"),
    Line2D([0],[0], color="steelblue",  lw=1.5, ls="--",         label="Post. mean (observed)"),
    Line2D([0],[0], color="steelblue",  marker="o", ls="none", ms=5, label="Observation y"),
    Line2D([0],[0], color="darkorange", lw=1.5, ls="-",          label="Truth (unobserved)"),
    Line2D([0],[0], color="darkorange", lw=1.5, ls="--",         label="Post. mean (unobserved)"),
    plt.Rectangle((0,0),1,1, fc="grey", alpha=0.3,               label=f"{args.ci:.0f}% CI"),
]
fig1.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8,
            bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.05, 1, 1])

# ---------------------------------------------------------------------------
# Figure 2: per-checkpoint diagnostics (3 panels)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(
    f"Diagnostics  -  PointwiseStateFlow ({args.flow_type}, {args.encoder_type}, W={args.seg_len})"
    f"  -  {_data_set_label(args.seg_len)}", fontsize=10,
)

# Panel A: RMSE per dimension
ax = axes2[0]
rmse_per_dim = np.sqrt(np.mean((x_test - post_mean)**2, axis=(0,1)))
ax.bar(np.arange(N), rmse_per_dim, color=bar_colors)
ax.axhline(args.obs_std, color="red", ls="--", lw=1)
ax.set_xlabel("State dimension $i$"); ax.set_ylabel("RMSE")
ax.set_title("RMSE per dimension")
ax.legend(handles=[
    Patch(color="steelblue",  label="Observed"),
    Patch(color="darkorange", label="Unobserved"),
    Line2D([0],[0], color="red", ls="--", label=f"obs_std={args.obs_std}"),
], fontsize=8)

# Panel B: coverage per dimension
ax = axes2[1]
cov_per_dim = np.mean((x_test >= lo) & (x_test <= hi), axis=(0,1))
ax.bar(np.arange(N), cov_per_dim * 100, color=bar_colors)
ax.axhline(args.ci, color="red", ls="--", lw=1, label=f"nominal {args.ci:.0f}%")
ax.set_ylim(0, 105)
ax.set_xlabel("State dimension $i$"); ax.set_ylabel("Coverage (%)")
ax.set_title(f"Empirical {args.ci:.0f}% CI coverage per dimension")
ax.legend(fontsize=8)

# Panel C: calibration scatter (posterior std vs |error|)
ax = axes2[2]
post_std  = samples.std(axis=0)
abs_error = np.abs(x_test - post_mean)
for label, idx, color in [("Observed",   obs_idx,   "steelblue"),
                           ("Unobserved", unobs_idx, "darkorange")]:
    sf = post_std[:,:,idx].ravel()
    ef = abs_error[:,:,idx].ravel()
    ri = np.random.default_rng(0).choice(len(sf), size=min(2000, len(sf)), replace=False)
    ax.scatter(sf[ri], ef[ri], alpha=0.15, s=6, color=color, label=label)
lim = max(post_std.max(), abs_error.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Posterior std"); ax.set_ylabel("|Error|")
ax.set_title("Calibration: std vs |error|")
ax.legend(fontsize=8)

fig2.tight_layout()

# ---------------------------------------------------------------------------
# Figure 3: RMSE vs window size  (scan all available seg_len checkpoints)
# ---------------------------------------------------------------------------
candidate_windows = [5, 10, 15, 20, 25, 30, 40, 50]
available = [(w, _ckpt_path(w)) for w in candidate_windows if os.path.exists(_ckpt_path(w))]

if len(available) > 1:
    print(f"\nFound {len(available)} checkpoints for window-size comparison: "
          f"{[w for w,_ in available]}")
    ws_list, rmse_all_list, rmse_obs_list, rmse_unobs_list = [], [], [], []

    for w, _ in available:
        m = _load_model(w)
        samp_w = np.array(
            m.sample(test_data["y"], test_data["u"], test_data["x0"],
                     key=jr.PRNGKey(1), num_samples=args.num_samples)
        )
        pm_w = samp_w.mean(axis=0)
        ws_list.append(w)
        rmse_all_list.append(float(np.mean(np.sqrt(np.mean((x_test - pm_w)**2, axis=(1,2))))))
        rmse_obs_list.append(float(np.mean(np.sqrt(np.mean((x_test[:,:,obs_idx] - pm_w[:,:,obs_idx])**2, axis=(1,2))))))
        rmse_unobs_list.append(float(np.mean(np.sqrt(np.mean((x_test[:,:,unobs_idx] - pm_w[:,:,unobs_idx])**2, axis=(1,2))))))
        print(f"  W={w:2d}: RMSE all={rmse_all_list[-1]:.4f}  obs={rmse_obs_list[-1]:.4f}  unobs={rmse_unobs_list[-1]:.4f}")

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(ws_list, rmse_all_list,   "k-o",  lw=2, ms=6, label="All dims")
    ax3.plot(ws_list, rmse_obs_list,   "b-s",  lw=2, ms=6, label="Observed")
    ax3.plot(ws_list, rmse_unobs_list, color="darkorange", marker="^", ls="-",
             lw=2, ms=6, label="Unobserved")
    ax3.axhline(args.obs_std, color="red", ls="--", lw=1, label=f"obs_std={args.obs_std}")
    ax3.set_xlabel("Window size W (seg_len)")
    ax3.set_ylabel("Test RMSE")
    ax3.set_title(f"RMSE vs window size  -  "
                  f"PointwiseStateFlow ({args.flow_type}, {args.encoder_type})")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
else:
    fig3 = None
    print("\nOnly one checkpoint found - skipping window-size comparison plot.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
data_set = _data_set_label(args.seg_len)
out_base = args.out or os.path.join(
    results_dir,
    f"PointwiseStateFlow_{args.flow_type}_{args.encoder_type}_{data_set}_eval"
)
fig1.savefig(out_base + "_windows.png",     dpi=150, bbox_inches="tight")
fig2.savefig(out_base + "_diagnostics.png", dpi=150, bbox_inches="tight")
saved = [out_base + "_windows.png", out_base + "_diagnostics.png"]
if fig3 is not None:
    p3 = os.path.join(results_dir,
        f"PointwiseStateFlow_{args.flow_type}_{args.encoder_type}_rmse_vs_window.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    saved.append(p3)
print("Saved:\n" + "\n".join(f"  {p}" for p in saved))
