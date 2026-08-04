"""
Plot state estimation results: ground truth, posterior mean, and confidence intervals.

Loads a trained AmortizedStateFlow checkpoint, runs posterior sampling on test
windows, and produces:
  1. Per-window panels: truth vs posterior mean ± CI for observed/unobserved dims
  2. RMSE per state dimension (observed vs unobserved)
  3. Coverage: fraction of truth inside 90% CI per dimension
  4. Aggregate posterior spread vs error scatter

Usage (same flags as training script):
    python3 -m experiments.lorenz96_state_estimation.utils.plot_state_estimation_results \
        --flow_type realnvp --n_coupling 4 --horizon 10 --seg_len 5 \
        --encoder_type transformer --num_samples 50 --n_windows 6
"""
import sys, os, argparse

import numpy as np
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from bayesinf.applications.state_estimation.amortized_state_flow import AmortizedStateFlow
from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    state_rmse,
)

# ---------------------------------------------------------------------------
# Arguments  (mirror training script so checkpoint path resolves correctly)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--flow_type",    type=str,   default="realnvp",
                    choices=("realnvp", "nsf", "maf", "fm", "sbm"))
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
parser.add_argument("--n_coupling",   type=int,   default=8)
parser.add_argument("--hidden_size",  type=int,   default=128)
parser.add_argument("--n_hidden",     type=int,   default=2)
parser.add_argument("--d_enc",        type=int,   default=64)
parser.add_argument("--d_out",        type=int,   default=96)
parser.add_argument("--encoder_type", type=str,   default="transformer",
                    choices=("gru", "transformer"))
parser.add_argument("--gru_hidden",   type=int,   default=32)
parser.add_argument("--prior",        type=str,   default="iid",
                    choices=("wiener_scaled_t", "wiener_increment", "iid"))
parser.add_argument("--seed",         type=int,   default=0)
# Plot-specific
parser.add_argument("--num_samples",  type=int,   default=50,
                    help="Posterior samples per window for CI")
parser.add_argument("--n_windows",    type=int,   default=6,
                    help="Number of test windows to plot individually")
parser.add_argument("--ci",           type=float, default=90.0,
                    help="Confidence interval percentage (default 90)")
parser.add_argument("--seg_len",      type=int,   default=5,
                    help="Must match the seg_len used during training (0 = no segmentation)")
parser.add_argument("--out",          type=str,   default=None,
                    help="Output path; default saves next to checkpoint")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths (mirror training script)
# ---------------------------------------------------------------------------
results_dir = "/home/zhangyi/bayesinf_jax/results/results_state_estimation"
D_y  = args.N // args.obs_interval
d_u  = 1
_seg_label = (f"seg{args.seg_len}" if args.seg_len > 0 and args.horizon > args.seg_len
              else f"h{args.horizon}")
data_set = f"lorenz96_N{args.N}_{_seg_label}_obs{args.obs_interval}"
if args.obs_std > 0:
    data_set += f"_noise_{args.obs_std}"
ckpt_path = os.path.join(results_dir, f"AmortizedStateFlow_{args.flow_type}_{data_set}.pt")
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
k_data = jr.PRNGKey(args.seed)
data = generate_lorenz96_dataset(
    k_data, n_samples=args.n_samples, horizon=args.horizon,
    N=args.N, obs_interval=args.obs_interval, obs_std=args.obs_std,
    dt=args.dt, F=args.F, spinup_steps=args.spinup_steps, d_u=d_u,
)
_, _, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1), train_frac=args.train_frac, val_frac=args.val_frac
)

x_test  = np.array(test_data["x"])   # (M_test, H, N)
y_test  = np.array(test_data["y"])   # (M_test, H, D_y)
u_test  = np.array(test_data["u"])
x0_test = np.array(test_data["x0"])

M_test, H, N = x_test.shape
t = np.arange(1, H + 1) * args.dt   # physical time axis

# Observed state indices
obs_idx = np.arange(0, N, args.obs_interval)   # length D_y
unobs_idx = np.array([i for i in range(N) if i not in obs_idx])

# ---------------------------------------------------------------------------
# Model + sampling
# ---------------------------------------------------------------------------
model = AmortizedStateFlow(
    d_x=N, d_y=D_y, d_u=d_u,
    d_enc=args.d_enc, d_out=args.d_out,
    key=jr.PRNGKey(42),
    flow_type=args.flow_type,
    n_coupling=args.n_coupling,
    hidden_size=args.hidden_size,
    n_hidden=args.n_hidden,
    encoder_type=args.encoder_type,
    gru_hidden=args.gru_hidden,
)
model = eqx.tree_deserialise_leaves(ckpt_path, model)
model = eqx.nn.inference_mode(model)

import jax.numpy as jnp

seg_len = args.seg_len
use_seg = seg_len > 0 and H > seg_len
n_segs  = H // seg_len if use_seg else 1
H_eff   = n_segs * seg_len if use_seg else H

print(f"Sampling {args.num_samples} posterior draws for {M_test} test windows "
      f"({'segmented: ' + str(n_segs) + '×' + str(seg_len) + ' steps' if use_seg else 'full horizon ' + str(H)}) …")

if use_seg:
    # Build per-segment inputs matching training: (M_test*n_segs, seg_len, *)
    x_in  = test_data["x"][:, :H_eff]
    y_in  = test_data["y"][:, :H_eff]
    u_in  = test_data["u"][:, :H_eff]
    x0_in = test_data["x0"]

    if n_segs > 1:
        boundaries = x_in[:, seg_len - 1 : H_eff - 1 : seg_len, :]  # (M, n_segs-1, N)
        x0_segs = jnp.concatenate([x0_in[:, None, :], boundaries], axis=1)  # (M, n_segs, N)
    else:
        x0_segs = x0_in[:, None, :]

    y_seg  = y_in.reshape(M_test, n_segs, seg_len, -1).reshape(M_test * n_segs, seg_len, -1)
    u_seg  = u_in.reshape(M_test, n_segs, seg_len, -1).reshape(M_test * n_segs, seg_len, -1)
    x0_seg = x0_segs.reshape(M_test * n_segs, N)

    # Sample per segment: (S, M_test*n_segs, seg_len, N)
    seg_samples = np.array(
        model.sample(y_seg, u_seg, x0_seg,
                     key=jr.PRNGKey(1), num_samples=args.num_samples, prior=args.prior)
    )
    # Reconstruct full time series: (S, M_test, H_eff, N)
    samples = seg_samples.reshape(args.num_samples, M_test, n_segs, seg_len, N) \
                         .reshape(args.num_samples, M_test, H_eff, N)
    x_test = np.array(x_in)
    y_test = np.array(y_in)
else:
    samples = np.array(
        model.sample(
            test_data["y"], test_data["u"], test_data["x0"],
            key=jr.PRNGKey(1), num_samples=args.num_samples, prior=args.prior,
        )
    )

# Time axis over the full reconstructed horizon
t = np.arange(1, H_eff + 1) * args.dt
# Segment boundary positions for vertical markers in plots
seg_boundaries = (np.arange(1, n_segs) * seg_len * args.dt) if use_seg else np.array([])

post_mean = samples.mean(axis=0)                               # (M_test, H, N)
lo = np.percentile(samples, (100 - args.ci) / 2,     axis=0)  # (M_test, H, N)
hi = np.percentile(samples, 100 - (100 - args.ci) / 2, axis=0)

rmse_all  = float(np.mean(np.sqrt(np.mean((x_test - post_mean) ** 2, axis=(1, 2)))))
rmse_obs  = float(np.mean(np.sqrt(np.mean((x_test[:, :, obs_idx] - post_mean[:, :, obs_idx]) ** 2, axis=(1, 2)))))
rmse_unobs = float(np.mean(np.sqrt(np.mean((x_test[:, :, unobs_idx] - post_mean[:, :, unobs_idx]) ** 2, axis=(1, 2)))))
coverage  = float(np.mean((x_test >= lo) & (x_test <= hi)))

print(f"Test RMSE (all):      {rmse_all:.4f}")
print(f"Test RMSE (observed): {rmse_obs:.4f}")
print(f"Test RMSE (unobs):    {rmse_unobs:.4f}")
print(f"Coverage @ {args.ci:.0f}% CI:  {coverage*100:.1f}%")

# ---------------------------------------------------------------------------
# Figure 1: per-window panels
# ---------------------------------------------------------------------------
n_win   = min(args.n_windows, M_test)
n_show  = min(4, N)   # state components per panel
n_cols  = 2
n_rows  = (n_win + 1) // n_cols

fig1, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows),
                           sharex=False, sharey=False)
fig1.suptitle(
    f"AmortizedStateFlow ({args.flow_type})  –  Lorenz-96 N={N}  –  {args.ci:.0f}% CI\n"
    f"RMSE all={rmse_all:.3f}  obs={rmse_obs:.3f}  unobs={rmse_unobs:.3f}  "
    f"coverage={coverage*100:.1f}%",
    fontsize=11,
)

colors_obs   = plt.cm.Blues(np.linspace(0.5, 0.9, len(obs_idx[:n_show])))
colors_unobs = plt.cm.Oranges(np.linspace(0.5, 0.9, len(unobs_idx[:n_show])))

for wi, ax in enumerate(np.array(axes).flat):
    if wi >= n_win:
        ax.set_visible(False)
        continue

    # Observed dims
    for ci_idx, dim in enumerate(obs_idx[:n_show]):
        c = colors_obs[ci_idx]
        ax.fill_between(t, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.25)
        ax.plot(t, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t, x_test[wi, :, dim],   color=c, lw=1.5, ls="-")
        oi = np.where(obs_idx == dim)[0][0]
        ax.scatter(t, y_test[wi, :, oi], color=c, s=6, zorder=5, marker="o")

    # Unobserved dims
    for ci_idx, dim in enumerate(unobs_idx[:n_show]):
        c = colors_unobs[ci_idx]
        ax.fill_between(t, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.25)
        ax.plot(t, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t, x_test[wi, :, dim],   color=c, lw=1.5, ls="-")

    # Segment boundary markers
    for sb in seg_boundaries:
        ax.axvline(sb, color="grey", lw=0.8, ls=":", alpha=0.7)

    ax.set_title(
        f"Window {wi}  (observed + unobserved; H={H_eff}, {n_segs}×{seg_len} segments)",
        fontsize=9,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("State value")

legend_elems = [
    Line2D([0], [0], color="steelblue", lw=1.5, ls="-",  label="Truth (observed dim)"),
    Line2D([0], [0], color="steelblue", lw=1.5, ls="--", label="Post. mean (observed)"),
    Line2D([0], [0], color="steelblue", marker="o", ls="none", ms=5, label="Observation y"),
    Line2D([0], [0], color="darkorange", lw=1.5, ls="-",  label="Truth (unobserved dim)"),
    Line2D([0], [0], color="darkorange", lw=1.5, ls="--", label="Post. mean (unobserved)"),
    plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3,     label=f"{args.ci:.0f}% CI"),
    Line2D([0], [0], color="grey", lw=0.8, ls=":",         label="Segment boundary"),
]
fig1.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8,
            bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.05, 1, 1])

# ---------------------------------------------------------------------------
# Figure 2: diagnostics
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(f"Diagnostics  –  {args.flow_type}  –  {data_set}", fontsize=11)

# Panel A: per-dimension RMSE, coloured by observed/unobserved
ax = axes2[0]
rmse_per_dim = np.sqrt(np.mean((x_test - post_mean) ** 2, axis=(0, 1)))  # (N,)
bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(N)]
ax.bar(np.arange(N), rmse_per_dim, color=bar_colors)
ax.axhline(args.obs_std, color="red", ls="--", lw=1, label=f"obs_std={args.obs_std}")
ax.set_xlabel("State dimension $i$")
ax.set_ylabel("RMSE")
ax.set_title("RMSE per dimension")
ax.legend(fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="steelblue",  label="Observed"),
    Patch(color="darkorange", label="Unobserved"),
    Line2D([0],[0], color="red", ls="--", label=f"obs_std={args.obs_std}"),
], fontsize=8)

# Panel B: coverage per dimension
ax = axes2[1]
coverage_per_dim = np.mean((x_test >= lo) & (x_test <= hi), axis=(0, 1))  # (N,)
ax.bar(np.arange(N), coverage_per_dim * 100, color=bar_colors)
ax.axhline(args.ci, color="red", ls="--", lw=1, label=f"nominal {args.ci:.0f}%")
ax.set_ylim(0, 105)
ax.set_xlabel("State dimension $i$")
ax.set_ylabel("Coverage (%)")
ax.set_title(f"Empirical {args.ci:.0f}% CI coverage per dimension")
ax.legend(fontsize=8)

# Panel C: posterior std vs absolute error (scatter, one point per dim per window)
ax = axes2[2]
post_std   = samples.std(axis=0)                               # (M_test, H, N)
abs_error  = np.abs(x_test - post_mean)
for label, idx, color in [("Observed", obs_idx, "steelblue"),
                           ("Unobserved", unobs_idx, "darkorange")]:
    std_flat = post_std[:, :, idx].ravel()
    err_flat = abs_error[:, :, idx].ravel()
    # Subsample to avoid overplotting
    rng_idx  = np.random.default_rng(0).choice(len(std_flat),
                                                size=min(2000, len(std_flat)),
                                                replace=False)
    ax.scatter(std_flat[rng_idx], err_flat[rng_idx],
               alpha=0.15, s=6, color=color, label=label)
lim = max(post_std.max(), abs_error.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Posterior std")
ax.set_ylabel("|Error|")
ax.set_title("Calibration: std vs |error|")
ax.legend(fontsize=8)

fig2.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_base = args.out or os.path.join(
    results_dir, f"AmortizedStateFlow_{args.flow_type}_{data_set}_eval"
)
fig1.savefig(out_base + "_windows.png", dpi=150, bbox_inches="tight")
fig2.savefig(out_base + "_diagnostics.png", dpi=150, bbox_inches="tight")
print(f"Saved:\n  {out_base}_windows.png\n  {out_base}_diagnostics.png")
