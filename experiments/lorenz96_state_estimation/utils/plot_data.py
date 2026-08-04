"""
Plot generated Lorenz-96 dataset: trajectories, observations, and state-space structure.

Usage:
    python plot_lorenz96_data.py [--N 12] [--horizon 20] [--n_samples 200] [--seed 0]
"""
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import argparse
import numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset,
    generate_lorenz96_trajectory,
)

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=12, help="State dimension")
parser.add_argument("--horizon", type=int, default=20, help="Steps per window")
parser.add_argument("--obs_interval", type=int, default=2)
parser.add_argument("--obs_std", type=float, default=0.5)
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument("--F", type=float, default=8.0)
parser.add_argument("--spinup_steps", type=int, default=500)
parser.add_argument("--n_samples", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None, help="Save path (default: show)")
args = parser.parse_args()

key = jr.PRNGKey(args.seed)
k_data, k_traj = jr.split(key)

# ── Generate dataset ──────────────────────────────────────────────────────────
data = generate_lorenz96_dataset(
    k_data,
    n_samples=args.n_samples,
    horizon=args.horizon,
    N=args.N,
    obs_interval=args.obs_interval,
    obs_std=args.obs_std,
    dt=args.dt,
    F=args.F,
    spinup_steps=args.spinup_steps,
    d_u=1,
)

x  = np.array(data["x"])   # (M, H, N)
y  = np.array(data["y"])   # (M, H, D_y)
x0 = np.array(data["x0"])  # (M, N)

M, H, N = x.shape
D_y = y.shape[-1]
t = np.arange(H) * args.dt

# ── Long trajectory for space-time / attractor plots ─────────────────────────
long_steps = 3000
traj = np.array(
    generate_lorenz96_trajectory(k_traj, args.N, long_steps, args.dt, args.F, spinup_steps=args.spinup_steps)
)  # (long_steps+1, N)
t_long = np.arange(traj.shape[0]) * args.dt

# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 14))
fig.suptitle(f"Lorenz-96  (N={N}, F={args.F}, dt={args.dt}, obs_interval={args.obs_interval}, obs_std={args.obs_std})",
             fontsize=13, y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 1. Space-time plot of a long trajectory ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
im = ax1.imshow(
    traj.T,
    aspect="auto",
    extent=[0, t_long[-1], -0.5, N - 0.5],
    origin="lower",
    cmap="RdBu_r",
    interpolation="nearest",
)
ax1.set_xlabel("Time")
ax1.set_ylabel("State index $i$")
ax1.set_title("Space-time plot of Lorenz-96 (single long trajectory)")
plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01, label="$X_i$")

# ── 2. A few individual component trajectories (truth vs observation) ─────────
n_show = min(4, N)
obs_indices = list(range(0, N, args.obs_interval))[:D_y]

sample_idx = 0  # show window 0
ax2 = fig.add_subplot(gs[1, :2])
colors = plt.cm.tab10(np.linspace(0, 0.5, n_show))
for ci, comp in enumerate(range(n_show)):
    ax2.plot(t, x[sample_idx, :, comp], color=colors[ci], lw=1.5, label=f"$X_{{{comp}}}$ true")
    if comp in obs_indices:
        oi = obs_indices.index(comp)
        ax2.scatter(t, y[sample_idx, :, oi], color=colors[ci], s=15, marker="x", zorder=5,
                    label=f"$Y_{{{oi}}}$ obs")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
ax2.set_title(f"Sample window #{sample_idx}: truth (lines) vs observations (×)")
ax2.legend(fontsize=7, ncol=2, loc="upper right")

# ── 3. Histogram of state values across dataset ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
vals = x.reshape(-1)
obs_vals = y.reshape(-1)
ax3.hist(vals, bins=60, alpha=0.6, density=True, label="True states")
ax3.hist(obs_vals, bins=60, alpha=0.6, density=True, label="Observations")
ax3.set_xlabel("Value")
ax3.set_ylabel("Density")
ax3.set_title("Marginal distribution")
ax3.legend(fontsize=8)

# ── 4. Correlation across state indices (averaged over time & samples) ────────
ax4 = fig.add_subplot(gs[2, 0])
x_flat = x.reshape(-1, N)  # (M*H, N)
corr = np.corrcoef(x_flat.T)
im4 = ax4.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax4.set_title("State correlation matrix")
ax4.set_xlabel("Index $j$")
ax4.set_ylabel("Index $i$")
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# ── 5. RMS per state index ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
rms_per_dim = np.sqrt(np.mean(x_flat ** 2, axis=0))
ax5.bar(np.arange(N), rms_per_dim, color="steelblue")
ax5.set_xlabel("State index $i$")
ax5.set_ylabel("RMS")
ax5.set_title("Per-dimension RMS (dataset)")

# ── 6. Multiple window trajectories for component 0 ──────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
n_win = min(30, M)
comp0 = x[:n_win, :, 0]  # (n_win, H)
for wi in range(n_win):
    ax6.plot(t, comp0[wi], color="steelblue", alpha=0.3, lw=0.8)
ax6.set_xlabel("Time")
ax6.set_ylabel("$X_0$")
ax6.set_title(f"First {n_win} windows, component 0")

# ── Save or show ──────────────────────────────────────────────────────────────
plt.tight_layout()
if args.out:
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")
else:
    out_path = os.path.join(
        project_root, "results", "results_state_estimation",
        f"lorenz96_data_N{N}_h{args.horizon}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()
