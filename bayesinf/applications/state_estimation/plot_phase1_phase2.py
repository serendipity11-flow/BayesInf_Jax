"""
Plot Phase 1 (ODE) and Phase 2 (SDE prior) trajectories vs ground truth.

Loads the Phase-2 checkpoint (ODE + RealNVPPrior trained, correction frozen/untrained)
and visualises:
  - Ground truth state
  - predict_ode  : deterministic ODE rollout  (Phase 1)
  - predict_only : stochastic SDE rollout     (Phase 2)

Usage
-----
    python plot_phase1_phase2.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.kalman_score_flow_sde import KalmanScoreFlowSDE
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
)

# ---------------------------------------------------------------------------
# Config — must match the saved checkpoint
# ---------------------------------------------------------------------------

N            = 12
H            = 100
dt           = 0.02
obs_interval = 2
obs_std      = 0.5
obs_df       = 5.0
seg_len      = 5
n_modes      = 1
decoder_modes= 1
corr_type    = "realnvp"
seed         = 0
n_samples    = 2000
train_frac   = 0.8
val_frac     = 0.1
spinup_steps = 500
F_lorenz     = 8.0
d_enc        = 64
d_hidden_gru = 64
n_prior_layers = 4
n_corr_layers  = 4
hidden         = 64
fm_steps       = 10

results_dir  = "/home/zhangyi/bayesinf_jax/results/results_state_estimation"
obs_tag      = f"_df{obs_df}"
data_tag     = (f"lorenz96_N{N}_H{H}_dt{dt}"
                f"_oi{obs_interval}_os{obs_std}{obs_tag}"
                f"_seg{seg_len}")
model_tag    = (f"KalmanSFSDE_{data_tag}"
                f"_modes{n_modes}_dec{decoder_modes}"
                f"_{corr_type}")

ckpt_p2   = os.path.join(results_dir, f"{model_tag}_phase2.pt")
stats_path = os.path.join(results_dir, f"{model_tag}_norm_stats.npz")
save_path  = os.path.join(results_dir, f"{model_tag}_phase1_phase2_trajectories.pdf")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

d_z = N
d_y = N // obs_interval

k_data, k_model, k_train = jr.split(jr.PRNGKey(seed), 3)

print("Generating Lorenz-96 data ...")
raw = generate_lorenz96_dataset(
    k_data,
    n_samples    = n_samples,
    horizon      = H,
    N            = N,
    obs_interval = obs_interval,
    obs_std      = obs_std,
    obs_df       = obs_df,
    dt           = dt,
    F            = F_lorenz,
    spinup_steps = spinup_steps,
)

H_obs = raw["H"]
k_y0a, k_y0b = jr.split(jr.fold_in(k_data, 99))
y0 = jnp.dot(raw["x0"], H_obs.T)
if obs_std > 0:
    alpha = obs_df / 2.0
    tau   = jr.gamma(k_y0a, alpha, shape=(y0.shape[0], 1)) / alpha
    y0    = y0 + obs_std * jr.normal(k_y0b, y0.shape) / jnp.sqrt(tau)

x_full = jnp.concatenate([raw["x0"][:, None, :], raw["x"]], axis=1)
y_full = jnp.concatenate([y0[:, None, :],         raw["y"]], axis=1)
ts     = dt * jnp.arange(H + 1)

T_use = (H // seg_len) * seg_len
if T_use < H:
    x_full = x_full[:, :T_use + 1]
    y_full = y_full[:, :T_use + 1]
    ts     = ts[:T_use + 1]

data = {"x": x_full, "y": y_full}
train_data, val_data, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=train_frac, val_frac=val_frac,
)

# Normalisation stats
stats = np.load(stats_path)
x_min_np, x_range_np = stats["x_min"], stats["x_range"]
y_min_np, y_range_np = stats["y_min"], stats["y_range"]
x_min_jnp   = jnp.array(x_min_np)
x_range_jnp = jnp.array(x_range_np)

for split in [train_data, val_data, test_data]:
    split["x"] = (split["x"] - jnp.array(x_min_np)) / jnp.array(x_range_np)
    split["y"] = (split["y"] - jnp.array(y_min_np)) / jnp.array(y_range_np)

# ---------------------------------------------------------------------------
# Model + checkpoint
# ---------------------------------------------------------------------------

model = KalmanScoreFlowSDE(
    d_z            = d_z,
    d_y            = d_y,
    d_enc          = d_enc,
    d_hidden_gru   = d_hidden_gru,
    n_prior_layers = n_prior_layers,
    n_corr_layers  = n_corr_layers,
    hidden         = hidden,
    seg_len        = seg_len,
    corr_type      = corr_type,
    fm_steps       = fm_steps,
    n_modes        = n_modes,
    decoder_modes  = decoder_modes,
    key            = k_model,
)
print(f"Loading Phase-2 checkpoint: {ckpt_p2}")
model = eqx.tree_deserialise_leaves(ckpt_p2, model)

# ---------------------------------------------------------------------------
# Run predictions on a few test samples
# ---------------------------------------------------------------------------

N_PLOT   = 4      # number of trajectories to plot
N_DIMS   = 4      # state dimensions to show (first N_DIMS)
N_SDE    = 3      # number of stochastic SDE samples per trajectory

test_x = test_data["x"][:N_PLOT]   # (N_PLOT, T+1, d_z)  normalised
ts_np  = np.array(ts)

@eqx.filter_jit
def _ode_batch(model, z0_batch, ts):
    return jax.vmap(lambda z0: model.predict_ode(z0, ts))(z0_batch)

@eqx.filter_jit
def _sde_batch(model, z0_batch, ts, keys):
    return jax.vmap(lambda z0, k: model.predict_only(z0, ts, k))(z0_batch, keys)

print("Running ODE predictions ...")
z_ode = np.array(_ode_batch(model, test_x[:, 0], ts))      # (N_PLOT, T+1, d_z)

print("Running SDE predictions ...")
rng = jr.PRNGKey(seed + 42)
sde_samples = []
for _ in range(N_SDE):
    rng, k = jr.split(rng)
    ks = jr.split(k, N_PLOT)
    z_sde = np.array(_sde_batch(model, test_x[:, 0], ts, ks))
    sde_samples.append(z_sde)
# sde_samples: list of N_SDE arrays each (N_PLOT, T+1, d_z)

# De-normalise
def denorm(z_norm):
    return z_norm * x_range_np + x_min_np

z_true_dn  = denorm(np.array(test_x))
z_ode_dn   = denorm(z_ode)
sde_dn     = [denorm(s) for s in sde_samples]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

dims_to_plot = list(range(min(N_DIMS, d_z)))
t_arr = ts_np

fig, axes = plt.subplots(
    N_PLOT, len(dims_to_plot),
    figsize=(4 * len(dims_to_plot), 3 * N_PLOT),
    sharex=True,
)

for i in range(N_PLOT):
    for j, dim in enumerate(dims_to_plot):
        ax = axes[i, j]

        # Ground truth
        ax.plot(t_arr, z_true_dn[i, :, dim],
                color="black", lw=1.5, label="Ground truth", zorder=5)

        # ODE (Phase 1)
        ax.plot(t_arr, z_ode_dn[i, :, dim],
                color="tab:blue", lw=1.2, ls="--", label="ODE (Phase 1)", zorder=4)

        # SDE samples (Phase 2)
        for s_idx, sde in enumerate(sde_dn):
            lbl = "SDE (Phase 2)" if s_idx == 0 else None
            ax.plot(t_arr, sde[i, :, dim],
                    color="tab:orange", lw=0.8, alpha=0.6, label=lbl, zorder=3)

        if i == 0:
            ax.set_title(f"State dim {dim}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"Sample {i+1}", fontsize=9)
        if i == N_PLOT - 1:
            ax.set_xlabel("t")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=9, ncol=3)

fig.suptitle(
    f"Lorenz-96 (N={N})  —  Phase 1 (ODE) vs Phase 2 (SDE prior) trajectories\n"
    f"(starting from true z₀, no correction)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig(save_path, bbox_inches="tight")
print(f"Saved → {save_path}")
