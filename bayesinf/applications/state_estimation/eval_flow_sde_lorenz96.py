"""
Evaluation and plotting for NICEFlowSDE on Lorenz-96.

Produces three figures:
  1. Trajectory panels: truth / posterior mean / 90% CI / obs dots
  2. Diagnostics:       RMSE per dim | coverage per dim | calibration scatter
  3. GMM analysis:      particle histograms at a fixed t | marginal distributions

Usage
-----
# Must match --train arguments used in train_flow_sde_lorenz96.py
python3 -m bayesinf.applications.state_estimation.eval_flow_sde_lorenz96 \\
    --obs_noise_type gmm --N 12 --horizon 100
"""

import sys, os, argparse
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.flow_sde import NICEFlowSDE
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    lorenz96_rhs,
)


# ---------------------------------------------------------------------------
# Arguments  (must match training script)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate NICEFlowSDE on Lorenz-96")
# Dataset — must match training
parser.add_argument("--horizon",       type=int,   default=100)
parser.add_argument("--N",             type=int,   default=12)
parser.add_argument("--obs_interval",  type=int,   default=2)
parser.add_argument("--obs_std",       type=float, default=0.5)
parser.add_argument("--obs_noise_type",type=str,   default="gmm",
                    choices=("gaussian", "student_t", "gmm"))
parser.add_argument("--obs_df",        type=float, default=5.0)
parser.add_argument("--gmm_weights",   type=str,   default="0.70,0.15,0.15")
parser.add_argument("--gmm_offsets",   type=str,   default="0.0,2.5,-2.5")
parser.add_argument("--gmm_stds",      type=str,   default="0.50,1.50,1.50")
parser.add_argument("--dt",            type=float, default=0.02)
parser.add_argument("--F_lorenz",      type=float, default=8.0)
parser.add_argument("--spinup_steps",  type=int,   default=500)
parser.add_argument("--n_samples",     type=int,   default=2000)
parser.add_argument("--train_frac",    type=float, default=0.8)
parser.add_argument("--val_frac",      type=float, default=0.1)
# Model — must match training
parser.add_argument("--d_c",           type=int,   default=64)
parser.add_argument("--d_hidden_enc",  type=int,   default=64)
parser.add_argument("--d_hidden",      type=int,   default=128)
parser.add_argument("--depth",         type=int,   default=3)
parser.add_argument("--time_emb_dim",  type=int,   default=32)
# Eval-specific
parser.add_argument("--seed",          type=int,   default=0)
parser.add_argument("--n_eval",        type=int,   default=100,
                    help="Number of test sequences to evaluate")
parser.add_argument("--n_plot_seqs",   type=int,   default=6,
                    help="Number of sequences to show in trajectory panel")
parser.add_argument("--n_particles",   type=int,   default=200,
                    help="Posterior particles per sequence for CI")
parser.add_argument("--ci",            type=float, default=90.0)
parser.add_argument("--sampler",        type=str,   default="gt_segmented",
                    choices=("euler", "heun", "reversible_heun", "midpoint",
                             "milstein", "adaptive_euler", "rk4", "gt_segmented"),
                    help="Inference sampler/solver")
parser.add_argument("--seg_length",     type=int,   default=20,
                    help="Segment length for gt_segmented inference")
parser.add_argument("--t_hist",         type=float, default=0.5,
                    help="Normalised time ∈ (0,1) for per-particle histogram")
parser.add_argument("--diffusion_temp", type=float, default=1.0,
                    help="Post-hoc diffusion temperature >1 widens CI (no retraining)")
parser.add_argument("--diffusion_reg_weight", type=float, default=0.0,
                    help="Deprecated; kept only so older launch scripts still parse.")
parser.add_argument("--n_nice_layers",  type=int,   default=2,
                    help="Number of NICE coupling layers — must match training")
parser.add_argument("--solver",         type=str,   default="heun",
                    choices=("euler", "heun", "reversible_heun", "midpoint", "milstein", "adaptive_euler", "rk4"),
                    help="SDE solver used during training (for checkpoint lookup)")
parser.add_argument("--use_custom_solver", action="store_true",
                    help="Use custom stochastic solvers instead of diffrax; affects checkpoint lookup")
parser.add_argument("--pathwise",       action="store_true",
                    help="Deprecated; pathwise checkpoints are now assumed.")
parser.add_argument("--use_multiple_shooting", action="store_true")
args = parser.parse_args()
args.pathwise = True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

results_dir = os.path.join(project_root, "results", "results_state_estimation")
d_x = args.N
d_y = args.N // args.obs_interval

def _parse_tuple(s):
    return tuple(float(x) for x in s.split(","))

gmm_weights = _parse_tuple(args.gmm_weights)
gmm_offsets = _parse_tuple(args.gmm_offsets)
gmm_stds    = _parse_tuple(args.gmm_stds)

_noise_tag = args.obs_noise_type
if _noise_tag == "gmm":
    _noise_tag = f"gmm{len(gmm_weights)}c"
elif _noise_tag == "student_t":
    _noise_tag = f"studt{args.obs_df}"

_solver_backend = "custom" if args.use_custom_solver else "diffrax"
_train_mode = f"pw{args.solver}_{_solver_backend}"
_seg_tag    = f"seg{args.seg_length}" if args.use_multiple_shooting else "noMS"
model_tag = (
    f"NICEFlowSDE"
    f"_N{args.N}_h{args.horizon}_obs{args.obs_interval}"
    f"_{_noise_tag}_std{args.obs_std}"
    f"_dc{args.d_c}_dh{args.d_hidden}_dep{args.depth}"
    f"_nl{args.n_nice_layers}"
    f"_{_seg_tag}"
    f"_{_train_mode}"
)
ckpt_path = os.path.join(results_dir, f"{model_tag}.pt")

out_base = os.path.join(results_dir, f"{model_tag}_eval")

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        "Run train_flow_sde_lorenz96.py --train first with matching arguments."
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print("Generating test data …")
k_data = jr.PRNGKey(args.seed)
data = generate_lorenz96_dataset(
    k_data,
    n_samples       = args.n_samples,
    horizon         = args.horizon,
    N               = args.N,
    obs_interval    = args.obs_interval,
    obs_std         = args.obs_std,
    obs_noise_type  = args.obs_noise_type,
    obs_df          = args.obs_df if args.obs_noise_type == "student_t" else None,
    gmm_weights     = gmm_weights if args.obs_noise_type == "gmm" else None,
    gmm_offsets     = gmm_offsets if args.obs_noise_type == "gmm" else None,
    gmm_stds        = gmm_stds    if args.obs_noise_type == "gmm" else None,
    dt              = args.dt,
    F               = args.F_lorenz,
    spinup_steps    = args.spinup_steps,
    d_u             = 1,
)
_, _, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=args.train_frac, val_frac=args.val_frac,
)
x_test_raw = np.array(test_data["x"])   # (M_test, H, d_x)  original scale
y_test_raw = np.array(test_data["y"])   # (M_test, H, d_y)
M_test     = x_test_raw.shape[0]
n_eval     = min(args.n_eval, M_test)
n_plot     = min(args.n_plot_seqs, n_eval)
T          = args.horizon
t_axis     = np.arange(T) * args.dt

obs_idx   = np.arange(0, d_x, args.obs_interval)
unobs_idx = np.array([i for i in range(d_x) if i not in obs_idx])


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

norm_stats_path = ckpt_path.replace(".pt", "_norm_stats.npz")
if not os.path.exists(norm_stats_path):
    raise FileNotFoundError(
        f"Norm stats not found: {norm_stats_path}\n"
        "Re-train with the updated training script to generate them."
    )
ns     = np.load(norm_stats_path)
x_mean = ns["x_mean"]   # (d_x,)
x_std  = ns["x_std"]
y_mean = ns["y_mean"]   # (d_y,)
y_std  = ns["y_std"]
print(f"Norm stats loaded: x mean≈{x_mean.mean():.2f} std≈{x_std.mean():.2f}")

x_test = jnp.array((x_test_raw - x_mean) / x_std)   # normalised
y_test = jnp.array((y_test_raw - y_mean) / y_std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print(f"Loading checkpoint: {ckpt_path}")
_F  = args.F_lorenz
_xm = jnp.array(x_mean)
_xs = jnp.array(x_std)
physics_fn = lambda z: lorenz96_rhs(_xm + _xs * z, _F) / _xs
model = NICEFlowSDE(
    d_x           = d_x,
    d_y           = d_y,
    d_c           = args.d_c,
    d_hidden_enc  = args.d_hidden_enc,
    d_hidden      = args.d_hidden,
    depth         = args.depth,
    time_emb_dim  = args.time_emb_dim,
    n_nice_layers = args.n_nice_layers,
    key           = jr.PRNGKey(42),
    physics_fn    = physics_fn,
)
model = eqx.tree_deserialise_leaves(ckpt_path, model)
model = eqx.nn.inference_mode(model)
print("Checkpoint loaded.")


# ---------------------------------------------------------------------------
# Posterior sampling  (chunked to control memory)
# ---------------------------------------------------------------------------

use_gt_seg   = args.sampler == "gt_segmented"
solver_name  = args.solver if use_gt_seg else args.sampler

@eqx.filter_jit
def _sample_chunk(model, y_chunk, x_chunk, key, n_particles,
                  solver_name, use_gt_seg, seg_length, diffusion_temp, use_custom_solver):
    B    = y_chunk.shape[0]
    keys = jr.split(key, B)
    if use_gt_seg:
        # gt_segmented uses diffrax internally
        return jax.vmap(
            lambda yi, xi, ki: model.sample_posterior_gt_segmented(
                yi, xi, ki, n_particles, seg_length, solver_name,
                diffusion_temp, use_custom_solver
            )
        )(y_chunk, x_chunk, keys)              # (B, n_particles, H, d_x)
    elif use_custom_solver:
        return jax.vmap(
            lambda yi, x0i, ki: model.sample_posterior_custom_solver(
                yi, x0i, ki, n_particles, solver_name,
                diffusion_temp=diffusion_temp,
            )
        )(y_chunk, x_chunk[:, 0, :], keys)    # (B, n_particles, H, d_x)
    else:
        return jax.vmap(
            lambda yi, x0i, ki: model.sample_posterior(
                yi, x0i, ki, n_particles, solver_name,
                diffusion_temp=diffusion_temp,
            )
        )(y_chunk, x_chunk[:, 0, :], keys)    # (B, n_particles, H, d_x)


chunk   = 20
k_eval  = jr.PRNGKey(999)
all_samps = []

print(
    f"Drawing {args.n_particles} posterior samples for {n_eval} sequences "
    f"(sampler={args.sampler}  solver={solver_name}  seg_length={args.seg_length}) …"
)
for start in range(0, n_eval, chunk):
    end  = min(start + chunk, n_eval)
    yb   = y_test[start:end]              # normalised
    xb   = x_test[start:end]             # normalised, full trajectory for gt_seg
    k_c  = jr.fold_in(k_eval, start)
    s    = np.array(_sample_chunk(
        model, yb, xb, k_c, args.n_particles,
        solver_name, use_gt_seg, args.seg_length, args.diffusion_temp, args.use_custom_solver
    ))
    all_samps.append(s)

all_samps  = np.concatenate(all_samps, axis=0)                          # (n_eval, n_p, H, d_x) norm
all_samps  = all_samps * x_std[None, None, None] + x_mean[None, None, None]  # denormalise


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

post_mean = all_samps.mean(axis=1)              # (n_eval, H, d_x)
lo        = np.percentile(all_samps, (100 - args.ci) / 2,       axis=1)
hi        = np.percentile(all_samps, 100 - (100 - args.ci) / 2, axis=1)

x_true = x_test_raw[:n_eval]                     # (n_eval, H, d_x) original scale

rmse_all   = float(np.mean(np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(1, 2)))))
rmse_obs   = float(np.sqrt(np.mean((x_true[:, :, obs_idx]   - post_mean[:, :, obs_idx])   ** 2)))
rmse_unobs = float(np.sqrt(np.mean((x_true[:, :, unobs_idx] - post_mean[:, :, unobs_idx]) ** 2))) if len(unobs_idx) else float("nan")
mae_all    = float(np.mean(np.abs(x_true - post_mean)))

cov_per_dim = np.mean((x_true >= lo) & (x_true <= hi), axis=(0, 1))  # (d_x,)
coverage    = float(cov_per_dim.mean())

print(f"\n── Evaluation  ({n_eval} seqs, {args.n_particles} particles) ──")
print(f"  RMSE posterior mean (all):       {rmse_all:.4f}")
print(f"  RMSE posterior mean (observed):  {rmse_obs:.4f}")
print(f"  RMSE posterior mean (unobserved):{rmse_unobs:.4f}")
print(f"  MAE  posterior mean:             {mae_all:.4f}")
print(f"  Coverage @ {args.ci:.0f}% CI:           {coverage * 100:.1f}%")


# ---------------------------------------------------------------------------
# Figure 1: Trajectory panels
# ---------------------------------------------------------------------------

n_cols  = 2
n_rows  = (n_plot + 1) // n_cols
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows), sharex=False)
fig1.suptitle(
    f"ConditionalNeuralSDE  –  Lorenz-96 N={d_x}  –  noise={args.obs_noise_type}  "
    f"–  {args.ci:.0f}% CI\n"
    f"RMSE={rmse_all:.3f}  obs={rmse_obs:.3f}  unobs={rmse_unobs:.3f}  "
    f"cov={coverage * 100:.1f}%",
    fontsize=10,
)

n_show = min(4, d_x)
c_obs   = plt.cm.Blues(np.linspace(0.45, 0.85, max(1, len(obs_idx[:n_show]))))
c_unobs = plt.cm.Oranges(np.linspace(0.45, 0.85, max(1, len(unobs_idx[:n_show]))))

for wi, ax in enumerate(np.array(axes1).flat):
    if wi >= n_plot:
        ax.set_visible(False)
        continue
    for ci_i, dim in enumerate(obs_idx[:n_show]):
        c = c_obs[ci_i]
        ax.fill_between(t_axis, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.20)
        ax.plot(t_axis, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t_axis, x_true[wi, :, dim],    color=c, lw=1.2, ls="-")
        oi = int(np.where(obs_idx == dim)[0][0])
        ax.scatter(t_axis, y_test_raw[wi, :, oi], color=c, s=4, zorder=5, alpha=0.5)
    for ci_i, dim in enumerate(unobs_idx[:n_show]):
        c = c_unobs[ci_i]
        ax.fill_between(t_axis, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.20)
        ax.plot(t_axis, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t_axis, x_true[wi, :, dim],    color=c, lw=1.2, ls="-")
    ax.set_title(f"Seq {wi}", fontsize=9)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")

legend_elems = [
    Line2D([0], [0], color="steelblue",  lw=1.2, ls="-",  label="Truth (obs dim)"),
    Line2D([0], [0], color="steelblue",  lw=1.5, ls="--", label="Post. mean (obs dim)"),
    Line2D([0], [0], color="steelblue",  marker="o", ls="", ms=4, label="Observation y"),
    Line2D([0], [0], color="darkorange", lw=1.2, ls="-",  label="Truth (unobs dim)"),
    Line2D([0], [0], color="darkorange", lw=1.5, ls="--", label="Post. mean (unobs dim)"),
    Patch(facecolor="grey", alpha=0.3, label=f"{args.ci:.0f}% CI"),
]
fig1.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8,
            bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.05, 1, 1])


# ---------------------------------------------------------------------------
# Figure 2: Diagnostics
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(
    f"ConditionalNeuralSDE diagnostics — {model_tag}\n"
    f"n_eval={n_eval}  n_particles={args.n_particles}",
    fontsize=10,
)
bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(d_x)]

# Panel A: RMSE per dimension
ax = axes2[0]
rmse_per_dim = np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(0, 1)))  # (d_x,)
ax.bar(np.arange(d_x), rmse_per_dim, color=bar_colors)
ax.axhline(args.obs_std, color="red", ls="--", lw=1.2)
ax.set_xlabel("State dimension")
ax.set_ylabel("RMSE")
ax.set_title("RMSE per state dim (posterior mean)")
ax.legend(handles=[
    Patch(color="steelblue",  label="Observed"),
    Patch(color="darkorange", label="Unobserved"),
    Line2D([0], [0], color="red", ls="--", label=f"obs_std={args.obs_std}"),
], fontsize=8)

# Panel B: coverage per dimension
ax = axes2[1]
ax.bar(np.arange(d_x), cov_per_dim * 100, color=bar_colors)
ax.axhline(args.ci, color="red", ls="--", lw=1.2, label=f"nominal {args.ci:.0f}%")
ax.set_ylim(0, 105)
ax.set_xlabel("State dimension")
ax.set_ylabel("Coverage (%)")
ax.set_title(f"Empirical {args.ci:.0f}% CI coverage per dim")
ax.legend(fontsize=8)

# Panel C: calibration scatter (posterior std vs |error|)
ax = axes2[2]
post_std = all_samps[:n_plot].std(axis=1)           # (n_plot, H, d_x)
abs_err  = np.abs(x_true[:n_plot] - post_mean[:n_plot])
rng_np   = np.random.default_rng(0)

for label, idx, color in [("Observed",   obs_idx,   "steelblue"),
                           ("Unobserved", unobs_idx, "darkorange")]:
    if len(idx) == 0:
        continue
    s_f = post_std[:, :, idx].ravel()
    e_f = abs_err[:, :, idx].ravel()
    sel = rng_np.choice(len(s_f), size=min(5000, len(s_f)), replace=False)
    ax.scatter(s_f[sel], e_f[sel], alpha=0.15, s=6, color=color, label=label)

lim = max(post_std.max(), abs_err.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Posterior std")
ax.set_ylabel("|Error|")
ax.set_title("Calibration: posterior std vs |error|")
ax.legend(fontsize=8)

fig2.tight_layout()


# ---------------------------------------------------------------------------
# Figure 3: Multi-modal analysis
# ---------------------------------------------------------------------------

t_idx  = int(round(args.t_hist * (T - 1)))   # time step index for histogram
t_val  = t_idx * args.dt

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.suptitle(
    f"ConditionalNeuralSDE — Multi-modal analysis\n"
    f"noise={args.obs_noise_type}  obs_std={args.obs_std}  "
    f"n_eval={n_eval}  n_particles={args.n_particles}",
    fontsize=10,
)

# Panel A: Per-particle histogram at t_hist for dim 0 (observed) and first unobserved
ax = axes3[0]
# all_samps: (n_eval, n_p, H, d_x)
particles_t = all_samps[:, :, t_idx, :]   # (n_eval, n_p, d_x)
true_t      = x_true[:, t_idx, :]         # (n_eval, d_x)

for color, dim, label in [
    ("steelblue",  int(obs_idx[0]),   f"dim {int(obs_idx[0])} (obs)"),
    ("darkorange", int(unobs_idx[0]), f"dim {int(unobs_idx[0])} (unobs)"),
] if len(unobs_idx) else [
    ("steelblue", 0, "dim 0 (obs)"),
]:
    p_flat = particles_t[:, :, dim].ravel()
    t_flat = true_t[:, dim].ravel()
    lo_p, hi_p = np.percentile(np.concatenate([p_flat, t_flat]), [0.5, 99.5])
    bins = np.linspace(lo_p, hi_p, 50)
    ax.hist(p_flat, bins=bins, density=True, alpha=0.5, color=color, label=f"particles – {label}")
    ax.hist(t_flat, bins=bins, density=True, alpha=0.5,
            color=color, histtype="step", lw=2, label=f"truth – {label}")
ax.set_xlabel(f"State value at t={t_val:.2f}")
ax.set_ylabel("Density")
ax.set_title(f"Particle distribution at t={t_val:.2f}\nvs ground truth")
ax.legend(fontsize=7)

# Panel B: Marginal distribution of observations vs posterior predictive
ax = axes3[1]
# Posterior predictive: add obs noise to state samples
rng_pred = np.random.default_rng(42)
state_at_obs = all_samps[:, :, :, obs_idx]   # (n_eval, n_p, H, d_y)
if args.obs_noise_type == "gmm":
    w_arr  = np.array(gmm_weights)
    mu_arr = np.array(gmm_offsets) * args.obs_std
    s_arr  = np.array(gmm_stds)    * args.obs_std
    cumw   = np.cumsum(w_arr[:-1])
    u_c    = rng_pred.uniform(size=state_at_obs.shape)
    comp   = (u_c[..., None] >= cumw).sum(axis=-1)
    eps    = rng_pred.standard_normal(state_at_obs.shape)
    pred_noise = mu_arr[comp] + s_arr[comp] * eps
elif args.obs_noise_type == "student_t" and args.obs_df < 1e6:
    a   = args.obs_df / 2.0
    tau = rng_pred.gamma(a, 1.0 / a, size=state_at_obs.shape)
    pred_noise = args.obs_std * rng_pred.standard_normal(state_at_obs.shape) / np.sqrt(tau)
else:
    pred_noise = args.obs_std * rng_pred.standard_normal(state_at_obs.shape)

y_pred = (state_at_obs + pred_noise).ravel()
y_data = y_test_raw[:n_eval].ravel()

lo_m, hi_m = np.percentile(np.concatenate([y_data, y_pred]), [0.5, 99.5])
bins_m = np.linspace(lo_m, hi_m, 70)
ax.hist(y_data, bins=bins_m, density=True, alpha=0.55,
        color="steelblue",  label="Data $y_{\\rm test}$")
ax.hist(y_pred, bins=bins_m, density=True, alpha=0.55,
        color="darkorange", label="Posterior predictive $\\hat{y}$")
ax.set_xlabel("Observation value")
ax.set_ylabel("Density")
ax.set_title("Marginal: data vs posterior predictive")
ax.legend(fontsize=8)

# Panel C: Spread of particle trajectories for one test sequence
ax = axes3[2]
seq_i   = 0
n_show_traj = min(30, args.n_particles)
dim_show    = int(obs_idx[0]) if len(obs_idx) else 0
colors_traj = plt.cm.viridis(np.linspace(0.1, 0.9, n_show_traj))

for pi in range(n_show_traj):
    ax.plot(t_axis, all_samps[seq_i, pi, :, dim_show],
            color=colors_traj[pi], alpha=0.35, lw=0.8)

ax.plot(t_axis, x_true[seq_i, :, dim_show], "k-", lw=2.0, zorder=10, label="Truth")
ax.plot(t_axis, post_mean[seq_i, :, dim_show], "r--", lw=1.5, label="Post. mean")
ax.set_xlabel("Time")
ax.set_ylabel(f"State dim {dim_show}")
ax.set_title(f"Particle trajectories (seq 0, dim {dim_show})\n"
             f"{n_show_traj} of {args.n_particles} particles shown")
ax.legend(fontsize=8)

fig3.tight_layout()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

_dt_tag = f"_dt{args.diffusion_temp:g}" if args.diffusion_temp != 1.0 else ""
_sam_tag = f"_{args.sampler}_seg{args.seg_length}{_dt_tag}"
p1 = out_base + _sam_tag + "_trajectories.png"
p2 = out_base + _sam_tag + "_diagnostics.png"
p3 = out_base + _sam_tag + "_multimodal.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"\nSaved:\n  {p1}\n  {p2}\n  {p3}")
