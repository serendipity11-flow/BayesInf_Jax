"""
Evaluation and plotting for LatentSSMFlow (state estimation) on Lorenz-96.

Matches the plotting style of AmortizedStateFlow / KalmanFilterFlow:
  - Full T-step trajectory panels: truth / posterior mean / noisy obs / CI
  - Diagnostics: RMSE per dim (bar), coverage per dim (bar), calibration scatter

Usage
-----
python3 -m bayesinf.applications.state_estimation.eval_latent_ssm_flow \\
    --flow_type realnvp --obs_df 5.0

python3 -m bayesinf.applications.state_estimation.eval_latent_ssm_flow \\
    --flow_type realnvp --obs_df 5.0 --n_eval 200
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
from scipy import stats

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.latent_ssm_flow import (
    build_latent_ssm_flow,
)
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    state_rmse,
    state_mae,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate LatentSSMFlow on Lorenz-96")
parser.add_argument("--flow_type",    type=str,   default="realnvp",
                    choices=("realnvp", "nsf", "maf", "fm"))
# Dataset — must match training
parser.add_argument("--horizon",      type=int,   default=100)
parser.add_argument("--N",            type=int,   default=12)
parser.add_argument("--obs_interval", type=int,   default=2)
parser.add_argument("--obs_std",      type=float, default=0.5)
parser.add_argument("--obs_df",       type=float, default=5.0)
parser.add_argument("--dt",           type=float, default=0.02)
parser.add_argument("--F_lorenz",     type=float, default=8.0)
parser.add_argument("--spinup_steps", type=int,   default=500)
parser.add_argument("--n_samples",    type=int,   default=2000)
parser.add_argument("--train_frac",   type=float, default=0.8)
parser.add_argument("--val_frac",     type=float, default=0.1)
# Model — must match training
parser.add_argument("--seg_len",      type=int,   default=5)
parser.add_argument("--d_enc",        type=int,   default=64)
parser.add_argument("--d_out",        type=int,   default=96)
parser.add_argument("--encoder_type", type=str,   default="transformer",
                    choices=("gru", "transformer", "pointwise"))
parser.add_argument("--local_window", action="store_true",
                    help="Must match the --local_window flag used at training time.")
parser.add_argument("--local_window_prev_only", action="store_true",
                    help="Must match the --local_window_prev_only flag used at training time.")
parser.add_argument("--gru_hidden",   type=int,   default=64)
parser.add_argument("--d_model",      type=int,   default=64)
parser.add_argument("--n_heads",      type=int,   default=4)
parser.add_argument("--d_ff",         type=int,   default=256)
parser.add_argument("--n_layers",     type=int,   default=3)
parser.add_argument("--n_coupling",   type=int,   default=8)
parser.add_argument("--hidden_size",  type=int,   default=128)
parser.add_argument("--n_hidden",     type=int,   default=2)
parser.add_argument("--num_bins",     type=int,   default=8)
parser.add_argument("--max_rho",      type=float, default=0.80)
parser.add_argument("--fix_ssm",      action="store_true",
                    help="Must match training flag to resolve correct checkpoint path.")
parser.add_argument("--ssm_noise",    type=float, default=0.5)
parser.add_argument("--lambda_ent",   type=float, default=0.0,
                    help="Must match --lambda_ent used at training time.")
# Eval-specific
parser.add_argument("--seed",         type=int,   default=0)
parser.add_argument("--n_eval",       type=int,   default=100)
parser.add_argument("--n_plot_seqs",  type=int,   default=6)
parser.add_argument("--num_samples",  type=int,   default=100,
                    help="Posterior samples per sequence for CI")
parser.add_argument("--ci",           type=float, default=90.0)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Paths  (must match training script naming)
# ---------------------------------------------------------------------------

results_dir = "/home/zhangyi/bayesinf_jax/results/results_state_estimation"
d_x = args.N
d_y = args.N // args.obs_interval
d_u = 1

_seg_label = f"seg{args.seg_len}" if args.seg_len > 0 and args.horizon > args.seg_len \
             else f"h{args.horizon}"
data_tag  = (
    f"lorenz96_N{args.N}_{_seg_label}_obs{args.obs_interval}"
    f"_studt{args.obs_df}_std{args.obs_std}"
)
_ssm_tag  = "_fixedSSM" if args.fix_ssm else ""
_ent_tag  = f"_ent{args.lambda_ent}" if args.lambda_ent > 0 else ""
_enc_tag  = f"_enc{args.encoder_type}" if args.encoder_type != "transformer" else ""
_lw_tag   = "_lwp" if args.local_window and args.local_window_prev_only else ("_lw" if args.local_window else "")
model_tag = f"LatentSSMFlow_{args.flow_type}{_ssm_tag}{_enc_tag}{_lw_tag}{_ent_tag}_{data_tag}"
ckpt_path = os.path.join(results_dir, f"{model_tag}.pt")

assert os.path.exists(ckpt_path), (
    f"Checkpoint not found: {ckpt_path}\n"
    f"Run training first with matching arguments."
)
out_base = os.path.join(results_dir, f"{model_tag}_eval")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print("Generating test data …")
k_data = jr.PRNGKey(args.seed)
data = generate_lorenz96_dataset(
    k_data,
    n_samples    = args.n_samples,
    horizon      = args.horizon,
    N            = args.N,
    obs_interval = args.obs_interval,
    obs_std      = args.obs_std,
    obs_df       = args.obs_df,
    dt           = args.dt,
    F            = args.F_lorenz,
    spinup_steps = args.spinup_steps,
    d_u          = d_u,
)
_, _, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=args.train_frac, val_frac=args.val_frac,
)

x_test  = np.array(test_data["x"])   # (M_test, T, N)
y_test  = np.array(test_data["y"])   # (M_test, T, d_y)
u_test  = np.array(test_data["u"])
x0_test = np.array(test_data["x0"])

T      = args.horizon
M_test = x_test.shape[0]
n_eval = min(args.n_eval, M_test)
n_plot = min(args.n_plot_seqs, n_eval)

obs_idx   = np.arange(0, args.N, args.obs_interval)
unobs_idx = np.array([i for i in range(args.N) if i not in obs_idx])
t_axis    = np.arange(1, T + 1) * args.dt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print(f"Loading checkpoint: {ckpt_path}")
model = build_latent_ssm_flow(
    d_x          = d_x,
    d_y          = d_y,
    d_u          = d_u,
    d_enc        = args.d_enc,
    d_out        = args.d_out,
    encoder_type = args.encoder_type,
    local_window = args.local_window,
    local_window_prev_only = args.local_window_prev_only,
    gru_hidden   = args.gru_hidden,
    d_model      = args.d_model,
    n_heads      = args.n_heads,
    d_ff         = args.d_ff,
    n_layers     = args.n_layers,
    flow_type    = args.flow_type,
    seg_len      = args.seg_len,
    n_coupling   = args.n_coupling,
    hidden_size  = args.hidden_size,
    n_hidden     = args.n_hidden,
    num_bins     = args.num_bins,
    max_rho      = args.max_rho,
    key          = jr.PRNGKey(42),
)
model = eqx.tree_deserialise_leaves(ckpt_path, model)
model = eqx.nn.inference_mode(model)
print("Checkpoint loaded.")


# ---------------------------------------------------------------------------
# JIT helpers
# ---------------------------------------------------------------------------

@eqx.filter_jit
def run_posterior(model, y, u, x0, x, key, n_samples):
    """
    Multiple-shooting posterior samples for a batch.  Returns (M, n_samples, T, d_x).

    Passes true states x to sample_posterior so each patch is anchored to the
    true first state (encode → z_0, SSM forward, autoregressive decode).
    Boundary states and patch firsts are computed inside sample_posterior.
    """
    keys = jr.split(key, y.shape[0])

    def one_seq(xi, yi, ui, x0i, ki):
        return model.sample_posterior(yi, ui, x0i, ki, n_samples, x=xi)
    return jax.vmap(one_seq)(x, y, u, x0, keys)


# ---------------------------------------------------------------------------
# Reconstruction (deterministic: encode then decode with true x_{t-1})
# ---------------------------------------------------------------------------

@eqx.filter_jit
def reconstruct_all(model, x, y, u, x0):
    def single(xi, yi, ui, x0i):
        if model.flow_type == "fm":
            x_hat = jnp.zeros_like(xi)
            nll = jnp.nan
        else:
            x_hat = model.reconstruct(xi, yi, ui, x0i)
            nll   = -model.log_prob(xi, yi, ui, x0i)
        return x_hat, nll
    return jax.vmap(single)(x, y, u, x0)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"Computing NLL for {n_eval} test sequences …")
_, nlls_arr = reconstruct_all(
    model,
    jnp.array(x_test[:n_eval]), jnp.array(y_test[:n_eval]),
    jnp.array(u_test[:n_eval]), jnp.array(x0_test[:n_eval]),
)
nlls = np.array(nlls_arr)

print(
    f"Drawing {args.num_samples} posterior samples for {n_eval} sequences …"
)
samps_all = np.array(run_posterior(
    model,
    jnp.array(y_test[:n_eval]),
    jnp.array(u_test[:n_eval]),
    jnp.array(x0_test[:n_eval]),
    jnp.array(x_test[:n_eval]),
    key       = jr.PRNGKey(999),
    n_samples = args.num_samples,
))  # (n_eval, S, T, d_x)

post_mean_all = samps_all.mean(axis=1)                                        # (n_eval, T, d_x)
lo_all = np.percentile(samps_all, (100 - args.ci) / 2,       axis=1)
hi_all = np.percentile(samps_all, 100 - (100 - args.ci) / 2, axis=1)

# Slice plot sequences from the full eval set
samps_plot     = samps_all[:n_plot]
post_mean_plot = post_mean_all[:n_plot]
lo_plot        = lo_all[:n_plot]
hi_plot        = hi_all[:n_plot]


# ---------------------------------------------------------------------------
# Metrics  (all from posterior samples — reconstruction RMSE is trivially 0)
# ---------------------------------------------------------------------------

# sample_posterior returns T_eff = (T // seg_len) * seg_len steps, which may
# be shorter than T when T is not divisible by seg_len.  Truncate x_true to
# match before computing metrics.
T_eff = samps_all.shape[2]
x_true_eval = x_test[:n_eval, :T_eff]

rmse_all   = float(np.mean(state_rmse(jnp.array(x_true_eval), jnp.array(post_mean_all))))
rmse_obs   = float(np.sqrt(np.mean((x_true_eval[:, :, obs_idx]   - post_mean_all[:, :, obs_idx])   ** 2)))
rmse_unobs = float(np.sqrt(np.mean((x_true_eval[:, :, unobs_idx] - post_mean_all[:, :, unobs_idx]) ** 2)))
mae_all    = float(np.mean(np.abs(x_true_eval - post_mean_all)))
mean_nll   = float(np.nanmean(nlls)) if not np.all(np.isnan(nlls)) else np.nan

cov_per_dim = np.mean(
    (x_true_eval >= lo_all) & (x_true_eval <= hi_all),
    axis=(0, 1),
)  # (d_x,)
coverage = float(cov_per_dim.mean())

print(f"\n── Evaluation  ({n_eval} sequences, {args.num_samples} posterior samples) ──")
print(f"  RMSE posterior mean (all):       {rmse_all:.4f}")
print(f"  RMSE posterior mean (observed):  {rmse_obs:.4f}")
print(f"  RMSE posterior mean (unobserved):{rmse_unobs:.4f}")
print(f"  MAE  posterior mean (all):       {mae_all:.4f}")
if np.isnan(mean_nll):
    print("  Mean NLL:                        n/a (FM has no exact likelihood)")
else:
    print(f"  Mean NLL:                        {mean_nll:.2f}")
print(f"  Coverage @ {args.ci:.0f}% CI:           {coverage * 100:.1f}%")
print(f"  Noise: Student-t(df={args.obs_df})")


# ---------------------------------------------------------------------------
# Figure 1: trajectory panels
# ---------------------------------------------------------------------------

n_show       = min(4, d_x)
n_cols       = 2
n_rows       = (n_plot + 1) // n_cols
colors_obs   = plt.cm.Blues(np.linspace(0.45, 0.85, max(1, len(obs_idx[:n_show]))))
colors_unobs = plt.cm.Oranges(np.linspace(0.45, 0.85, max(1, len(unobs_idx[:n_show]))))

fig1, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows), sharex=False)
fig1.suptitle(
    f"LatentSSMFlow ({args.flow_type})  –  Lorenz-96 N={args.N}  "
    f"–  Student-t(df={args.obs_df})  –  {args.ci:.0f}% CI\n"
    f"RMSE all={rmse_all:.3f}  obs={rmse_obs:.3f}  unobs={rmse_unobs:.3f}  "
    f"coverage={coverage * 100:.1f}%  |  "
    f"NLL={'n/a' if np.isnan(mean_nll) else f'{mean_nll:.2f}'}",
    fontsize=10,
)

t_axis_eff = t_axis[:T_eff]

for wi, ax in enumerate(np.array(axes).flat):
    if wi >= n_plot:
        ax.set_visible(False)
        continue

    for ci_i, dim in enumerate(obs_idx[:n_show]):
        c = colors_obs[ci_i]
        ax.fill_between(t_axis_eff, lo_plot[wi, :, dim], hi_plot[wi, :, dim],
                        color=c, alpha=0.20)
        ax.plot(t_axis_eff, post_mean_plot[wi, :, dim], color=c, lw=1.5, ls="--",
                label=f"x{dim} (post mean)")
        ax.plot(t_axis_eff, x_test[wi, :T_eff, dim],    color=c, lw=1.2, ls="-")
        oi = int(np.where(obs_idx == dim)[0][0])
        ax.scatter(t_axis_eff, y_test[wi, :T_eff, oi], color=c, s=5, zorder=5,
                   marker="o", alpha=0.6)

    for ci_i, dim in enumerate(unobs_idx[:n_show]):
        c = colors_unobs[ci_i]
        ax.fill_between(t_axis_eff, lo_plot[wi, :, dim], hi_plot[wi, :, dim],
                        color=c, alpha=0.20)
        ax.plot(t_axis_eff, post_mean_plot[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t_axis_eff, x_test[wi, :T_eff, dim],    color=c, lw=1.2, ls="-")

    ax.set_title(f"Test seq {wi}  (seg_len={args.seg_len})", fontsize=9)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")

legend_elems = [
    Line2D([0], [0], color="steelblue",  lw=1.2, ls="-",    label="Truth (observed)"),
    Line2D([0], [0], color="steelblue",  lw=1.5, ls="--",   label="Posterior mean (observed)"),
    Line2D([0], [0], color="steelblue",  marker="o", ls="", ms=5, label="Observation y"),
    Line2D([0], [0], color="darkorange", lw=1.2, ls="-",    label="Truth (unobserved)"),
    Line2D([0], [0], color="darkorange", lw=1.5, ls="--",   label="Posterior mean (unobserved)"),
    Patch(facecolor="grey", alpha=0.3,                       label=f"{args.ci:.0f}% CI"),
]
fig1.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8,
            bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.05, 1, 1])


# ---------------------------------------------------------------------------
# Figure 2: diagnostics
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(
    f"LatentSSMFlow ({args.flow_type}) – {data_tag}  |  "
    f"Student-t(df={args.obs_df})  n_eval={n_eval}",
    fontsize=10,
)
bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(d_x)]

# Panel A: RMSE per dimension (posterior mean)
ax = axes2[0]
rmse_per_dim = np.sqrt(np.mean((x_true_eval - post_mean_all) ** 2, axis=(0, 1)))
ax.bar(np.arange(d_x), rmse_per_dim, color=bar_colors)
ax.axhline(args.obs_std, color="red", ls="--", lw=1.2)
ax.set_xlabel("State dimension")
ax.set_ylabel("RMSE")
ax.set_title("RMSE per state dimension")
ax.legend(handles=[
    Patch(color="steelblue",  label="Observed"),
    Patch(color="darkorange", label="Unobserved"),
    Line2D([0], [0], color="red", ls="--", label=f"obs_std={args.obs_std}"),
], fontsize=8)

# Panel B: empirical coverage per dimension
ax = axes2[1]
ax.bar(np.arange(d_x), cov_per_dim * 100, color=bar_colors)
ax.axhline(args.ci, color="red", ls="--", lw=1.2, label=f"nominal {args.ci:.0f}%")
ax.set_ylim(0, 105)
ax.set_xlabel("State dimension")
ax.set_ylabel("Coverage (%)")
ax.set_title(f"Empirical {args.ci:.0f}% CI coverage per dimension\n({n_eval} sequences)")
ax.legend(fontsize=8)

# Panel C: calibration — posterior std vs |error|
ax = axes2[2]
post_std_plot = samps_plot.std(axis=1)             # (n_plot, T_eff, d_x)
abs_err_plot  = np.abs(x_test[:n_plot, :T_eff] - post_mean_plot)

rng_np = np.random.default_rng(0)
for label, idx, color in [("Observed",   obs_idx,   "steelblue"),
                           ("Unobserved", unobs_idx, "darkorange")]:
    s_f = post_std_plot[:, :, idx].ravel()
    e_f = abs_err_plot[:, :, idx].ravel()
    sel = rng_np.choice(len(s_f), size=min(3000, len(s_f)), replace=False)
    ax.scatter(s_f[sel], e_f[sel], alpha=0.15, s=6, color=color, label=label)

lim = max(post_std_plot.max(), abs_err_plot.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal (y = x)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Posterior std")
ax.set_ylabel("|Error|")
ax.set_title("Calibration: posterior std vs |error|")
ax.legend(fontsize=8)

fig2.tight_layout()


# ---------------------------------------------------------------------------
# Figure 3: Distribution reconstruction vs data distribution
# ---------------------------------------------------------------------------

# ── Posterior predictive at observed dims ──────────────────────────────────
# Posterior samples in state space, projected to obs dims: (n_eval, S, T_eff, d_y)
y_post_samps = samps_all[:, :, :, obs_idx]

# Add Student-t noise to turn state posteriors into *observation* predictives.
# Noise model: y = Hx + obs_std * eps/sqrt(tau),  tau ~ Gamma(df/2, df/2).
rng_pred = np.random.default_rng(42)
if args.obs_df < 1e6:
    _alpha = args.obs_df / 2.0
    _tau   = rng_pred.gamma(_alpha, 1.0 / _alpha, size=y_post_samps.shape)
    _eps   = rng_pred.standard_normal(y_post_samps.shape)
    y_pred_predictive = y_post_samps + args.obs_std * _eps / np.sqrt(_tau)
else:
    y_pred_predictive = y_post_samps + args.obs_std * rng_pred.standard_normal(y_post_samps.shape)

# Flatten for plotting
y_data_flat = y_test[:n_eval, :T_eff, :].ravel()    # noisy observations
y_pred_flat = y_pred_predictive.ravel()              # posterior predictive

# Standardised residuals: (y - x_true_at_obs) / obs_std
x_true_obs  = x_true_eval[:, :, obs_idx]            # (n_eval, T_eff, d_y)
residuals   = (y_test[:n_eval, :T_eff, :] - x_true_obs) / args.obs_std
res_flat    = residuals.ravel()

# ── Figure ─────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
fig3.suptitle(
    f"LatentSSMFlow ({args.flow_type})  –  Observation Distribution Comparison\n"
    f"Student-t(df={args.obs_df}, scale={args.obs_std})  |  n_eval={n_eval}",
    fontsize=10,
)

# Panel A: marginal distribution — data vs posterior predictive
ax = axes3[0]
lo_pct, hi_pct = np.percentile(
    np.concatenate([y_data_flat, y_pred_flat]),
    [0.5, 99.5],
)
bins_marg = np.linspace(lo_pct, hi_pct, 70)
ax.hist(y_data_flat, bins=bins_marg, density=True, alpha=0.55,
        color="steelblue",  label="Data  $y_{\\rm test}$")
ax.hist(y_pred_flat, bins=bins_marg, density=True, alpha=0.55,
        color="darkorange", label="Posterior predictive  $\\hat{y}$")
ax.set_xlabel("Observation value")
ax.set_ylabel("Density")
ax.set_title("Marginal distribution:\ndata vs posterior predictive")
ax.legend(fontsize=8)

# Panel B: standardised residuals vs Student-t PDF
ax = axes3[1]
clip = np.percentile(np.abs(res_flat), 99.5)
res_show = res_flat[np.abs(res_flat) <= clip]
bins_res = np.linspace(-clip, clip, 65)
ax.hist(res_show, bins=bins_res, density=True, alpha=0.6,
        color="steelblue", label="Empirical")

x_pdf = np.linspace(-clip, clip, 400)
ax.plot(x_pdf, stats.t.pdf(x_pdf, df=args.obs_df),  "r-",  lw=2.0,
        label=f"Student-t(df={args.obs_df})")
ax.plot(x_pdf, stats.norm.pdf(x_pdf),               "k--", lw=1.5,
        label="Gaussian N(0,1)")
ax.set_xlabel(r"Standardised residual $(y - x_{\rm true}) / \sigma$")
ax.set_ylabel("Density")
ax.set_title("Residuals vs theoretical\nobservation noise model")
ax.legend(fontsize=8)

# Panel C: Q-Q plot of residuals against Student-t(df)
ax = axes3[2]
rng_qq = np.random.default_rng(7)
n_qq = min(5000, len(res_flat))
res_qq = rng_qq.choice(res_flat, size=n_qq, replace=False)
probs  = (np.arange(1, n_qq + 1) - 0.5) / n_qq
q_theo = stats.t.ppf(probs, df=args.obs_df)
q_emp  = np.sort(res_qq)
ax.scatter(q_theo, q_emp, s=4, alpha=0.3, color="steelblue", label="Q-Q points")
_qq_lim = max(np.abs(q_theo).max(), np.abs(q_emp).max()) * 1.05
ax.plot([-_qq_lim, _qq_lim], [-_qq_lim, _qq_lim], "r--", lw=1.5, label="Ideal (y = x)")
ax.set_xlim(-_qq_lim, _qq_lim)
ax.set_ylim(-_qq_lim, _qq_lim)
ax.set_xlabel(f"Theoretical quantiles  Student-t(df={args.obs_df})")
ax.set_ylabel("Empirical quantiles")
ax.set_title(f"Q-Q plot: residuals vs\nStudent-t(df={args.obs_df})")
ax.legend(fontsize=8)

fig3.tight_layout()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

path1 = out_base + "_trajectories.png"
path2 = out_base + "_diagnostics.png"
path3 = out_base + "_dist_comparison.png"
fig1.savefig(path1, dpi=150, bbox_inches="tight")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
print(f"\nSaved:\n  {path1}\n  {path2}\n  {path3}")
