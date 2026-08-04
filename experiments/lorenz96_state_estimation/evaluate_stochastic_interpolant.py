"""
Evaluation and plotting for StochasticInterpolantSE on Lorenz-96.

Produces three figures:
  1. Trajectory panels: truth / posterior mean / 90% CI / obs dots
  2. Diagnostics:       RMSE per dim | coverage per dim | calibration scatter
  3. GMM analysis:      particle histograms at a fixed t | marginal distributions

Usage
-----
# Must match --train arguments used in train_si_lorenz96.py
python3 -m experiments.lorenz96_state_estimation.evaluate_stochastic_interpolant \\
    --obs_noise_type gmm --N 12 --horizon 100 \\
    --seg_len 5 --sampler segmented
"""

import sys, os, argparse, time, subprocess
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
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

from bayesinf.applications.state_estimation.stochastic_interpolant import StochasticInterpolantSE
from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    lorenz96_rhs,
)


# ---------------------------------------------------------------------------
# Arguments  (must match training script)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate StochasticInterpolantSE on Lorenz-96")
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
parser.add_argument("--d_c",           type=int,   default=128)
parser.add_argument("--d_hidden",      type=int,   default=256)
parser.add_argument("--n_heads",       type=int,   default=4)
parser.add_argument("--depth",         type=int,   default=4)
parser.add_argument("--time_emb_dim",  type=int,   default=32)
parser.add_argument("--sigma_0",            type=float, default=1.0)
parser.add_argument("--g0",                 type=float, default=0.5)
parser.add_argument("--score_weight",       type=float, default=0.25)
parser.add_argument("--lifter_loss_weight", type=float, default=0.1)
parser.add_argument("--score_sigma_floor",  type=float, default=0.05)
parser.add_argument("--s_min",              type=float, default=0.02)
parser.add_argument("--interpolant_type",   type=str,   default="linear",
                    choices=("linear", "cosine"),
                    help="Interpolation path used by the checkpoint.")
# Training tag — must match what was used (--ms + --seg_len)
parser.add_argument("--ms",            action="store_true",
                    help="Was multiple-shooting used during training?")
parser.add_argument("--seg_len",       type=int,   default=5,
                    help="Segment length used during training (for checkpoint lookup)")
# Eval-specific
parser.add_argument("--seed",          type=int,   default=0)
parser.add_argument("--n_eval",        type=int,   default=100)
parser.add_argument("--n_plot_seqs",   type=int,   default=6)
parser.add_argument("--n_particles",   type=int,   default=200)
parser.add_argument("--n_steps",       type=int,   default=100)
parser.add_argument("--ci",            type=float, default=90.0)
parser.add_argument("--sampler",       type=str,   default="segmented",
                    choices=("sde", "ode", "segmented"))
parser.add_argument("--seg_len_eval",  type=int,   default=None,
                    help="Segment length for segmented inference; defaults to --seg_len")
parser.add_argument("--g0_override",   type=float, default=None,
                    help="Post-hoc diffusion strength override (>g0 widens posteriors)")
parser.add_argument("--t_hist",        type=float, default=0.5,
                    help="Normalised time in (0,1) for per-particle histogram")
parser.add_argument("--ckpt_path",     type=str,   default=None,
                    help="Optional explicit checkpoint path; overrides tag-based lookup.")
parser.add_argument("--out_tag",       type=str,   default=None,
                    help="Optional suffix for output figure names when --ckpt_path is used.")
parser.add_argument("--legacy_lifter", action="store_true",
                    help="Load legacy lifter-based SI checkpoints from before q_eta log-sigma heads.")
parser.add_argument("--output_dir",    type=str,   default=None,
                    help="Optional directory for saved figures; defaults to results directory.")
parser.add_argument("--file_ext",      type=str,   default="png",
                    choices=("png", "pdf"),
                    help="Figure file extension.")
parser.add_argument("--no_title",      action="store_true",
                    help="Suppress figure and axes titles.")
parser.add_argument("--font_scale",    type=float, default=1.0,
                    help="Scale Matplotlib font sizes for saved figures.")
args = parser.parse_args()

if args.seg_len_eval is None:
    args.seg_len_eval = args.seg_len


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

_train_tag = f"ms{args.seg_len}" if args.ms else "full"
_interp_tag = "lin" if args.interpolant_type == "linear" else "cos"
model_tag = (
    f"SI"
    f"_N{args.N}_h{args.horizon}_obs{args.obs_interval}"
    f"_{_noise_tag}_std{args.obs_std}"
    f"_dc{args.d_c}_dh{args.d_hidden}_nh{args.n_heads}_dep{args.depth}"
    f"_sig{args.sigma_0}_g{args.g0}_sw{args.score_weight}"
    f"_lw{args.lifter_loss_weight}_sf{args.score_sigma_floor}"
    f"_{_interp_tag}"
    f"_{_train_tag}"
)
ckpt_path = args.ckpt_path or os.path.join(results_dir, f"{model_tag}.pt")
if args.ckpt_path:
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    model_tag = ckpt_stem
    out_base = os.path.join(results_dir, f"{ckpt_stem}_eval")
    if args.out_tag:
        out_base += f"_{args.out_tag}"
else:
    out_base  = os.path.join(results_dir, f"{model_tag}_eval")
if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.join(args.output_dir, os.path.basename(out_base))

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        "Run train_si_lorenz96.py --train first with matching arguments."
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
x_test_raw = np.array(test_data["x"])   # (M_test, H, d_x)
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
x_mean = ns["x_mean"]
x_std  = ns["x_std"]
y_mean = ns["y_mean"]
y_std  = ns["y_std"]
print(f"Norm stats loaded: x mean≈{x_mean.mean():.2f} std≈{x_std.mean():.2f}")

x_test = jnp.array((x_test_raw - x_mean) / x_std)
y_test = jnp.array((y_test_raw - y_mean) / y_std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print(f"Loading checkpoint: {ckpt_path}")
ModelClass = StochasticInterpolantSE
if args.legacy_lifter:
    legacy_src = subprocess.check_output(
        ["git", "show", "5443df1:bayesinf/applications/state_estimation/stochastic_interpolant.py"],
        cwd=project_root,
        text=True,
    )
    legacy_ns = {}
    exec(compile(legacy_src, "legacy_stochastic_interpolant.py", "exec"), legacy_ns)
    ModelClass = legacy_ns["StochasticInterpolantSE"]

model = ModelClass(
    d_x          = d_x,
    d_y          = d_y,
    d_c          = args.d_c,
    d_hidden     = args.d_hidden,
    n_heads      = args.n_heads,
    depth        = args.depth,
    time_emb_dim = args.time_emb_dim,
    sigma_0      = args.sigma_0,
    g0           = args.g0,
    score_weight = args.score_weight,
    lifter_loss_weight = args.lifter_loss_weight,
    score_sigma_floor  = args.score_sigma_floor,
    s_min        = args.s_min,
    interpolant_type = args.interpolant_type,
    key          = jr.PRNGKey(42),
)

def _load_array_leaf(f, x):
    # Older checkpoints differ slightly in which Equinox metadata leaves were
    # serialized. Scalar leaves are metadata; array leaves are the trained
    # weights. Peek around scalar leaves and skip stray serialized scalars when
    # the current tree expects an array.
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

    if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
        return jnp.load(f)
    if isinstance(x, np.ndarray):
        return np.load(f, allow_pickle=True)
    return x

if args.legacy_lifter:
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
else:
    model = eqx.tree_deserialise_leaves(ckpt_path, model, filter_spec=_load_array_leaf)
model = eqx.nn.inference_mode(model)
print("Checkpoint loaded.")
print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")


# ---------------------------------------------------------------------------
# Posterior sampling  (chunked to control memory)
# ---------------------------------------------------------------------------

_n_steps    = args.n_steps
_seg_eval   = args.seg_len_eval
_g0_ov      = args.g0_override
_sampler    = args.sampler
_n_p        = args.n_particles

@eqx.filter_jit
def _sample_chunk(model, y_chunk, key):
    B    = y_chunk.shape[0]
    keys = jr.split(key, B)
    if _sampler == "segmented":
        fn = lambda yi, ki: model.sample_posterior_segmented(
            yi, ki, _n_p, _n_steps, _seg_eval,
            g0_override=_g0_ov,
        )
    elif _sampler == "ode":
        fn = lambda yi, ki: model.sample_posterior_ode(yi, ki, _n_p, _n_steps)
    else:
        fn = lambda yi, ki: model.sample_posterior(
            yi, ki, _n_p, _n_steps, g0_override=_g0_ov,
        )
    return jax.vmap(fn)(y_chunk, keys)   # (B, n_p, T', d_x)


chunk     = 20
k_eval    = jr.PRNGKey(999)
all_samps = []

# Segmented sampler may cover fewer time steps than T
if _sampler == "segmented":
    n_segs_ev = (T - 1) // _seg_eval
    T_eval    = n_segs_ev * _seg_eval + 1
    print(
        f"Segmented sampler: seg_len_eval={_seg_eval}  "
        f"n_segs={n_segs_ev}  T_eval={T_eval}"
    )
else:
    T_eval = T

print(
    f"Drawing {args.n_particles} posterior samples for {n_eval} sequences "
    f"(sampler={args.sampler}  n_steps={args.n_steps}) …"
)
t_sample0 = time.perf_counter()
for start in range(0, n_eval, chunk):
    end  = min(start + chunk, n_eval)
    yb   = y_test[start:end]
    k_c  = jr.fold_in(k_eval, start)
    t0   = time.perf_counter()
    s    = np.array(_sample_chunk(model, yb, k_c))   # (B, n_p, T', d_x) norm
    print(f"  chunk {start:4d}:{end:<4d} sampled in {time.perf_counter() - t0:.1f}s")
    all_samps.append(s)
print(f"Sampling wall time: {time.perf_counter() - t_sample0:.1f}s")

all_samps = np.concatenate(all_samps, axis=0)                              # (n_eval, n_p, T', d_x) norm
all_samps = all_samps * x_std[None, None, None] + x_mean[None, None, None]  # denormalise


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

post_mean  = all_samps.mean(axis=1)             # (n_eval, T', d_x)
lo         = np.percentile(all_samps, (100 - args.ci) / 2,       axis=1)
hi         = np.percentile(all_samps, 100 - (100 - args.ci) / 2, axis=1)

x_true     = x_test_raw[:n_eval, :T_eval, :]   # (n_eval, T', d_x)

rmse_all   = float(np.mean(np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(1, 2)))))
rmse_obs   = float(np.sqrt(np.mean((x_true[:, :, obs_idx]   - post_mean[:, :, obs_idx])   ** 2)))
rmse_unobs = (
    float(np.sqrt(np.mean((x_true[:, :, unobs_idx] - post_mean[:, :, unobs_idx]) ** 2)))
    if len(unobs_idx) else float("nan")
)
mae_all    = float(np.mean(np.abs(x_true - post_mean)))

cov_per_dim = np.mean((x_true >= lo) & (x_true <= hi), axis=(0, 1))   # (d_x,)
coverage    = float(cov_per_dim.mean())

print(f"\n── Evaluation  ({n_eval} seqs, {args.n_particles} particles) ──")
print(f"  RMSE posterior mean (all):        {rmse_all:.4f}")
print(f"  RMSE posterior mean (observed):   {rmse_obs:.4f}")
print(f"  RMSE posterior mean (unobserved): {rmse_unobs:.4f}")
print(f"  MAE  posterior mean:              {mae_all:.4f}")
print(f"  Coverage @ {args.ci:.0f}% CI:            {coverage * 100:.1f}%")

plt.rcParams.update({
    "font.size": 10 * args.font_scale,
    "axes.titlesize": 11 * args.font_scale,
    "axes.labelsize": 11 * args.font_scale,
    "xtick.labelsize": 9 * args.font_scale,
    "ytick.labelsize": 9 * args.font_scale,
    "legend.fontsize": 9 * args.font_scale,
})


# ---------------------------------------------------------------------------
# Figure 1: Trajectory panels
# ---------------------------------------------------------------------------

t_axis_eval = np.arange(T_eval) * args.dt
n_cols  = 2
n_rows  = (n_plot + 1) // n_cols
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows), sharex=False)
if not args.no_title:
    fig1.suptitle(
        f"StochasticInterpolantSE  –  Lorenz-96 N={d_x}  –  noise={args.obs_noise_type}  "
        f"–  {args.ci:.0f}% CI\n"
        f"RMSE={rmse_all:.3f}  obs={rmse_obs:.3f}  unobs={rmse_unobs:.3f}  "
        f"cov={coverage * 100:.1f}%",
        fontsize=10 * args.font_scale,
    )

n_show  = min(4, d_x)
c_obs   = plt.cm.Blues(np.linspace(0.45, 0.85, max(1, len(obs_idx[:n_show]))))
c_unobs = plt.cm.Oranges(np.linspace(0.45, 0.85, max(1, len(unobs_idx[:n_show]))))

for wi, ax in enumerate(np.array(axes1).flat):
    if wi >= n_plot:
        ax.set_visible(False)
        continue
    for ci_i, dim in enumerate(obs_idx[:n_show]):
        c = c_obs[ci_i]
        ax.fill_between(t_axis_eval, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.20)
        ax.plot(t_axis_eval, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t_axis_eval, x_true[wi, :, dim],    color=c, lw=1.2, ls="-")
        oi = int(np.where(obs_idx == dim)[0][0])
        ax.scatter(t_axis_eval, y_test_raw[wi, :T_eval, oi],
                   color=c, s=4, zorder=5, alpha=0.5)
    for ci_i, dim in enumerate(unobs_idx[:n_show]):
        c = c_unobs[ci_i]
        ax.fill_between(t_axis_eval, lo[wi, :, dim], hi[wi, :, dim], color=c, alpha=0.20)
        ax.plot(t_axis_eval, post_mean[wi, :, dim], color=c, lw=1.5, ls="--")
        ax.plot(t_axis_eval, x_true[wi, :, dim],    color=c, lw=1.2, ls="-")
    if not args.no_title:
        ax.set_title(f"Seq {wi}", fontsize=9 * args.font_scale)
    ax.set_xlim(t_axis_eval[0], t_axis_eval[-1])
    ax.margins(x=0)
    ax.set_xlabel("Time", fontsize=9 * args.font_scale)
    ax.set_ylabel("State", fontsize=9 * args.font_scale)
    ax.tick_params(labelsize=9 * args.font_scale)

legend_elems = [
    Line2D([0], [0], color="steelblue",  lw=1.2, ls="-",  label="Truth (obs dim)"),
    Line2D([0], [0], color="steelblue",  lw=1.5, ls="--", label="Post. mean (obs dim)"),
    Line2D([0], [0], color="steelblue",  marker="o", ls="", ms=4, label="Observation y"),
    Line2D([0], [0], color="darkorange", lw=1.2, ls="-",  label="Truth (unobs dim)"),
    Line2D([0], [0], color="darkorange", lw=1.5, ls="--", label="Post. mean (unobs dim)"),
    Patch(facecolor="grey", alpha=0.3, label=f"{args.ci:.0f}% CI"),
]
fig1.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8 * args.font_scale,
            bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.05, 1, 1])


# ---------------------------------------------------------------------------
# Figure 2: Diagnostics
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
if not args.no_title:
    fig2.suptitle(
        f"StochasticInterpolantSE diagnostics — {model_tag}\n"
        f"n_eval={n_eval}  n_particles={args.n_particles}  sampler={args.sampler}",
        fontsize=10 * args.font_scale,
    )
bar_colors = ["steelblue" if i in obs_idx else "darkorange" for i in range(d_x)]

# Panel A: RMSE per dimension
ax = axes2[0]
rmse_per_dim = np.sqrt(np.mean((x_true - post_mean) ** 2, axis=(0, 1)))
ax.bar(np.arange(d_x), rmse_per_dim, color=bar_colors)
ax.axhline(args.obs_std, color="red", ls="--", lw=1.2)
ax.set_xlabel("State dimension")
ax.set_ylabel("RMSE")
if not args.no_title:
    ax.set_title("RMSE per state dim (posterior mean)")
ax.legend(handles=[
    Patch(color="steelblue",  label="Observed"),
    Patch(color="darkorange", label="Unobserved"),
    Line2D([0], [0], color="red", ls="--", label=f"obs_std={args.obs_std}"),
], fontsize=8 * args.font_scale)

# Panel B: coverage per dimension
ax = axes2[1]
ax.bar(np.arange(d_x), cov_per_dim * 100, color=bar_colors)
ax.axhline(args.ci, color="red", ls="--", lw=1.2, label=f"nominal {args.ci:.0f}%")
ax.set_ylim(0, 105)
ax.set_xlabel("State dimension")
ax.set_ylabel("Coverage (%)")
if not args.no_title:
    ax.set_title(f"Empirical {args.ci:.0f}% CI coverage per dim")
ax.legend(fontsize=8 * args.font_scale)

# Panel C: calibration scatter (posterior std vs |error|)
ax = axes2[2]
post_std = all_samps[:n_plot].std(axis=1)        # (n_plot, T', d_x)
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
if not args.no_title:
    ax.set_title("Calibration: posterior std vs |error|")
ax.legend(fontsize=8 * args.font_scale)

fig2.tight_layout()


# ---------------------------------------------------------------------------
# Figure 3: Multi-modal analysis
# ---------------------------------------------------------------------------

t_idx = int(round(args.t_hist * (T_eval - 1)))
t_val = t_idx * args.dt

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
if not args.no_title:
    fig3.suptitle(
        f"StochasticInterpolantSE — Multi-modal analysis\n"
        f"noise={args.obs_noise_type}  obs_std={args.obs_std}  "
        f"n_eval={n_eval}  n_particles={args.n_particles}",
        fontsize=10 * args.font_scale,
    )

# Panel A: Per-particle histogram at t_hist
ax = axes3[0]
particles_t = all_samps[:, :, t_idx, :]   # (n_eval, n_p, d_x)
true_t      = x_true[:, t_idx, :]         # (n_eval, d_x)

dim_pairs = (
    [("steelblue",  int(obs_idx[0]),   f"dim {int(obs_idx[0])} (obs)"),
     ("darkorange", int(unobs_idx[0]), f"dim {int(unobs_idx[0])} (unobs)")]
    if len(unobs_idx) else
    [("steelblue", 0, "dim 0 (obs)")]
)
for color, dim, label in dim_pairs:
    p_flat = particles_t[:, :, dim].ravel()
    t_flat = true_t[:, dim].ravel()
    lo_p, hi_p = np.percentile(np.concatenate([p_flat, t_flat]), [0.5, 99.5])
    bins = np.linspace(lo_p, hi_p, 50)
    ax.hist(p_flat, bins=bins, density=True, alpha=0.5, color=color,
            label=f"particles – {label}")
    ax.hist(t_flat, bins=bins, density=True, alpha=0.5,
            color=color, histtype="step", lw=2, label=f"truth – {label}")
ax.set_xlabel(f"State value at t={t_val:.2f}")
ax.set_ylabel("Density")
if not args.no_title:
    ax.set_title(f"Particle distribution at t={t_val:.2f}\nvs ground truth")
ax.legend(fontsize=7 * args.font_scale)

# Panel B: Marginal distribution of observations vs posterior predictive
ax = axes3[1]
rng_pred     = np.random.default_rng(42)
state_at_obs = all_samps[:, :, :T_eval, obs_idx]   # (n_eval, n_p, T', d_y)
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
y_data = y_test_raw[:n_eval, :T_eval, :].ravel()

lo_m, hi_m = np.percentile(np.concatenate([y_data, y_pred]), [0.5, 99.5])
bins_m = np.linspace(lo_m, hi_m, 70)
ax.hist(y_data, bins=bins_m, density=True, alpha=0.55,
        color="steelblue",  label="Data $y_{\\rm test}$")
ax.hist(y_pred, bins=bins_m, density=True, alpha=0.55,
        color="darkorange", label="Posterior predictive $\\hat{y}$")
ax.set_xlabel("Observation value")
ax.set_ylabel("Density")
if not args.no_title:
    ax.set_title("Marginal: data vs posterior predictive")
ax.legend(fontsize=8 * args.font_scale)

# Panel C: Spread of particle trajectories for one test sequence
ax = axes3[2]
seq_i        = 0
n_show_traj  = min(30, args.n_particles)
dim_show     = int(obs_idx[0]) if len(obs_idx) else 0
colors_traj  = plt.cm.viridis(np.linspace(0.1, 0.9, n_show_traj))

for pi in range(n_show_traj):
    ax.plot(t_axis_eval, all_samps[seq_i, pi, :, dim_show],
            color=colors_traj[pi], alpha=0.35, lw=0.8)

ax.plot(t_axis_eval, x_true[seq_i, :, dim_show], "k-", lw=2.0, zorder=10, label="Truth")
ax.plot(t_axis_eval, post_mean[seq_i, :, dim_show], "r--", lw=1.5, label="Post. mean")
ax.set_xlim(t_axis_eval[0], t_axis_eval[-1])
ax.margins(x=0)
ax.set_xlabel("Time", fontsize=9 * args.font_scale)
ax.set_ylabel(f"State dim {dim_show}", fontsize=9 * args.font_scale)
if not args.no_title:
    ax.set_title(f"Particle trajectories (seq 0, dim {dim_show})\n"
                 f"{n_show_traj} of {args.n_particles} particles shown",
                 fontsize=9 * args.font_scale)
ax.tick_params(labelsize=9 * args.font_scale)
ax.legend(fontsize=8 * args.font_scale)

fig3.tight_layout()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

_g0_tag = f"_g0ov{args.g0_override:g}" if args.g0_override is not None else ""
_eval_tag = f"_sam{args.sampler}_segev{args.seg_len_eval}{_g0_tag}"
ext = args.file_ext

p1 = out_base + _eval_tag + f"_trajectories.{ext}"
p2 = out_base + _eval_tag + f"_diagnostics.{ext}"
p3 = out_base + _eval_tag + f"_multimodal.{ext}"
fig1.savefig(p1, dpi=300, bbox_inches="tight")
fig2.savefig(p2, dpi=300, bbox_inches="tight")
fig3.savefig(p3, dpi=300, bbox_inches="tight")
print(f"\nSaved:\n  {p1}\n  {p2}\n  {p3}")
