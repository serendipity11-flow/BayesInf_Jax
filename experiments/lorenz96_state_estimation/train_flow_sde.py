"""
Training script for RealNVPFlowSDE on Lorenz-96 with multi-modal observations.

Model:
    custom solver: dz_t = f_θ(z_t, c_t, t) dt + G_φ(eps_t, z_t, c_t, t) sqrt(dt)
    eps_t ~ N(0, I)
    c_t  = BiGRUEncoder(y_{1:T})_t   (bidirectional, full-sequence context)
    T_φ  = bounded conditional RealNVP transition-noise map.

Observation noise: Gaussian mixture (multi-modal) | Student-t | Gaussian.

Training modes
--------------
  pathwise: integrate trajectories and minimise ||ẑ - x||².
            Segment starts are sampled from a diagonal Gaussian parameterised
            by a GRU applied to each observation segment, with an initial-state
            NLL and optional adjacent-segment continuity loss.

Usage
-----
python3 -m experiments.lorenz96_state_estimation.train_flow_sde --train --use_custom_solver --solver heun
"""

import sys, os, argparse
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPFlowSDE,
    make_step_realnvp_pathwise,
)
from experiments.lorenz96_state_estimation.utils.dataset import (
    lorenz96_rhs,
)
from experiments.lorenz96_state_estimation.dataset_selection import (
    add_dataset_arguments,
    load_selected_dataset,
)
from bayesinf.applications.state_estimation.utils.plots import save_state_estimation_plots


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train RealNVPFlowSDE on Lorenz-96")
add_dataset_arguments(parser)
parser.add_argument("--train",            action="store_true")
parser.add_argument("--test",             action="store_true")
# Dataset
parser.add_argument("--horizon",          type=int,   default=100)
parser.add_argument("--N",                type=int,   default=12)
parser.add_argument("--obs_interval",     type=int,   default=2)
parser.add_argument("--obs_std",          type=float, default=0.5)
parser.add_argument("--obs_noise_type",   type=str,   default="gmm",
                    choices=("gaussian", "student_t", "gmm"))
parser.add_argument("--obs_df",           type=float, default=5.0,
                    help="Student-t degrees of freedom (only for student_t noise)")
parser.add_argument("--gmm_weights",      type=str,   default="0.70,0.15,0.15",
                    help="GMM component weights, comma-separated")
parser.add_argument("--gmm_offsets",      type=str,   default="0.0,2.5,-2.5",
                    help="GMM component offsets in units of obs_std")
parser.add_argument("--gmm_stds",         type=str,   default="0.50,1.50,1.50",
                    help="GMM component stds in units of obs_std")
parser.add_argument("--dt",               type=float, default=0.02)
parser.add_argument("--F_lorenz",         type=float, default=8.0)
parser.add_argument("--spinup_steps",     type=int,   default=500)
parser.add_argument("--n_samples",        type=int,   default=2000)
parser.add_argument("--train_frac",       type=float, default=0.8)
parser.add_argument("--val_frac",         type=float, default=0.1)
# Model
parser.add_argument("--d_c",              type=int,   default=64,
                    help="Context vector dimension (BiGRU output)")
parser.add_argument("--d_hidden_enc",     type=int,   default=64,
                    help="BiGRU hidden state size")
parser.add_argument("--d_hidden",         type=int,   default=128,
                    help="Drift/diffusion MLP hidden width")
parser.add_argument("--d_hidden_diffusion", type=int, default=64,
                    help="NICE/RealNVP coupling MLP hidden width")
parser.add_argument("--depth",            type=int,   default=3,
                    help="Drift/diffusion MLP depth")
parser.add_argument("--time_emb_dim",     type=int,   default=32)
parser.add_argument("--observation_stride", type=int, default=5,
                    help="Observation interval used by drift/diffusion context")
# Training
parser.add_argument("--batch_size",       type=int,   default=64)
parser.add_argument("--max_epochs",       type=int,   default=500)
parser.add_argument("--learning_rate",    type=float, default=3e-4)
parser.add_argument("--grad_norm",        type=float, default=1.0)
parser.add_argument("--weight_decay",     type=float, default=1e-4)
parser.add_argument("--diffusion_reg_weight", type=float, default=0.0,
                    help="Deprecated; kept only so older launch scripts still parse.")
parser.add_argument("--patience",         type=int,   default=40)
parser.add_argument("--seed",             type=int,   default=0)
# Multiple-shooting training
parser.add_argument("--use_multiple_shooting", action="store_true",
                    help="Enable multiple-shooting training (shorter gradient paths)")
parser.add_argument("--seg_length",       type=int,   default=20,
                    help="Segment length for multiple-shooting pathwise training")
parser.add_argument("--seg_length_start", type=int,   default=5,
                    help="Initial segment length for scheduled multiple-shooting")
parser.add_argument("--seg_length_schedule_epochs", type=int, default=100,
                    help="Epochs used to grow seg_length_start to seg_length")
parser.add_argument("--shooting_noise_std", type=float, default=0.0,
                    help="Stddev of normalized-state noise added to non-initial segment starts")
parser.add_argument("--rollout_loss_weight", type=float, default=0.0,
                    help="Weight for auxiliary rollout loss during training")
parser.add_argument("--rollout_loss_type", choices=("mse", "energy_score", "gaussian_nll"),
                    default="energy_score",
                    help="Auxiliary full-horizon rollout objective")
parser.add_argument("--rollout_num_ensemble", type=int, default=4,
                    help="Number of rollout particles for probabilistic rollout losses")
parser.add_argument("--rollout_energy_beta", type=float, default=1.0,
                    help="Power beta for energy score rollout loss")
parser.add_argument("--rollout_uncertainty_weight", type=float, default=0.5,
                    help="Pairwise spread weight for energy score rollout loss")
parser.add_argument("--rollout_every", type=int, default=1,
                    help="Compute auxiliary rollout loss every K optimizer steps")
parser.add_argument("--rollout_warmup_epochs", type=int, default=0,
                    help="Train without auxiliary rollout loss for this many initial epochs")
parser.add_argument("--rollout_window", type=int, default=0,
                    help="Random contiguous rollout-loss window length; 0 uses full horizon")
parser.add_argument("--transition_loss_weight", type=float, default=0.0,
                    help="Weight of exact one-step RealNVP transition NLL, including Jacobian.")
parser.add_argument("--exclude_control_from_diffusion", action="store_true",
                    help="Mask control channels only for the normalizing-flow diffusion context.")
parser.add_argument("--experiment_tag", type=str, default="",
                    help="Optional checkpoint suffix for settings not otherwise encoded in the model name.")
parser.add_argument("--kl_weight",        type=float, default=1e-3,
                    help="Deprecated; retained for launch-script compatibility.")
parser.add_argument("--continuity_loss_weight", type=float, default=0.1,
                    help="Weight for adjacent-segment continuity loss")
parser.add_argument("--init_loss_weight", type=float, default=1.0,
                    help="Weight for Gaussian NLL of each segment initial state")
parser.add_argument("--initial_state_mode", choices=("inferred", "provided"),
                    default="inferred",
                    help="Infer segment starts from observations or use known states")
parser.add_argument("--n_realnvp_layers", type=int, default=2,
                    help="Number of affine RealNVP coupling layers")
parser.add_argument("--realnvp_alpha", type=float, default=1.0,
                    help="Bound alpha for log scale = alpha * tanh(raw scale)")
parser.add_argument("--flow_type", choices=("realnvp", "nice"), default="realnvp")
parser.add_argument("--n_nice_layers", type=int, default=2)
parser.add_argument("--nice_noise_scale", type=float, default=0.5,
                    help="Initial value of the learnable NICE global scale")
parser.add_argument("--pathwise",         action="store_true",
                    help="Deprecated; pathwise training is now always used.")
parser.add_argument("--solver",           type=str,   default="heun",
                    choices=("euler", "heun", "reversible_heun", "midpoint", "milstein", "adaptive_euler", "rk4"),
                    help="SDE solver for pathwise training and inference")
parser.add_argument("--use_custom_solver", action="store_true",
                    help="Use custom stochastic solvers instead of diffrax")
# Evaluation
parser.add_argument("--n_particles",      type=int,   default=50,
                    help="Posterior particles used for test RMSE")
parser.add_argument("--sampler",          type=str,   default="heun",
                    choices=("euler", "heun", "reversible_heun", "midpoint",
                             "milstein", "adaptive_euler", "rk4"),
                    help="Inference sampler / solver")
parser.add_argument("--results_dir",      type=str, default=None)
parser.add_argument("--plot_eval",        action="store_true")
parser.add_argument("--eval_chunk_size", type=int, default=32)
args = parser.parse_args()
args.pathwise = True
if args.flow_type == "realnvp":
    # RealNVP applies nonlinear transition noise and therefore cannot be
    # represented by a Diffrax ControlTerm.
    args.use_custom_solver = True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

results_dir = args.results_dir or os.path.join(project_root, "results", "results_state_estimation")
os.makedirs(results_dir, exist_ok=True)

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
_robust_tag = ""
if args.shooting_noise_std != 0.0:
    _robust_tag += f"_sn{args.shooting_noise_std:g}"
if args.rollout_loss_weight != 0.0:
    _robust_tag += f"_rw{args.rollout_loss_weight:g}"
    if args.rollout_loss_type != "mse":
        _robust_tag += (
            f"_{args.rollout_loss_type.replace('_', '')}"
            f"{args.rollout_num_ensemble}"
        )
        if args.rollout_loss_type == "energy_score" and args.rollout_energy_beta != 1.0:
            _robust_tag += f"b{args.rollout_energy_beta:g}"
        if args.rollout_loss_type == "energy_score" and args.rollout_uncertainty_weight != 0.5:
            _robust_tag += f"uw{args.rollout_uncertainty_weight:g}"
    if args.rollout_every > 1:
        _robust_tag += f"_re{args.rollout_every}"
    if args.rollout_warmup_epochs > 0:
        _robust_tag += f"_rwu{args.rollout_warmup_epochs}"
    if args.rollout_window > 0:
        _robust_tag += f"_rwnd{args.rollout_window}"
if args.continuity_loss_weight != 0.0:
    _robust_tag += f"_cw{args.continuity_loss_weight:g}"
if args.transition_loss_weight != 0.0:
    _robust_tag += f"_tnll{args.transition_loss_weight:g}"
if args.exclude_control_from_diffusion:
    _robust_tag += "_nou_nf"
if args.experiment_tag:
    safe_tag = args.experiment_tag.strip().replace(" ", "_").replace("/", "_")
    _robust_tag += f"_{safe_tag}"
if args.initial_state_mode == "inferred" and args.init_loss_weight != 0.0:
    _robust_tag += f"_iw{args.init_loss_weight:g}"
if args.use_multiple_shooting and args.seg_length_start != args.seg_length:
    _robust_tag += f"_sched{args.seg_length_start}to{args.seg_length}"
model_tag = (
    f"{'RealNVP' if args.flow_type == 'realnvp' else 'NICE'}FlowSDE"
    f"_N{args.N}_h{args.horizon}_obs{args.obs_interval}"
    f"_{_noise_tag}_std{args.obs_std}"
    f"_{_train_mode}"
    f"{_robust_tag}"
)
save_path = os.path.join(results_dir, f"{model_tag}.pt")


# ---------------------------------------------------------------------------
# Wandb (optional)
# ---------------------------------------------------------------------------

if args.train and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes"):
    try:
        import wandb
        wandb.init(project="SI_State_Estimation", name=model_tag)
    except Exception as exc:
        wandb = None
        print(f"W&B disabled: {exc}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

k_data, k_model, k_train = jr.split(jr.PRNGKey(args.seed), 3)

train_data, val_data, test_data, d_x, d_y_obs, d_u, data_tag = load_selected_dataset(
    args,
    k_data,
    obs_noise_type=args.obs_noise_type,
    obs_df=args.obs_df if args.obs_noise_type == "student_t" else None,
    gmm_weights=gmm_weights if args.obs_noise_type == "gmm" else None,
    gmm_offsets=gmm_offsets if args.obs_noise_type == "gmm" else None,
    gmm_stds=gmm_stds if args.obs_noise_type == "gmm" else None,
    clean_initial_state=True,
)
d_y = d_y_obs + d_u
model_tag = f"{'RealNVP' if args.flow_type == 'realnvp' else 'NICE'}FlowSDE_{data_tag}_{_train_mode}{_robust_tag}"
save_path = os.path.join(results_dir, f"{model_tag}.pt")

n_train    = train_data["x"].shape[0]
n_val      = val_data["x"].shape[0]
n_test     = test_data["x"].shape[0]
batch_size = min(args.batch_size, n_train)
n_batches  = max(1, n_train // batch_size)

print(
    f"Sequences: train={n_train}  val={n_val}  test={n_test}\n"
    f"Batch: {batch_size}  batches/epoch: {n_batches}"
)


# ---------------------------------------------------------------------------
# Normalization  (stats from training split only)
# ---------------------------------------------------------------------------

x_mean = np.mean(train_data["x"], axis=(0, 1))          # (d_x,)
x_std  = np.std( train_data["x"], axis=(0, 1)).clip(1e-6)
y_mean = np.mean(train_data["y"], axis=(0, 1))          # (d_y,)
y_std  = np.std( train_data["y"], axis=(0, 1)).clip(1e-6)

norm_stats_path = save_path.replace(".pt", "_norm_stats.npz")
np.savez(norm_stats_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print(f"Norm stats: x mean≈{x_mean.mean():.2f} std≈{x_std.mean():.2f} | "
      f"y mean≈{y_mean.mean():.2f} std≈{y_std.mean():.2f}")

def _normalize(d):
    return {
        "x": jnp.array((d["x"] - x_mean) / x_std),
        "y": jnp.concatenate([
            jnp.array((d["y"] - y_mean) / y_std),
            jnp.array(d["u"]),
        ], axis=-1),
        "x0": jnp.array((d["x0"] - x_mean) / x_std),
    }

train_data = _normalize(train_data)
val_data   = _normalize(val_data)
# Keep test_data raw for final metrics; normalize on the fly during sampling.
x_test_raw = np.array(test_data["x"])   # (M_test, H, d_x)  original scale
y_test_norm = jnp.concatenate([
    jnp.array((test_data["y"] - y_mean) / y_std),
    jnp.array(test_data["u"]),
], axis=-1)
x_test_norm = jnp.array((test_data["x"] - x_mean) / x_std)
x0_test_norm = jnp.array((test_data["x0"] - x_mean) / x_std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_F   = args.F_lorenz
_xm  = jnp.array(x_mean)
_xs  = jnp.array(x_std)
physics_fn = (
    (lambda z: lorenz96_rhs(_xm + _xs * z, _F) / _xs)
    if args.dataset == "lorenz96"
    else None
)

model = RealNVPFlowSDE(
    d_x           = d_x,
    d_y           = d_y,
    d_u           = d_u,
    exclude_control_from_diffusion = args.exclude_control_from_diffusion,
    d_c           = args.d_c,
    d_hidden_enc  = args.d_hidden_enc,
    d_hidden      = args.d_hidden,
    d_hidden_diffusion = args.d_hidden_diffusion,
    depth         = args.depth,
    time_emb_dim  = args.time_emb_dim,
    n_realnvp_layers = args.n_realnvp_layers,
    realnvp_alpha = args.realnvp_alpha,
    flow_type      = args.flow_type,
    n_nice_layers  = args.n_nice_layers,
    nice_noise_scale = args.nice_noise_scale,
    observation_stride = args.observation_stride,
    time_step      = args.dt,
    key           = k_model,
    physics_fn    = physics_fn,
)
print(
    f"RealNVPFlowSDE  d_x={d_x}  d_y={d_y}  d_c={args.d_c}  "
    f"d_hidden={args.d_hidden}  depth={args.depth}  "
    f"flow={args.flow_type}  diffusion_hidden={args.d_hidden_diffusion}  layers="
    f"{args.n_realnvp_layers if args.flow_type == 'realnvp' else args.n_nice_layers}\n"
    f"  training=pathwise"
    f"{'+transition_nll' if args.transition_loss_weight else ''}"
    f"  solver={args.solver}  backend={_solver_backend}"
)
if args.exclude_control_from_diffusion:
    print(f"  control routing: deterministic drift only (masked from flow context, d_u={d_u})")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def shuffle_batch(data_dict, key):
    M = data_dict["x"].shape[0]
    perm = jr.permutation(key, M)
    return {
        k: (v[perm] if isinstance(v, jnp.ndarray) and v.shape[0] == M else v)
        for k, v in data_dict.items()
    }


def scheduled_seg_length(epoch: int, horizon: int) -> int:
    if not args.use_multiple_shooting:
        return horizon - 1
    start = max(1, min(args.seg_length_start, horizon - 1))
    end = max(start, min(args.seg_length, horizon - 1))
    sched_epochs = max(1, args.seg_length_schedule_epochs)
    alpha = min(1.0, float(epoch) / float(sched_epochs))
    seg = int(round(start + alpha * (end - start)))
    return max(1, min(seg, horizon - 1))


@eqx.filter_jit
def eval_pathwise(model, x, y, key, seg_length, solver_name, use_custom_solver,
                  rollout_loss_weight, kl_weight, continuity_loss_weight,
                  init_loss_weight, infer_initial, rollout_num_ensemble,
                  rollout_loss_type, rollout_energy_beta,
                  rollout_uncertainty_weight, rollout_window,
                  transition_loss_weight):
    path_loss = model.nll_pathwise(
        x, y, key, seg_length, solver_name, use_custom_solver,
        kl_weight=kl_weight, continuity_weight=continuity_loss_weight,
        init_loss_weight=init_loss_weight, infer_initial=infer_initial,
    )
    total_loss = path_loss
    if transition_loss_weight != 0.0:
        def one_transition_nll(xi, yi):
            drift_context = model.encode_observation_stride(
                yi, model.observation_stride
            )
            diffusion_context = model.encode_diffusion_stride(
                yi, model.observation_stride
            )
            return model.transition_nll(
                xi, drift_context, diffusion_context,
                jnp.asarray(0.0, xi.dtype), model.time_step
            )

        transition_loss = jnp.mean(jax.vmap(one_transition_nll)(x, y))
        total_loss = total_loss + transition_loss_weight * transition_loss
    if rollout_loss_weight != 0.0:
        rollout_loss = model.rollout_loss(
            x, y, jr.fold_in(key, 29), solver_name, use_custom_solver,
            infer_initial, rollout_num_ensemble, rollout_loss_type,
            rollout_energy_beta, rollout_uncertainty_weight, rollout_window
        )
        total_loss = total_loss + rollout_loss_weight * rollout_loss
    return total_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if args.train:
    n_total_steps = args.max_epochs * n_batches
    warmup_steps  = min(500, n_total_steps // 20)

    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, args.learning_rate, warmup_steps),
            optax.cosine_decay_schedule(args.learning_rate, n_total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model       = model
    best_val_loss    = np.inf
    patience_counter = 0

    print(
        f"Training  max_epochs={args.max_epochs}  lr={args.learning_rate}"
        f"  grad_norm={args.grad_norm}  weight_decay={args.weight_decay}"
    )
    if args.use_multiple_shooting:
        print(
            f"  Multiple-shooting: seg_length {args.seg_length_start}"
            f" -> {args.seg_length} over {args.seg_length_schedule_epochs} epochs"
        )
        print(f"  Continuity loss weight: {args.continuity_loss_weight}")
        print(f"  Initial-state loss weight: {args.init_loss_weight}")
    infer_initial = args.initial_state_mode == "inferred"
    if infer_initial:
        print("  Initial state: z0 ~ N(mu_GRU(y_segment), diag(sigma_GRU(y_segment)^2))")
    else:
        print("  Initial state: provided ground-truth segment start (inference network bypassed)")
    if args.shooting_noise_std > 0:
        print(f"  Segment-start noise std: {args.shooting_noise_std}")
    if args.rollout_loss_weight > 0:
        print(
            f"  Auxiliary rollout loss weight: {args.rollout_loss_weight}  "
            f"type={args.rollout_loss_type}  "
            f"ensemble={args.rollout_num_ensemble}  "
            f"uncertainty_weight={args.rollout_uncertainty_weight}  "
            f"every={max(1, args.rollout_every)} step(s)  "
            f"warmup={max(0, args.rollout_warmup_epochs)} epoch(s)  "
            f"window={args.rollout_window if args.rollout_window > 0 else 'full'}"
        )
    if args.transition_loss_weight != 0.0:
        print(
            f"  Exact RealNVP transition NLL weight: "
            f"{args.transition_loss_weight} (base density + flow Jacobian)"
        )
    print(f"  Path-wise training  solver={args.solver}  backend={_solver_backend}")

    rng = k_train
    with tqdm(range(args.max_epochs), desc="epoch") as pbar:
        for epoch in pbar:
            rng, k_shuf, k_step = jr.split(rng, 3)
            shuffled     = shuffle_batch(train_data, k_shuf)
            epoch_losses = []
            seg_epoch = scheduled_seg_length(epoch, train_data["x"].shape[1])

            for b in range(n_batches):
                global_step = epoch * n_batches + b
                s, e = b * batch_size, (b + 1) * batch_size
                xb   = shuffled["x"][s:e]   # (B, H, d_x)
                yb   = shuffled["y"][s:e]   # (B, H, d_y)
                rollout_every = max(1, args.rollout_every)
                rollout_warmup_done = epoch >= max(0, args.rollout_warmup_epochs)
                rollout_weight_step = (
                    args.rollout_loss_weight
                    if (
                        rollout_warmup_done
                        and args.rollout_loss_weight > 0.0
                        and global_step % rollout_every == 0
                    )
                    else 0.0
                )

                # Path-wise: integrate per segment and backprop through solver.
                # seg_length=H-1 (no MS) means one segment = full sequence.
                key_step = jr.fold_in(k_step, global_step)
                model, opt_state, loss = make_step_realnvp_pathwise(
                    model, xb, yb, key_step, seg_epoch, opt_state, optimizer,
                    args.solver, args.use_custom_solver,
                    args.shooting_noise_std, rollout_weight_step,
                    args.kl_weight, args.continuity_loss_weight,
                    args.init_loss_weight, infer_initial,
                    args.rollout_num_ensemble, args.rollout_loss_type,
                    args.rollout_energy_beta, args.rollout_uncertainty_weight,
                    args.rollout_window, args.transition_loss_weight,
                )
                epoch_losses.append(float(loss))

            mean_loss = float(np.mean(epoch_losses))
            
            rng, k_val = jr.split(rng)
            seg = scheduled_seg_length(epoch, val_data["x"].shape[1])
            rollout_weight_eval = (
                args.rollout_loss_weight
                if epoch >= max(0, args.rollout_warmup_epochs)
                else 0.0
            )
            val_loss = float(eval_pathwise(
                model, val_data["x"], val_data["y"], k_val,
                seg, args.solver, args.use_custom_solver,
                rollout_weight_eval, args.kl_weight,
                args.continuity_loss_weight, args.init_loss_weight,
                infer_initial, args.rollout_num_ensemble,
                args.rollout_loss_type, args.rollout_energy_beta,
                args.rollout_uncertainty_weight, args.rollout_window,
                args.transition_loss_weight,
            ))

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model       = model
                patience_counter = 0
                eqx.tree_serialise_leaves(save_path, best_model)
            else:
                patience_counter += 1

            pbar.set_postfix(
                loss=f"{mean_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                seg=seg_epoch,
                pat=patience_counter,
            )

            try:
                import wandb
                if getattr(wandb, "run", None) is not None:
                    wandb.log({"loss": mean_loss, "val_loss": val_loss, "seg_length": seg_epoch})
            except Exception:
                pass

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Best model saved to {save_path}  (val loss = {best_val_loss:.4f})")

else:
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {save_path}\n"
            f"Run with --train first."
        )
    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Loaded checkpoint from {save_path}")


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

if args.test or args.train:
    print("\nEvaluating on test set …")
    inf_model = eqx.nn.inference_mode(model)

    x_test  = test_data["x"]   # (M_test, H, d_x)
    y_test  = test_data["y"]   # (M_test, H, d_y)
    M_test  = x_test.shape[0]
    H       = args.horizon

    # Posterior samples: start from first state of each test trajectory.
    # sample_posterior(y_i, x0_i, key, n_particles) -> (n_particles, H, d_x)
    # We vmap this across the test set in chunks to avoid OOM.
    chunk     = args.eval_chunk_size
    all_samps = []

    @eqx.filter_jit
    def _sample_chunk(model, y_chunk, x_chunk, x0_chunk, key, n_particles,
                      solver_name, use_custom_solver, seg_length):
        B    = y_chunk.shape[0]
        keys = jr.split(key, B)
        if use_custom_solver:
            return jax.vmap(
                lambda yi, x0i, ki: model.sample_posterior_custom_solver(
                    yi, x0i, ki, n_particles, solver_name
                )
            )(y_chunk, x0_chunk, keys)
        return jax.vmap(
            lambda yi, x0i, ki: model.sample_posterior(
                yi, x0i, ki, n_particles, solver_name
            )
        )(y_chunk, x0_chunk, keys)

    solver_name = args.sampler
    k_test      = jr.PRNGKey(999)

    print(
        f"  Sampling {args.n_particles} particles / sequence  "
        f"(solver={solver_name}) …"
    )
    for start in range(0, M_test, chunk):
        end   = min(start + chunk, M_test)
        yb    = y_test_norm[start:end]
        xb    = x_test_norm[start:end]
        k_c   = jr.fold_in(k_test, start)
        samps = np.array(
            _sample_chunk(
                inf_model, yb, xb, x0_test_norm[start:end], k_c, args.n_particles,
                solver_name, args.use_custom_solver, args.seg_length,
            )
        )                                        # (B_chunk, n_p, H, d_x)  normalised
        all_samps.append(samps)

    all_samps   = np.concatenate(all_samps, axis=0)           # (M_test, n_p, H, d_x) norm
    all_samps   = all_samps * x_std[None, None, None] + x_mean[None, None, None]  # denorm
    post_mean   = all_samps.mean(axis=1)                      # (M_test, H, d_x)

    rmse_all    = float(np.mean(np.sqrt(np.mean((x_test_raw - post_mean) ** 2, axis=(1, 2)))))
    mae_all     = float(np.mean(np.abs(x_test_raw - post_mean)))

    # 90 % CI coverage
    lo = np.percentile(all_samps, 5,  axis=1)         # (M_test, H, d_x)
    hi = np.percentile(all_samps, 95, axis=1)
    coverage = float(np.mean((x_test_raw >= lo) & (x_test_raw <= hi)))

    print(f"\n── Test metrics  ({M_test} sequences, {args.n_particles} samples) ──")
    print(f"  RMSE posterior mean: {rmse_all:.4f}")
    print(f"  MAE  posterior mean: {mae_all:.4f}")
    print(f"  Coverage @ 90% CI:   {coverage * 100:.1f}%")
    if args.plot_eval:
        save_state_estimation_plots(
            results_dir,
            "realnvp_flow_sde"
            + (f"_{args.obs_noise_type}" if args.obs_noise_type != "gaussian" else "")
            + ("_nou_nf" if args.exclude_control_from_diffusion else "")
            + (f"_{safe_tag}" if args.experiment_tag else ""),
            x_test_raw,
            np.asarray(test_data["y"]),
            all_samps,
        )

    try:
        import wandb
        if getattr(wandb, "run", None) is not None:
            wandb.log({"test_rmse": rmse_all, "test_mae": mae_all, "test_cov90": coverage})
    except Exception:
        pass
