"""
Training script for StochasticInterpolantSE on Lorenz-96.

The model refines observation trajectories y_{1:T} into posterior state
trajectories x_{1:T} | y_{1:T} via a flow-time Refinement SDE.

Training modes
--------------
Full-trajectory   (default): one SI loss over the whole T-step sequence.
Multiple-shooting (--ms)    : SI loss over overlapping seg_len-step segments,
                              processed in parallel via vmap. Better gradient
                              flow on chaotic long sequences.

Inference modes
---------------
sde       : full-trajectory Euler-Maruyama SDE
ode       : full-trajectory deterministic ODE
segmented : boundary-pinned segmented SDE — more stable on long Lorenz-96

Usage
-----
python3 -m bayesinf.applications.state_estimation.train_si_lorenz96 --train
python3 -m bayesinf.applications.state_estimation.train_si_lorenz96 --train \\
    --ms --seg_len 20 --obs_noise_type gmm
python3 -m bayesinf.applications.state_estimation.train_si_lorenz96 --test \\
    --sampler segmented --seg_len_eval 20
"""

import sys, os, argparse
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
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

from bayesinf.applications.state_estimation.stochastic_interpolant import (
    StochasticInterpolantSE,
    make_step_si,
    make_step_si_ms,
)
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    lorenz96_rhs,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train StochasticInterpolantSE on Lorenz-96")
parser.add_argument("--train",          action="store_true")
parser.add_argument("--test",           action="store_true")
# Dataset
parser.add_argument("--horizon",        type=int,   default=100)
parser.add_argument("--N",              type=int,   default=12)
parser.add_argument("--obs_interval",   type=int,   default=2)
parser.add_argument("--obs_std",        type=float, default=0.5)
parser.add_argument("--obs_noise_type", type=str,   default="gmm",
                    choices=("gaussian", "student_t", "gmm"))
parser.add_argument("--obs_df",         type=float, default=5.0)
parser.add_argument("--gmm_weights",    type=str,   default="0.70,0.15,0.15")
parser.add_argument("--gmm_offsets",    type=str,   default="0.0,2.5,-2.5")
parser.add_argument("--gmm_stds",       type=str,   default="0.50,1.50,1.50")
parser.add_argument("--dt",             type=float, default=0.02)
parser.add_argument("--F_lorenz",       type=float, default=8.0)
parser.add_argument("--spinup_steps",   type=int,   default=500)
parser.add_argument("--n_samples",      type=int,   default=2000)
parser.add_argument("--train_frac",     type=float, default=0.8)
parser.add_argument("--val_frac",       type=float, default=0.1)
# Model
parser.add_argument("--d_c",            type=int,   default=64)
parser.add_argument("--d_hidden",       type=int,   default=128)
parser.add_argument("--n_heads",        type=int,   default=4,
                    help="Number of attention heads in transformer vel/score nets")
parser.add_argument("--depth",          type=int,   default=3)
parser.add_argument("--time_emb_dim",   type=int,   default=32)
# SI hyperparameters
parser.add_argument("--sigma_0",             type=float, default=1.0)
parser.add_argument("--g0",                  type=float, default=0.5)
parser.add_argument("--score_weight",        type=float, default=0.25)
parser.add_argument("--lifter_loss_weight",  type=float, default=0.1)
parser.add_argument("--score_sigma_floor",   type=float, default=0.05,
                    help="Floor used when converting the trained sigma-scaled score to an actual score during sampling.")
parser.add_argument("--s_min",               type=float, default=0.02)
parser.add_argument("--interpolant_type",    type=str,   default="cosine",
                    choices=("linear", "cosine"),
                    help="Interpolation path between source and target.")
# Training
parser.add_argument("--ms",             action="store_true",
                    help="Multiple-shooting training (more stable on long sequences)")
parser.add_argument("--seg_len",        type=int,   default=20,
                    help="Segment length for multiple-shooting training. "
                         "Best when (horizon-1) %% seg_len == 0.")
parser.add_argument("--batch_size",     type=int,   default=64)
parser.add_argument("--max_epochs",     type=int,   default=500)
parser.add_argument("--learning_rate",  type=float, default=3e-4)
parser.add_argument("--grad_norm",      type=float, default=1.0)
parser.add_argument("--weight_decay",   type=float, default=1e-4)
parser.add_argument("--patience",       type=int,   default=40)
parser.add_argument("--seed",           type=int,   default=0)
parser.add_argument("--wandb",          action="store_true",
                    help="Enable wandb logging. Disabled by default so training works on restricted nodes.")
# Inference
parser.add_argument("--n_particles",    type=int,   default=50)
parser.add_argument("--n_steps",        type=int,   default=100,
                    help="Euler-Maruyama steps in flow time")
parser.add_argument("--sampler",        type=str,   default="sde",
                    choices=("sde", "ode", "segmented"))
parser.add_argument("--seg_len_eval",   type=int,   default=20,
                    help="Segment length for segmented inference")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tuple(s):
    return tuple(float(x) for x in s.split(","))

def _check_seg_len(T, seg_len, name):
    n_segs = (T - 1) // seg_len
    covered = n_segs * seg_len + 1
    if covered < T:
        print(f"  [{name}] seg_len={seg_len}: covers {covered}/{T} steps "
              f"(last {T - covered} steps dropped per sequence)")
    return n_segs


# ---------------------------------------------------------------------------
# Paths and tags
# ---------------------------------------------------------------------------

results_dir = os.path.join(project_root, "results", "results_state_estimation")
os.makedirs(results_dir, exist_ok=True)

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
model_tag  = (
    f"SI"
    f"_N{args.N}_h{args.horizon}_obs{args.obs_interval}"
    f"_{_noise_tag}_std{args.obs_std}"
    f"_dc{args.d_c}_dh{args.d_hidden}_nh{args.n_heads}_dep{args.depth}"
    f"_sig{args.sigma_0}_g{args.g0}_sw{args.score_weight}"
    f"_lw{args.lifter_loss_weight}_sf{args.score_sigma_floor}"
    f"_{_interp_tag}"
    f"_{_train_tag}"
)
save_path = os.path.join(results_dir, f"{model_tag}.pt")


# ---------------------------------------------------------------------------
# Wandb (optional)
# ---------------------------------------------------------------------------

if args.train and args.wandb:
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project="SI_State_Estimation", name=model_tag)
    except Exception as exc:
        print(f"Wandb disabled: {exc}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

d_x = args.N
d_y = args.N // args.obs_interval

k_data, k_model, k_train = jr.split(jr.PRNGKey(args.seed), 3)

print(
    f"Generating Lorenz-96 data  "
    f"(N={args.N}, obs_interval={args.obs_interval}, noise={args.obs_noise_type}) ..."
)
data = generate_lorenz96_dataset(
    k_data,
    n_samples      = args.n_samples,
    horizon        = args.horizon,
    N              = args.N,
    obs_interval   = args.obs_interval,
    obs_std        = args.obs_std,
    obs_noise_type = args.obs_noise_type,
    obs_df         = args.obs_df if args.obs_noise_type == "student_t" else None,
    gmm_weights    = gmm_weights if args.obs_noise_type == "gmm" else None,
    gmm_offsets    = gmm_offsets if args.obs_noise_type == "gmm" else None,
    gmm_stds       = gmm_stds    if args.obs_noise_type == "gmm" else None,
    dt             = args.dt,
    F              = args.F_lorenz,
    spinup_steps   = args.spinup_steps,
    d_u            = 1,
)
train_data, val_data, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=args.train_frac, val_frac=args.val_frac,
)

n_train    = train_data["x"].shape[0]
n_val      = val_data["x"].shape[0]
n_test     = test_data["x"].shape[0]
batch_size = min(args.batch_size, n_train)
n_batches  = max(1, n_train // batch_size)

print(
    f"Sequences: train={n_train}  val={n_val}  test={n_test}\n"
    f"Batch: {batch_size}  batches/epoch: {n_batches}"
)

if args.ms:
    n_segs_train = _check_seg_len(args.horizon, args.seg_len, "train MS")
    print(f"  Multiple-shooting: seg_len={args.seg_len}  "
          f"n_segs={n_segs_train}  "
          f"effective parallelism={batch_size * n_segs_train} segs/step")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

x_mean = np.mean(train_data["x"], axis=(0, 1))
x_std  = np.std( train_data["x"], axis=(0, 1)).clip(1e-6)
y_mean = np.mean(train_data["y"], axis=(0, 1))
y_std  = np.std( train_data["y"], axis=(0, 1)).clip(1e-6)

norm_stats_path = save_path.replace(".pt", "_norm_stats.npz")
np.savez(norm_stats_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print(
    f"Norm stats: x mean≈{x_mean.mean():.2f} std≈{x_std.mean():.2f} | "
    f"y mean≈{y_mean.mean():.2f} std≈{y_std.mean():.2f}"
)

def _normalize(d):
    return {
        "x": jnp.array((d["x"] - x_mean) / x_std),
        "y": jnp.array((d["y"] - y_mean) / y_std),
    }

train_data  = _normalize(train_data)
val_data    = _normalize(val_data)
x_test_raw  = np.array(test_data["x"])
y_test_norm = jnp.array((test_data["y"] - y_mean) / y_std)
x_test_norm = jnp.array((test_data["x"] - x_mean) / x_std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = StochasticInterpolantSE(
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
    key          = k_model,
)
print(
    f"StochasticInterpolantSE  d_x={d_x}  d_y={d_y}  d_c={args.d_c}\n"
    f"  d_hidden={args.d_hidden}  n_heads={args.n_heads}  depth={args.depth}\n"
    f"  sigma_0={args.sigma_0}  g0={args.g0}  score_weight={args.score_weight}\n"
    f"  lifter_loss_weight={args.lifter_loss_weight}  score_sigma_floor={args.score_sigma_floor}\n"
    f"  s_min={args.s_min}  interpolant_type={args.interpolant_type}"
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def shuffle_batch(data_dict, key):
    M    = data_dict["x"].shape[0]
    perm = jr.permutation(key, M)
    return {k: (v[perm] if isinstance(v, jnp.ndarray) and v.shape[0] == M else v)
            for k, v in data_dict.items()}


@eqx.filter_jit
def _eval_loss_full(model, x, y, key):
    return model.loss(x, y, key)

@eqx.filter_jit
def _eval_loss_ms(model, x, y, key, seg_len):
    return model.loss_ms_batch(x, y, key, seg_len)


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

    mode_str = f"multiple-shooting (seg_len={args.seg_len})" if args.ms else "full-trajectory"
    print(
        f"Training [{mode_str}]  "
        f"max_epochs={args.max_epochs}  lr={args.learning_rate}"
    )

    rng = k_train
    with tqdm(range(args.max_epochs), desc="epoch") as pbar:
        for epoch in pbar:
            rng, k_shuf = jr.split(rng)
            shuffled     = shuffle_batch(train_data, k_shuf)
            epoch_losses = []
            epoch_flow   = []
            epoch_score  = []
            epoch_lift   = []

            for b in range(n_batches):
                s, e        = b * batch_size, (b + 1) * batch_size
                xb          = shuffled["x"][s:e]
                yb          = shuffled["y"][s:e]
                rng, k_step = jr.split(rng)

                if args.ms:
                    model, opt_state, loss, components = make_step_si_ms(
                        model, xb, yb, k_step, args.seg_len, opt_state, optimizer
                    )
                else:
                    model, opt_state, loss, components = make_step_si(
                        model, xb, yb, k_step, opt_state, optimizer
                    )
                l_flow, l_score, l_lift = components
                epoch_losses.append(float(loss))
                epoch_flow.append(float(l_flow))
                epoch_score.append(float(l_score))
                epoch_lift.append(float(l_lift))

            mean_loss = float(np.mean(epoch_losses))
            mean_flow = float(np.mean(epoch_flow))
            mean_score = float(np.mean(epoch_score))
            mean_lift = float(np.mean(epoch_lift))

            rng, k_val = jr.split(rng)
            if args.ms:
                val_loss = float(_eval_loss_ms(
                    model, val_data["x"], val_data["y"], k_val, args.seg_len
                ))
            else:
                val_loss = float(_eval_loss_full(
                    model, val_data["x"], val_data["y"], k_val
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
                flow=f"{mean_flow:.4f}",
                score=f"{mean_score:.4f}",
                lift=f"{mean_lift:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                pat=patience_counter,
            )

            if args.wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "loss": mean_loss,
                            "loss_flow": mean_flow,
                            "loss_score": mean_score,
                            "loss_lift": mean_lift,
                            "val_loss": val_loss,
                            "epoch": epoch,
                        })
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
            f"Checkpoint not found: {save_path}\nRun with --train first."
        )
    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Loaded checkpoint from {save_path}")


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

if args.test or args.train:
    print(f"\nEvaluating on test set (sampler={args.sampler}) ...")
    inf_model = eqx.nn.inference_mode(model)

    use_seg = args.sampler == "segmented"
    use_ode = args.sampler == "ode"

    if use_seg:
        n_segs_eval = _check_seg_len(args.horizon, args.seg_len_eval, "eval segmented")
        T_eval = n_segs_eval * args.seg_len_eval + 1
        print(
            f"  Segmented sampler: seg_len={args.seg_len_eval}  "
            f"n_segs={n_segs_eval}  T_eval={T_eval}"
        )
    else:
        T_eval = args.horizon

    @eqx.filter_jit
    def _sample_chunk(model, y_chunk, key, n_particles, n_steps, sampler, seg_len):
        B    = y_chunk.shape[0]
        keys = jr.split(key, B)
        if sampler == "segmented":
            fn = lambda yi, ki: model.sample_posterior_segmented(
                yi, ki, n_particles, n_steps, seg_len
            )
        elif sampler == "ode":
            fn = lambda yi, ki: model.sample_posterior_ode(
                yi, ki, n_particles, n_steps
            )
        else:
            fn = lambda yi, ki: model.sample_posterior(
                yi, ki, n_particles, n_steps
            )
        return jax.vmap(fn)(y_chunk, keys)   # (B, n_p, T', d_x)

    M_test    = x_test_raw.shape[0]
    chunk     = 32
    k_test    = jr.PRNGKey(999)
    all_samps = []

    print(
        f"  Sampling {args.n_particles} particles / sequence  "
        f"(n_steps={args.n_steps}) ..."
    )
    for start in range(0, M_test, chunk):
        end  = min(start + chunk, M_test)
        yb   = y_test_norm[start:end]
        k_c  = jr.fold_in(k_test, start)
        samp = np.array(_sample_chunk(
            inf_model, yb, k_c,
            args.n_particles, args.n_steps, args.sampler, args.seg_len_eval,
        ))   # (B_chunk, n_p, T', d_x) normalised
        all_samps.append(samp)

    all_samps = np.concatenate(all_samps, axis=0)   # (M, n_p, T', d_x) norm
    all_samps = all_samps * x_std[None, None, None] + x_mean[None, None, None]

    # Trim ground truth to the covered time range
    x_ref     = x_test_raw[:, :T_eval, :]
    post_mean = all_samps.mean(axis=1)[:, :T_eval, :]

    rmse_all = float(np.mean(np.sqrt(np.mean((x_ref - post_mean) ** 2, axis=(1, 2)))))
    mae_all  = float(np.mean(np.abs(x_ref - post_mean)))

    lo       = np.percentile(all_samps[:, :, :T_eval, :],  5, axis=1)
    hi       = np.percentile(all_samps[:, :, :T_eval, :], 95, axis=1)
    coverage = float(np.mean((x_ref >= lo) & (x_ref <= hi)))

    print(f"\n── Test metrics  ({M_test} seqs, {args.n_particles} particles, "
          f"T_eval={T_eval}) ──")
    print(f"  RMSE posterior mean: {rmse_all:.4f}")
    print(f"  MAE  posterior mean: {mae_all:.4f}")
    print(f"  Coverage @ 90% CI:   {coverage * 100:.1f}%")

    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"test_rmse": rmse_all, "test_mae": mae_all, "test_cov90": coverage})
        except Exception:
            pass
