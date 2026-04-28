"""
Training script for NICEFlowSDE on Lorenz-96 with multi-modal observations.

Model:
    dz_t = f_θ(z_t, c_t, t) dt + G_φ(z_t, c_t, t) dW_t
    c_t  = BiGRUEncoder(y_{1:T})_t   (bidirectional, full-sequence context)
    G_φ  = NICE diffusion: det(G_φ)=1 always, dW_t from VirtualBrownianTree.

Observation noise: Gaussian mixture (multi-modal) | Student-t | Gaussian.

Training modes
--------------
  stepwise  (default): per-step NLL via NICE inverse — no integration needed
  pathwise  (--pathwise): integrate SDE with diffrax, minimise ||ẑ - x||²

Usage
-----
python3 -m bayesinf.applications.state_estimation.train_flow_sde_lorenz96 --train
python3 -m bayesinf.applications.state_estimation.train_flow_sde_lorenz96 --train --pathwise --solver heun
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
    NICEFlowSDE,
    make_step_nice_stepwise,
    make_step_nice_stepwise_ms,
    make_step_nice_pathwise,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    get_solver,
)
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    state_rmse,
    state_mae,
    lorenz96_rhs,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train NICEFlowSDE on Lorenz-96")
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
parser.add_argument("--depth",            type=int,   default=3,
                    help="Drift/diffusion MLP depth")
parser.add_argument("--time_emb_dim",     type=int,   default=32)
# Training
parser.add_argument("--batch_size",       type=int,   default=64)
parser.add_argument("--max_epochs",       type=int,   default=500)
parser.add_argument("--learning_rate",    type=float, default=3e-4)
parser.add_argument("--grad_norm",        type=float, default=1.0)
parser.add_argument("--weight_decay",     type=float, default=1e-4)
parser.add_argument("--diffusion_reg_weight", type=float, default=0.0,
                    help="L2 penalty on NICE diffusion coupling, mean((G_phi-I)^2)")
parser.add_argument("--patience",         type=int,   default=40)
parser.add_argument("--seed",             type=int,   default=0)
# Multiple-shooting training
parser.add_argument("--use_multiple_shooting", action="store_true",
                    help="Enable multiple-shooting training (shorter gradient paths)")
parser.add_argument("--seg_length",       type=int,   default=20,
                    help="Segment length for multiple-shooting (pathwise or stepwise)")
parser.add_argument("--n_nice_layers",    type=int,   default=2,
                    help="Number of NICE coupling layers in G_φ")
parser.add_argument("--pathwise",         action="store_true",
                    help="Train path-wise (integrate SDE, minimise ||ẑ-x||²) "
                         "instead of per-step NLL")
parser.add_argument("--solver",           type=str,   default="heun",
                    choices=("euler", "heun", "reversible_heun", "midpoint", "milstein", "adaptive_euler", "rk4"),
                    help="SDE solver for pathwise training and inference")
parser.add_argument("--use_custom_solver", action="store_true",
                    help="Use custom stochastic solvers instead of diffrax")
# Evaluation
parser.add_argument("--n_particles",      type=int,   default=50,
                    help="Posterior particles used for test RMSE")
parser.add_argument("--sampler",          type=str,   default="heun",
                    choices=("euler", "heun", "reversible_heun", "midpoint", "gt_segmented"),
                    help="Inference sampler / solver")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

results_dir = os.path.join(project_root, "results", "results_state_estimation")
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

_train_mode = ("pw" if args.pathwise else "sw") + args.solver
_diff_reg_tag = (
    f"_dreg{args.diffusion_reg_weight:g}"
    if args.diffusion_reg_weight != 0.0 else ""
)
model_tag = (
    f"NICEFlowSDE"
    f"_N{args.N}_h{args.horizon}_obs{args.obs_interval}"
    f"_{_noise_tag}_std{args.obs_std}"
    f"_dc{args.d_c}_dh{args.d_hidden}_dep{args.depth}"
    f"_nl{args.n_nice_layers}"
    f"_seg{args.seg_length if args.use_multiple_shooting else 'noMS'}"
    f"{_diff_reg_tag}"
    f"_{_train_mode}"
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

d_x = args.N
d_y = args.N // args.obs_interval

k_data, k_model, k_train = jr.split(jr.PRNGKey(args.seed), 3)

print(
    f"Generating Lorenz-96 data  "
    f"(N={args.N}, obs_interval={args.obs_interval}, noise={args.obs_noise_type}) …"
)
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
        "y": jnp.array((d["y"] - y_mean) / y_std),
    }

train_data = _normalize(train_data)
val_data   = _normalize(val_data)
# Keep test_data raw for final metrics; normalize on the fly during sampling.
x_test_raw = np.array(test_data["x"])   # (M_test, H, d_x)  original scale
y_test_norm = jnp.array((test_data["y"] - y_mean) / y_std)
x_test_norm = jnp.array((test_data["x"] - x_mean) / x_std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_F   = args.F_lorenz
_xm  = jnp.array(x_mean)
_xs  = jnp.array(x_std)
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
    key           = k_model,
    physics_fn    = physics_fn,
)
print(
    f"NICEFlowSDE  d_x={d_x}  d_y={d_y}  d_c={args.d_c}  "
    f"d_hidden={args.d_hidden}  depth={args.depth}  n_nice_layers={args.n_nice_layers}\n"
    f"  training={'pathwise' if args.pathwise else 'stepwise'}  solver={args.solver}"
    f"  diffusion_reg_weight={args.diffusion_reg_weight:g}"
)


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


@eqx.filter_jit
def eval_nll(model, x, y):
    return model.nll_stepwise(x, y)


@eqx.filter_jit
def eval_nll_multiple_shooting(model, x, y, seg_length):
    return model.nll_stepwise_ms(x, y, seg_length)


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
        f"  diffusion_reg_weight={args.diffusion_reg_weight:g}"
    )
    if args.use_multiple_shooting:
        print(f"  Multiple-shooting: seg_length={args.seg_length}")
    if args.pathwise:
        print(f"  Path-wise training  solver={args.solver}")

    rng = k_train
    with tqdm(range(args.max_epochs), desc="epoch") as pbar:
        for epoch in pbar:
            rng, k_shuf, k_step = jr.split(rng, 3)
            shuffled     = shuffle_batch(train_data, k_shuf)
            epoch_losses = []

            for b in range(n_batches):
                s, e = b * batch_size, (b + 1) * batch_size
                xb   = shuffled["x"][s:e]   # (B, H, d_x)
                yb   = shuffled["y"][s:e]   # (B, H, d_y)

                if args.pathwise:
                    # Path-wise: integrate SDE per segment, backprop through solver.
                    # seg_length=H-1 (no MS) means one segment = full sequence.
                    seg = args.seg_length if args.use_multiple_shooting else xb.shape[1] - 1
                    model, opt_state, loss = make_step_nice_pathwise(
                        model, xb, yb, k_step, seg, opt_state, optimizer, 
                        args.solver, args.use_custom_solver, args.diffusion_reg_weight
                    )
                elif args.use_multiple_shooting:
                    model, opt_state, loss = make_step_nice_stepwise_ms(
                        model, xb, yb, args.seg_length, opt_state, optimizer,
                        args.diffusion_reg_weight
                    )
                else:
                    model, opt_state, loss = make_step_nice_stepwise(
                        model, xb, yb, opt_state, optimizer,
                        args.diffusion_reg_weight
                    )
                epoch_losses.append(float(loss))

            mean_loss = float(np.mean(epoch_losses))
            
            # Validation loss
            if args.use_multiple_shooting:
                val_loss = float(eval_nll_multiple_shooting(model, val_data["x"], val_data["y"], args.seg_length))
            else:
                val_loss = float(eval_nll(model, val_data["x"], val_data["y"]))

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
                pat=patience_counter,
            )

            try:
                import wandb
                if getattr(wandb, "run", None) is not None:
                    wandb.log({"loss": mean_loss, "val_loss": val_loss})
            except Exception:
                pass

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Best model saved to {save_path}  (val NLL = {best_val_loss:.4f})")

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
    chunk     = 32
    all_samps = []

    @eqx.filter_jit
    def _sample_chunk(model, y_chunk, x0_chunk, key, n_particles, solver_name):
        B    = y_chunk.shape[0]
        keys = jr.split(key, B)
        return jax.vmap(
            lambda yi, x0i, ki: model.sample_posterior(
                yi, x0i, ki, n_particles, solver_name
            )
        )(y_chunk, x0_chunk, keys)   # (B, n_particles, H, d_x)

    solver_name = args.sampler if args.sampler != "gt_segmented" else args.solver
    k_test      = jr.PRNGKey(999)

    print(
        f"  Sampling {args.n_particles} particles / sequence  "
        f"(solver={solver_name}) …"
    )
    for start in range(0, M_test, chunk):
        end   = min(start + chunk, M_test)
        yb    = y_test_norm[start:end]
        x0b   = x_test_norm[start:end, 0, :]    # normalised first state
        k_c   = jr.fold_in(k_test, start)
        samps = np.array(
            _sample_chunk(inf_model, yb, x0b, k_c, args.n_particles, solver_name)
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

    try:
        import wandb
        if getattr(wandb, "run", None) is not None:
            wandb.log({"test_rmse": rmse_all, "test_mae": mae_all, "test_cov90": coverage})
    except Exception:
        pass
