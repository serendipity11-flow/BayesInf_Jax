"""
Training script for AmortizedStateFlow on Lorenz-96.

Style aligned with bayesinf.models.train_parameter_identification_model:
  --train / --test, argparse at top, results_dir under project root,
  wandb.init when not test, early stopping + save best, tqdm loop.
"""
import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm

from bayesinf.applications.state_estimation.amortized_state_flow import (
    AmortizedStateFlow,
    state_flow_loss,
)
from experiments.lorenz96_state_estimation.utils.dataset import (
    state_rmse,
)
from experiments.lorenz96_state_estimation.dataset_selection import (
    add_dataset_arguments,
    load_selected_dataset,
)
from bayesinf.applications.state_estimation.utils.plots import save_state_estimation_plots

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train AmortizedStateFlow on Lorenz-96")
add_dataset_arguments(parser)
parser.add_argument("--train", action="store_true", help="Run training")
parser.add_argument("--test", action="store_true", help="Load checkpoint and run test eval only")
parser.add_argument("--flow_type", type=str, default="realnvp", choices=("realnvp", "nsf", "maf", "fm", "sbm"))
parser.add_argument("--horizon", type=int, default=100, help="Trajectory window length (dataset); model operates on --seg_len steps")
parser.add_argument("--N", type=int, default=12, help="Lorenz-96 state dimension")
parser.add_argument("--obs_interval", type=int, default=2, help="Observe every obs_interval-th state")
parser.add_argument("--obs_std", type=float, default=0.5, help="Observation noise std")
parser.add_argument("--obs_noise_type", choices=("gaussian", "student_t", "gmm"),
                    default="gaussian")
parser.add_argument("--obs_df", type=float, default=4.0)
parser.add_argument("--gmm_weights", default="0.70,0.15,0.15")
parser.add_argument("--gmm_offsets", default="0.0,2.5,-2.5")
parser.add_argument("--gmm_stds", default="0.50,1.50,1.50")
parser.add_argument("--dt", type=float, default=0.02, help="Integration time step")
parser.add_argument("--F", type=float, default=8.0, help="Lorenz-96 forcing")
parser.add_argument("--spinup_steps", type=int, default=500, help="Spinup steps before slicing windows")
parser.add_argument("--n_samples", type=int, default=2000, help="Total number of windows")
parser.add_argument("--limit", action="store_true", help="Use smaller dataset (n_samples=256) for quick run")
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--grad_norm", type=float, default=1.0, help="Gradient clip (0 = no clip)")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 = disabled)")
parser.add_argument("--n_coupling", type=int, default=8)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--n_hidden", type=int, default=2)
parser.add_argument("--d_enc", type=int, default=64)
parser.add_argument("--d_out", type=int, default=96)
parser.add_argument("--encoder_type", type=str, default="transformer", choices=("gru", "transformer"))
parser.add_argument("--gru_hidden", type=int, default=32)
parser.add_argument("--prior", type=str, default="iid", choices=("wiener_scaled_t", "wiener_increment", "iid"))
parser.add_argument("--seg_len", type=int, default=5,
                    help="Segment each horizon-H window into chunks of seg_len steps. "
                         "Reshapes (M,H,*) -> (M*n_segs, seg_len, *) before the model, "
                         "reducing Transformer attention from O(H^2) to O(seg_len^2) and "
                         "increasing batch parallelism. 0 = disabled.")
parser.add_argument("--segment_stride", type=int, default=0,
                    help="Sliding segment stride; 0 uses non-overlapping segments.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--experiment_tag", type=str, default="",
                    help="Optional checkpoint suffix for distinguishing runs.")
parser.add_argument("--results_dir", type=str, default=None)
parser.add_argument("--plot_eval", action="store_true")
parser.add_argument("--eval_samples", type=int, default=5)
args = parser.parse_args()

train = args.train
test = args.test
flow_type = args.flow_type
horizon = args.horizon
N = args.N
obs_interval = args.obs_interval
obs_std = args.obs_std
dt = args.dt
F = args.F
spinup_steps = args.spinup_steps
n_samples = args.n_samples
if args.limit:
    n_samples = 256
train_frac = args.train_frac
val_frac = args.val_frac
batch_size = args.batch_size
max_epochs = args.max_epochs
learning_rate = args.learning_rate
grad_norm = args.grad_norm
patience = args.patience

# Dataset name for save path and wandb.
# Use seg{seg_len} when segmentation is active (the model's effective horizon),
# else h{horizon} (the model operates on the full window).
_seg = args.seg_len
_seg_label = f"seg{_seg}" if _seg > 0 and horizon > _seg else f"h{horizon}"
data_set = f"lorenz96_N{N}_{_seg_label}_obs{obs_interval}"

# Results dir for state estimation experiments
results_dir = args.results_dir or "/home/zhangyi/bayesinf_jax/results/results_state_estimation"
os.makedirs(results_dir, exist_ok=True)
save_path_pt = os.path.join(results_dir, f"AmortizedStateFlow_{flow_type}_{data_set}.pt")
if args.experiment_tag:
    safe_tag = args.experiment_tag.strip().replace(" ", "_").replace("/", "_")
    save_path_pt = os.path.join(
        results_dir, f"AmortizedStateFlow_{flow_type}_{data_set}_{safe_tag}.pt"
    )

# Wandb (when not test)
if not test and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes"):
    try:
        import wandb
        wandb.init(project="State_Estimation", name=f"AmortizedStateFlow_{flow_type}_{data_set}")
    except Exception as exc:
        print(f"W&B disabled: {exc}")

# -----------------------------------------------------------------------------
# Data: generate once, split train/val/test
# -----------------------------------------------------------------------------
train_key = jr.PRNGKey(args.seed)
k_data, k_model, k_train = jr.split(train_key, 3)
train_data, val_data, test_data, N, D_y, d_u, data_set = load_selected_dataset(
    args, k_data,
    obs_noise_type=args.obs_noise_type,
    obs_df=args.obs_df,
    gmm_weights=tuple(float(v) for v in args.gmm_weights.split(",")),
    gmm_offsets=tuple(float(v) for v in args.gmm_offsets.split(",")),
    gmm_stds=tuple(float(v) for v in args.gmm_stds.split(",")),
)
data_set = data_set.replace(f"_h{horizon}", f"_{_seg_label}")
save_path_pt = os.path.join(results_dir, f"AmortizedStateFlow_{flow_type}_{data_set}.pt")
if args.experiment_tag:
    safe_tag = args.experiment_tag.strip().replace(" ", "_").replace("/", "_")
    save_path_pt = os.path.join(
        results_dir, f"AmortizedStateFlow_{flow_type}_{data_set}_{safe_tag}.pt"
    )
n_train = train_data["x"].shape[0]
n_val = val_data["x"].shape[0]
n_test = test_data["x"].shape[0]
batch_size = min(batch_size, n_train)
n_batches_per_epoch = n_train // batch_size


def shuffle_batch(data_dict, key):
    """Shuffle arrays in data_dict by first dim; return shuffled dict."""
    M = data_dict["x"].shape[0]
    perm = jr.permutation(key, M)
    return {
        k: (v[perm] if isinstance(v, jnp.ndarray) and v.shape[0] == M else v)
        for k, v in data_dict.items()
    }


def segment_batch(x, y, u, x0, seg_len, segment_stride=0):
    """
    Split horizon-H windows into non-overlapping segments of length seg_len.

    Reshapes (M, H, *) -> (M*n_segs, seg_len, *), with x0 for each segment set
    to the true state at the end of the previous segment (or the given x0 for
    the first segment). Steps beyond n_segs*seg_len are dropped.

    Benefits
    --------
    - Transformer: O(H^2) attention -> O(seg_len^2) per segment (n_segs x speedup).
    - GRU: shorter scan -> smaller intermediate tensors, better cache use.
    - Flow: same total M*H tokens; no change in asymptotic cost but improved
      batch parallelism and XLA tile efficiency.

    Parameters
    ----------
    x   : (M, H, D_x)
    y   : (M, H, D_y)
    u   : (M, H, D_u)
    x0  : (M, D_x)
    seg_len : int, must divide H (or H is truncated to the largest multiple)

    Returns
    -------
    x, y, u, x0 all with leading dim M*n_segs and sequence dim seg_len.
    """
    M, H, D_x = x.shape
    D_y = y.shape[-1]
    D_u = u.shape[-1]
    stride = segment_stride if segment_stride > 0 else seg_len
    starts = tuple(range(0, H - seg_len + 1, stride))
    if not starts:
        raise ValueError(f"seg_len={seg_len} exceeds horizon={H}")
    if stride != seg_len:
        x_out = jnp.stack([x[:, s : s + seg_len] for s in starts], axis=1)
        y_out = jnp.stack([y[:, s : s + seg_len] for s in starts], axis=1)
        u_out = jnp.stack([u[:, s : s + seg_len] for s in starts], axis=1)
        x0_out = jnp.stack(
            [x0 if s == 0 else x[:, s - 1] for s in starts], axis=1
        )
        return (
            x_out.reshape(M * len(starts), seg_len, D_x),
            y_out.reshape(M * len(starts), seg_len, D_y),
            u_out.reshape(M * len(starts), seg_len, D_u),
            x0_out.reshape(M * len(starts), D_x),
        )
    n_segs = H // seg_len
    H_eff  = n_segs * seg_len

    x = x[:, :H_eff]
    y = y[:, :H_eff]
    u = u[:, :H_eff]

    # x0 for segment s: true state at end of segment s-1.
    # Boundary indices: seg_len-1, 2*seg_len-1, ..., (n_segs-1)*seg_len-1
    if n_segs > 1:
        boundaries = x[:, seg_len - 1 : H_eff - 1 : seg_len, :]  # (M, n_segs-1, D_x)
        x0_segs = jnp.concatenate([x0[:, None, :], boundaries], axis=1)  # (M, n_segs, D_x)
    else:
        x0_segs = x0[:, None, :]  # (M, 1, D_x)

    # Reshape (M, n_segs, seg_len, *) -> (M*n_segs, seg_len, *)
    x_out  = x.reshape(M, n_segs, seg_len, D_x).reshape(M * n_segs, seg_len, D_x)
    y_out  = y.reshape(M, n_segs, seg_len, D_y).reshape(M * n_segs, seg_len, D_y)
    u_out  = u.reshape(M, n_segs, seg_len, D_u).reshape(M * n_segs, seg_len, D_u)
    x0_out = x0_segs.reshape(M * n_segs, D_x)

    return x_out, y_out, u_out, x0_out


# -----------------------------------------------------------------------------
# Model and loss
# -----------------------------------------------------------------------------
model_key = jr.PRNGKey(42)
model = AmortizedStateFlow(
    d_x=N,
    d_y=D_y,
    d_u=d_u,
    d_enc=args.d_enc,
    d_out=args.d_out,
    key=model_key,
    flow_type=flow_type,
    n_coupling=args.n_coupling,
    hidden_size=args.hidden_size,
    n_hidden=args.n_hidden,
    encoder_type=args.encoder_type,
    gru_hidden=args.gru_hidden,
)

loss_fn = lambda m, x, y, u, x0, k: jnp.mean(state_flow_loss(m, x, y, u, x0, key=k, prior=args.prior))


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
if train:
    if grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_norm),
            optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),
        )
    else:
        optimizer = optax.adam(learning_rate)

    trainable_filter = eqx.is_inexact_array
    opt_state = optimizer.init(eqx.filter(model, trainable_filter))

    # JIT-compiled step: captures optimizer and trainable_filter in closure so
    # that XLA compiles the full forward+backward+update into a single kernel.
    # Defined here (after optimizer is built) so the closure is valid.
    @eqx.filter_jit
    def make_step(model, x, y, u, x0, key, opt_state):
        diff_model, static_model = eqx.partition(model, trainable_filter)
        def _loss(diff_m):
            return loss_fn(eqx.combine(diff_m, static_model), x, y, u, x0, key)
        loss, grads = eqx.filter_value_and_grad(_loss)(diff_model)
        updates, new_opt_state = optimizer.update(grads, opt_state, diff_model)
        new_model = eqx.combine(eqx.apply_updates(diff_model, updates), static_model)
        return new_model, new_opt_state, loss

    @eqx.filter_jit
    def eval_loss(model, x, y, u, x0, key):
        return loss_fn(model, x, y, u, x0, key)

    best_model = model
    best_val_loss = np.inf
    patience_counter = 0

    seg_len = args.seg_len
    use_seg = seg_len > 0 and horizon > seg_len
    if use_seg:
        stride = args.segment_stride if args.segment_stride > 0 else seg_len
        n_segs = (horizon - seg_len) // stride + 1
        print(
            f"Segmentation enabled: horizon={horizon} -> {n_segs} segments of "
            f"{seg_len} steps (stride={stride}, effective batch size "
            f"{batch_size * n_segs})"
        )

    print(f"Starting AmortizedStateFlow ({flow_type}) training on {data_set}. Max epochs: {max_epochs}")
    print(f"Train: {n_train}, val: {n_val}, test: {n_test}, batch_size: {batch_size}, batches/epoch: {n_batches_per_epoch}")

    # Pre-segment validation data once (avoids repeated reshape each epoch)
    if use_seg:
        val_x, val_y, val_u, val_x0 = segment_batch(
            val_data["x"], val_data["y"], val_data["u"], val_data["x0"],
            seg_len, args.segment_stride
        )
    else:
        val_x, val_y, val_u, val_x0 = val_data["x"], val_data["y"], val_data["u"], val_data["x0"]

    rng = train_key
    with tqdm(range(max_epochs), desc="epoch") as pstep:
        for i in pstep:
            rng, k_shuffle, k_val, k_batch = jr.split(rng, 4)
            shuffled = shuffle_batch(train_data, k_shuffle)
            epoch_losses = []
            for b in range(n_batches_per_epoch):
                start = b * batch_size
                end = start + batch_size
                xb  = shuffled["x"][start:end]
                yb  = shuffled["y"][start:end]
                ub  = shuffled["u"][start:end]
                x0b = shuffled["x0"][start:end]
                if use_seg:
                    xb, yb, ub, x0b = segment_batch(
                        xb, yb, ub, x0b, seg_len, args.segment_stride
                    )
                k_b = jr.fold_in(k_batch, b)
                model, opt_state, loss = make_step(
                    model, xb, yb, ub, x0b, k_b, opt_state
                )
                epoch_losses.append(float(loss))
            mean_loss = np.mean(epoch_losses)

            val_loss = float(eval_loss(model, val_x, val_y, val_u, val_x0, k_val))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                eqx.tree_serialise_leaves(save_path_pt, best_model)
                patience_counter = 0
            else:
                patience_counter += 1
            pstep.set_postfix(loss=mean_loss, val_loss=val_loss, best=best_val_loss, patience=patience_counter)
            if os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes"):
                try:
                    import wandb
                    wandb.log({"loss": mean_loss, "val_loss": val_loss})
                except Exception:
                    pass
            if patience > 0 and patience_counter >= patience:
                print(f"\nEarly stopping at epoch {i}")
                break

    model = eqx.tree_deserialise_leaves(save_path_pt, model)
    print(f"Best model saved to {save_path_pt}")

# -----------------------------------------------------------------------------
# Test / eval: load checkpoint and compute test RMSE
# -----------------------------------------------------------------------------
else:
    model = eqx.tree_deserialise_leaves(save_path_pt, model)

if n_test > 0:
    inference_model = eqx.nn.inference_mode(model)
    sample_key = jr.PRNGKey(0)
    test_samples = inference_model.sample(
        test_data["y"], test_data["u"], test_data["x0"],
        key=sample_key, num_samples=args.eval_samples, prior=args.prior,
    )
    test_rmse = float(jnp.mean(state_rmse(test_data["x"], test_samples)))
    print(f"Test RMSE (posterior mean vs true): {test_rmse:.4f}")
    if args.plot_eval:
        save_state_estimation_plots(
            results_dir,
            f"amortized_{flow_type}"
            + (f"_{args.obs_noise_type}" if args.obs_noise_type != "gaussian" else "")
            + (f"_{safe_tag}" if args.experiment_tag else ""),
            test_data["x"],
            test_data["y"],
            np.asarray(test_samples).transpose(1, 0, 2, 3),
        )
    if not test:
        if os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes"):
            try:
                import wandb
                wandb.log({"test_rmse": test_rmse})
            except Exception:
                pass
