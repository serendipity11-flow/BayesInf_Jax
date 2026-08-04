"""
Training script for PointwiseStateFlow on Lorenz-96.

The model uses a stateless MLP encoder (y_t, u_t, x_0 -> cond_t) instead of a
GRU or Transformer, so each timestep is treated independently.

Usage examples
--------------
# Train (NSF flow, default hyperparams):
python3 -m experiments.lorenz96_state_estimation.train_pointwise_state_flow --train

# Train with RealNVP:
python3 -m experiments.lorenz96_state_estimation.train_pointwise_state_flow --train --flow_type realnvp

# Evaluate a saved checkpoint:
python3 -m experiments.lorenz96_state_estimation.train_pointwise_state_flow --test --flow_type nsf
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
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from bayesinf.applications.state_estimation.pointwise_state_flow import PointwiseStateFlow
from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    state_rmse,
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train PointwiseStateFlow on Lorenz-96")
parser.add_argument("--train",       action="store_true")
parser.add_argument("--test",        action="store_true")
parser.add_argument("--flow_type",   type=str,   default="nsf",
                    choices=("realnvp", "nsf", "maf", "fm"))
# Dataset
parser.add_argument("--horizon",      type=int,   default=100)
parser.add_argument("--N",            type=int,   default=12)
parser.add_argument("--obs_interval", type=int,   default=2)
parser.add_argument("--obs_std",      type=float, default=0.5)
parser.add_argument("--dt",           type=float, default=0.02)
parser.add_argument("--F",            type=float, default=8.0)
parser.add_argument("--spinup_steps", type=int,   default=500)
parser.add_argument("--n_samples",    type=int,   default=2000)
parser.add_argument("--limit",        action="store_true")
parser.add_argument("--train_frac",   type=float, default=0.8)
parser.add_argument("--val_frac",     type=float, default=0.1)
# Encoder
parser.add_argument("--encoder_type", type=str,   default="gru",
                    choices=("gru", "transformer"),
                    help="Sequence encoder backend for the observation window")
parser.add_argument("--seg_len",      type=int,   default=5,
                    help="Causal observation window length fed into the encoder")
parser.add_argument("--d_enc",        type=int,   default=64,
                    help="Encoder internal dimension")
parser.add_argument("--d_out",        type=int,   default=96,
                    help="Flow conditioning dimension (encoder output)")
parser.add_argument("--gru_hidden",   type=int,   default=64,
                    help="GRU hidden size")
parser.add_argument("--d_model",      type=int,   default=64,
                    help="Transformer model dim")
parser.add_argument("--n_heads",      type=int,   default=4,
                    help="Transformer attention heads")
parser.add_argument("--d_ff",         type=int,   default=256,
                    help="Transformer FFN hidden dim")
parser.add_argument("--n_layers",     type=int,   default=3,
                    help="Transformer depth")
# Flow
parser.add_argument("--n_coupling",   type=int,   default=8,
                    help="Flow coupling block pairs")
parser.add_argument("--hidden_size",  type=int,   default=128,
                    help="Coupling net hidden width")
parser.add_argument("--n_hidden",     type=int,   default=2,
                    help="Coupling net hidden depth")
parser.add_argument("--num_bins",     type=int,   default=8,
                    help="Spline bins (NSF only)")
# Training
parser.add_argument("--batch_size",   type=int,   default=64)
parser.add_argument("--max_epochs",   type=int,   default=10000)
parser.add_argument("--learning_rate",type=float, default=1e-4)
parser.add_argument("--grad_norm",    type=float, default=1.0)
parser.add_argument("--patience",     type=int,   default=30)
parser.add_argument("--seed",         type=int,   default=0)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
results_dir = "/home/zhangyi/bayesinf_jax/results/results_state_estimation"
os.makedirs(results_dir, exist_ok=True)

_seg_label = (f"seg{args.seg_len}"
              if args.seg_len > 0 and args.horizon > args.seg_len
              else f"h{args.horizon}")
data_set   = f"lorenz96_N{args.N}_{_seg_label}_obs{args.obs_interval}"
if args.obs_std > 0:
    data_set += f"_noise_{args.obs_std}"
model_tag  = f"PointwiseStateFlow_{args.flow_type}_{data_set}"
save_path  = os.path.join(results_dir, f"{model_tag}.pt")

# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------
if not args.test:
    try:
        import wandb
        wandb.init(project="State_Estimation", name=model_tag)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
D_y = args.N // args.obs_interval
d_u = 1
n_samples = 256 if args.limit else args.n_samples

k_data, k_model, k_train = jr.split(jr.PRNGKey(args.seed), 3)
data = generate_lorenz96_dataset(
    k_data,
    n_samples    = n_samples,
    horizon      = args.horizon,
    N            = args.N,
    obs_interval = args.obs_interval,
    obs_std      = args.obs_std,
    dt           = args.dt,
    F            = args.F,
    spinup_steps = args.spinup_steps,
    d_u          = d_u,
)
train_data, val_data, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=args.train_frac, val_frac=args.val_frac,
)
n_train = train_data["x"].shape[0]
n_val   = val_data["x"].shape[0]
n_test  = test_data["x"].shape[0]
batch_size = min(args.batch_size, n_train)
n_batches  = n_train // batch_size

print(f"Dataset: {data_set}")
print(f"Train={n_train}  val={n_val}  test={n_test}  "
      f"batch={batch_size}  batches/epoch={n_batches}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = PointwiseStateFlow(
    d_x          = args.N,
    d_y          = D_y,
    d_u          = d_u,
    d_enc        = args.d_enc,
    d_out        = args.d_out,
    window_size  = args.seg_len,
    encoder_type = args.encoder_type,
    gru_hidden   = args.gru_hidden,
    d_model      = args.d_model,
    n_heads      = args.n_heads,
    d_ff         = args.d_ff,
    n_layers     = args.n_layers,
    flow_type    = args.flow_type,
    n_coupling   = args.n_coupling,
    hidden_size  = args.hidden_size,
    n_hidden     = args.n_hidden,
    num_bins     = args.num_bins,
    key          = k_model,
)
print(f"Encoder: {args.encoder_type}, window_size={args.seg_len}, "
      f"d_enc={args.d_enc}, d_out={args.d_out}")


def loss_fn(m, x, y, u, x0, key=None):
    return m.nll_loss(x, y, u, x0, key=key)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def shuffle_batch(data_dict, key):
    M    = data_dict["x"].shape[0]
    perm = jr.permutation(key, M)
    return {k: (v[perm] if isinstance(v, jnp.ndarray) and v.shape[0] == M else v)
            for k, v in data_dict.items()}


if args.train:
    if args.grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_norm),
            optax.adamw(learning_rate=args.learning_rate, weight_decay=1e-4),
        )
    else:
        optimizer = optax.adam(args.learning_rate)

    trainable_filter = eqx.is_inexact_array
    opt_state = optimizer.init(eqx.filter(model, trainable_filter))

    @eqx.filter_jit
    def make_step(model, x, y, u, x0, key, opt_state):
        diff_model, static_model = eqx.partition(model, trainable_filter)
        def _loss(dm):
            return loss_fn(eqx.combine(dm, static_model), x, y, u, x0, key=key)
        loss, grads = eqx.filter_value_and_grad(_loss)(diff_model)
        updates, new_opt_state = optimizer.update(grads, opt_state, diff_model)
        new_model = eqx.combine(eqx.apply_updates(diff_model, updates), static_model)
        return new_model, new_opt_state, loss

    @eqx.filter_jit
    def eval_loss(model, x, y, u, x0, key):
        return loss_fn(model, x, y, u, x0, key=key)

    best_model       = model
    best_val_loss    = np.inf
    patience_counter = 0

    print(f"Training PointwiseStateFlow ({args.flow_type}), max_epochs={args.max_epochs}")

    rng = k_train
    with tqdm(range(args.max_epochs), desc="epoch") as pbar:
        for epoch in pbar:
            rng, k_shuf, k_val, k_batch = jr.split(rng, 4)
            shuffled    = shuffle_batch(train_data, k_shuf)
            epoch_losses = []
            for b in range(n_batches):
                s, e = b * batch_size, (b + 1) * batch_size
                xb   = shuffled["x"][s:e]
                yb   = shuffled["y"][s:e]
                ub   = shuffled["u"][s:e]
                x0b  = shuffled["x0"][s:e]
                kb   = jr.fold_in(k_batch, b)
                model, opt_state, loss = make_step(model, xb, yb, ub, x0b, kb, opt_state)
                epoch_losses.append(float(loss))
            mean_loss = np.mean(epoch_losses)

            val_loss = float(eval_loss(
                model,
                val_data["x"], val_data["y"], val_data["u"], val_data["x0"],
                key=k_val,
            ))
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model       = model
                patience_counter = 0
                eqx.tree_serialise_leaves(save_path, best_model)
            else:
                patience_counter += 1

            pbar.set_postfix(loss=mean_loss, val=val_loss,
                             best=best_val_loss, pat=patience_counter)
            try:
                import wandb
                wandb.log({"loss": mean_loss, "val_loss": val_loss})
            except ImportError:
                pass

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Best model saved to {save_path}")

else:
    model = eqx.tree_deserialise_leaves(save_path, model)

# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
if n_test > 0:
    inf_model   = eqx.nn.inference_mode(model)
    test_samples = inf_model.sample(
        test_data["y"], test_data["u"], test_data["x0"],
        key=jr.PRNGKey(0), num_samples=5,
    )  # (5, M_test, H, D_x)
    test_rmse = float(jnp.mean(state_rmse(test_data["x"], test_samples)))
    print(f"Test RMSE (posterior mean vs true): {test_rmse:.4f}")
    try:
        import wandb
        wandb.log({"test_rmse": test_rmse})
    except ImportError:
        pass
