"""
Training script for LatentSSMFlow on Lorenz-96 (state estimation mode).

Generative model:
    cond_t   = enc(y_{1:T})_t              [full-sequence encoder, includes y_t]
    z_{t+1}  = F z_t + G ε_t, ε_t ~ N(0,I) [linear Gaussian SSM]
    x_t      = flow⁻¹(z_t ; cond)          [conditional flow decoder]

Training objective (exact MLE):
    log p(x_{1:T}) = Σ_t log|det J(x_t; cond_t)| + log p_SSM(z_{1:T})

Sequences are optionally segmented (--seg_len) for computational efficiency,
exactly as in AmortizedStateFlow.  x0 for each segment is set to the true
boundary state so the encoder has a valid anchor.

Observation noise: Student-t(df, 0, obs_std² I) via Gaussian scale mixture.
Only difference from AmortizedStateFlow is the SSM latent prior structure.

Usage
-----
python3 -m bayesinf.applications.state_estimation.train_latent_ssm_flow --train
python3 -m bayesinf.applications.state_estimation.train_latent_ssm_flow --test
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

from bayesinf.applications.state_estimation.latent_ssm_flow import (
    LatentSSMFlow,
    build_latent_ssm_flow,
)
from bayesinf.applications.state_estimation.lorenz96_dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
    state_rmse,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train LatentSSMFlow on Lorenz-96")
parser.add_argument("--train",             action="store_true")
parser.add_argument("--test",              action="store_true")
parser.add_argument("--flow_type",         type=str,   default="realnvp",
                    choices=("realnvp", "nsf", "maf"))
# Dataset
parser.add_argument("--horizon",           type=int,   default=100)
parser.add_argument("--N",                 type=int,   default=12)
parser.add_argument("--obs_interval",      type=int,   default=2)
parser.add_argument("--obs_std",           type=float, default=0.5)
parser.add_argument("--obs_df",            type=float, default=5.0,
                    help="Student-t degrees of freedom for obs noise")
parser.add_argument("--dt",                type=float, default=0.02)
parser.add_argument("--F_lorenz",          type=float, default=8.0)
parser.add_argument("--spinup_steps",      type=int,   default=500)
parser.add_argument("--n_samples",         type=int,   default=2000)
parser.add_argument("--train_frac",        type=float, default=0.8)
parser.add_argument("--val_frac",          type=float, default=0.1)
# Encoder
parser.add_argument("--encoder_type",      type=str,   default="transformer",
                    choices=("gru", "transformer", "pointwise"))
parser.add_argument("--local_window",      action="store_true",
                    help="Limit encoder context to ±1 step: GRU sees [y_{t-1},y_t,y_{t+1}]; "
                         "Transformer applies a ±1 windowed attention mask.")
parser.add_argument("--seg_len",           type=int,   default=5,
                    help="Segment horizon into chunks of seg_len for efficiency "
                         "(0 = no segmentation, use full horizon)")
parser.add_argument("--d_enc",             type=int,   default=64)
parser.add_argument("--d_out",             type=int,   default=96)
parser.add_argument("--gru_hidden",        type=int,   default=64)
parser.add_argument("--d_model",           type=int,   default=64)
parser.add_argument("--n_heads",           type=int,   default=4)
parser.add_argument("--d_ff",              type=int,   default=256)
parser.add_argument("--n_layers",          type=int,   default=3)
# Flow
parser.add_argument("--n_coupling",        type=int,   default=8)
parser.add_argument("--hidden_size",       type=int,   default=128)
parser.add_argument("--n_hidden",          type=int,   default=2)
parser.add_argument("--num_bins",          type=int,   default=8)
# Training
parser.add_argument("--batch_size",        type=int,   default=32)
parser.add_argument("--max_epochs",        type=int,   default=10000)
parser.add_argument("--learning_rate",     type=float, default=5e-4)
parser.add_argument("--grad_norm",         type=float, default=1.0)
parser.add_argument("--patience",          type=int,   default=30)
parser.add_argument("--max_rho",           type=float, default=0.80,
                    help="Spectral radius cap on SSM transition matrix F. "
                         "0.8 gives 0.8^seg_len ≈ 0.33 residual memory per segment, "
                         "better suited to chaotic dynamics than the old default 0.99.")
parser.add_argument("--fix_ssm",           action="store_true",
                    help="Fix F=max_rho*I and G_chol=ssm_noise*I (no SSM param training). "
                         "Tests whether fixed linear latent dynamics help the flow.")
parser.add_argument("--ssm_noise",         type=float, default=1.0,
                    help="Process noise std for fixed SSM (G_chol = ssm_noise * I). "
                         "1.0 (2× obs_std) widens the SSM prior and improves coverage. "
                         "Only used when --fix_ssm is set.")
parser.add_argument("--seed",              type=int,   default=0)
# Entropy regularization
parser.add_argument("--lambda_ent",        type=float, default=0.0,
                    help="Entropy regularization weight (0 = disabled).")
parser.add_argument("--n_ent_samples",     type=int,   default=4,
                    help="MC samples per sequence used to estimate H[q].")
parser.add_argument("--ent_warmup_frac",   type=float, default=0.3,
                    help="Fraction of total training steps over which lambda_ent "
                         "warms up linearly from 0 to --lambda_ent.")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

results_dir = "/home/zhangyi/bayesinf_jax/results/results_state_estimation"
os.makedirs(results_dir, exist_ok=True)

_seg = args.seg_len
_seg_label = f"seg{_seg}" if _seg > 0 and args.horizon > _seg else f"h{args.horizon}"
data_tag  = (
    f"lorenz96_N{args.N}_{_seg_label}_obs{args.obs_interval}"
    f"_studt{args.obs_df}_std{args.obs_std}"
)
_ssm_tag  = f"_fixedSSM" if args.fix_ssm else ""
_ent_tag  = f"_ent{args.lambda_ent}" if args.lambda_ent > 0 else ""
_enc_tag  = f"_enc{args.encoder_type}" if args.encoder_type != "transformer" else ""
_lw_tag   = "_lw" if args.local_window else ""
model_tag = f"LatentSSMFlow_{args.flow_type}{_ssm_tag}{_enc_tag}{_lw_tag}{_ent_tag}_{data_tag}"
save_path = os.path.join(results_dir, f"{model_tag}.pt")


# ---------------------------------------------------------------------------
# Wandb (optional)
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

d_x = args.N
d_y = args.N // args.obs_interval
d_u = 1

k_data, k_model, k_train = jr.split(jr.PRNGKey(args.seed), 3)

noise_desc = f"Student-t(df={args.obs_df})" if args.obs_df else "Gaussian"
print(
    f"Generating Lorenz-96 data  "
    f"(N={args.N}, obs_interval={args.obs_interval}, noise={noise_desc}) …"
)
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
train_data, val_data, test_data = train_val_test_split(
    data, jr.fold_in(k_data, 1),
    train_frac=args.train_frac, val_frac=args.val_frac,
)

n_train = train_data["x"].shape[0]
n_val   = val_data["x"].shape[0]
n_test  = test_data["x"].shape[0]
batch_size = min(args.batch_size, n_train)
n_batches  = n_train // batch_size

print(
    f"Sequences: train={n_train}  val={n_val}  test={n_test}\n"
    f"Batch: {batch_size}  batches/epoch: {n_batches}"
)


# ---------------------------------------------------------------------------
# Segmentation helper  (same as AmortizedStateFlow)
# ---------------------------------------------------------------------------

def segment_batch(x, y, u, x0, seg_len):
    """
    Split (M, H, *) → (M*n_segs, seg_len, *) with correct x0 per segment.
    x0 for segment s > 0 is the true state at the last step of segment s-1.
    """
    M, H, D_x = x.shape
    D_y = y.shape[-1]
    D_u = u.shape[-1]
    n_segs = H // seg_len
    H_eff  = n_segs * seg_len

    x = x[:, :H_eff]
    y = y[:, :H_eff]
    u = u[:, :H_eff]

    if n_segs > 1:
        boundaries = x[:, seg_len - 1 : H_eff - 1 : seg_len, :]   # (M, n_segs-1, D_x)
        x0_segs = jnp.concatenate([x0[:, None, :], boundaries], axis=1)
    else:
        x0_segs = x0[:, None, :]

    x_out  = x.reshape(M, n_segs, seg_len, D_x).reshape(M * n_segs, seg_len, D_x)
    y_out  = y.reshape(M, n_segs, seg_len, D_y).reshape(M * n_segs, seg_len, D_y)
    u_out  = u.reshape(M, n_segs, seg_len, D_u).reshape(M * n_segs, seg_len, D_u)
    x0_out = x0_segs.reshape(M * n_segs, D_x)
    return x_out, y_out, u_out, x0_out


def shuffle_batch(data_dict, key):
    M = data_dict["x"].shape[0]
    perm = jr.permutation(key, M)
    return {
        k: (v[perm] if hasattr(v, "shape") and v.shape[0] == M else v)
        for k, v in data_dict.items()
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = build_latent_ssm_flow(
    d_x          = d_x,
    d_y          = d_y,
    d_u          = d_u,
    d_enc        = args.d_enc,
    d_out        = args.d_out,
    encoder_type = args.encoder_type,
    local_window = args.local_window,
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
    key          = k_model,
)
print(
    f"Model: LatentSSMFlow  d_x={d_x}  d_y={d_y}  "
    f"flow={args.flow_type}  encoder={args.encoder_type}"
    + ("  local_window=True" if args.local_window else "")
)

# Optionally freeze SSM parameters to fixed simple values.
# F = max_rho * I  (stable scalar decay),  G_chol = ssm_noise * I
# This removes pathological F-growth and tests whether SSM structure alone helps.
if args.fix_ssm:
    fixed_F      = args.max_rho * jnp.eye(d_x)
    fixed_G_chol = args.ssm_noise * jnp.eye(d_x)
    model = eqx.tree_at(
        lambda m: (m.F, m.G_chol),
        model,
        replace=(fixed_F, fixed_G_chol),
    )
    print(
        f"  [fix_ssm] F = {args.max_rho:.2f}*I   "
        f"G_chol = {args.ssm_noise:.3f}*I  (Q diag = {args.ssm_noise**2:.4f})"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

use_seg = args.seg_len > 0 and args.horizon > args.seg_len

# Pre-segment validation data once (outside the training loop)
if use_seg:
    val_x, val_y, val_u, val_x0 = segment_batch(
        val_data["x"], val_data["y"], val_data["u"], val_data["x0"],
        args.seg_len,
    )
else:
    val_x, val_y, val_u, val_x0 = (
        val_data["x"], val_data["y"], val_data["u"], val_data["x0"]
    )

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
    if args.grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_norm),
            optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
        )
    else:
        optimizer = optax.adam(lr_schedule)

    # Build a pytree filter spec (True = trainable, False = frozen).
    # When fix_ssm is True, exclude F and G_chol so they stay at their
    # fixed initial values and receive no gradient updates.
    filter_spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    if args.fix_ssm:
        filter_spec = eqx.tree_at(
            lambda m: (m.F, m.G_chol),
            filter_spec,
            replace=(False, False),
        )

    opt_state = optimizer.init(eqx.filter(model, filter_spec))

    @eqx.filter_jit
    def make_step(model, x, y, u, x0, opt_state, key, lam_ent):
        diff, static = eqx.partition(model, filter_spec)

        def _loss(dm):
            m   = eqx.combine(dm, static)
            nll = m.nll_loss(x, y, u, x0)
            if args.lambda_ent > 0:
                ent = m.entropy_reg_loss(y, u, x0, key, args.n_ent_samples)
                return nll + lam_ent * ent, ent
            return nll, jnp.float32(0.0)

        (total_loss, ent), grads = eqx.filter_value_and_grad(
            _loss, has_aux=True
        )(diff)
        updates, new_opt_state = optimizer.update(grads, opt_state, diff)
        new_model = eqx.combine(eqx.apply_updates(diff, updates), static)
        return new_model, new_opt_state, total_loss, ent

    @eqx.filter_jit
    def eval_loss(model, x, y, u, x0):
        return model.nll_loss(x, y, u, x0)

    best_model       = model
    best_val_loss    = np.inf
    patience_counter = 0

    total_steps   = args.max_epochs * n_batches
    warmup_steps  = int(args.ent_warmup_frac * total_steps)
    global_step   = 0

    print(f"Training  max_epochs={args.max_epochs}  lr={args.learning_rate}")
    if args.lambda_ent > 0:
        print(
            f"  Entropy reg: lambda_ent={args.lambda_ent}"
            f"  n_ent_samples={args.n_ent_samples}"
            f"  warmup_frac={args.ent_warmup_frac}"
            f"  ({warmup_steps} warmup steps)"
        )
    print(f"  F spectral radius init: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(model.F)))):.4f}")
    print(f"  Q diag init: mean={float(jnp.diag(model.Q).mean()):.4f}  max={float(jnp.diag(model.Q).max()):.4f}")

    rng = k_train
    with tqdm(range(args.max_epochs), desc="epoch") as pbar:
        for epoch in pbar:
            rng, k_shuf = jr.split(rng)
            shuffled = shuffle_batch(train_data, k_shuf)
            epoch_losses = []
            epoch_ent    = []

            for b in range(n_batches):
                s, e = b * batch_size, (b + 1) * batch_size
                xb  = shuffled["x"][s:e]
                yb  = shuffled["y"][s:e]
                ub  = shuffled["u"][s:e]
                x0b = shuffled["x0"][s:e]
                if use_seg:
                    xb, yb, ub, x0b = segment_batch(xb, yb, ub, x0b, args.seg_len)

                # Linearly warm up lambda_ent from 0 → args.lambda_ent
                eff_lam = jnp.float32(
                    args.lambda_ent * min(1.0, global_step / max(1, warmup_steps))
                )
                rng, k_step = jr.split(rng)
                model, opt_state, loss, ent = make_step(
                    model, xb, yb, ub, x0b, opt_state, k_step, eff_lam
                )
                global_step  += 1
                epoch_losses.append(float(loss))
                epoch_ent.append(float(ent))

            mean_loss = np.mean(epoch_losses)
            mean_ent  = np.mean(epoch_ent)
            val_loss  = float(eval_loss(model, val_x, val_y, val_u, val_x0))

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model       = model
                patience_counter = 0
                eqx.tree_serialise_leaves(save_path, best_model)
            else:
                patience_counter += 1

            # Diagnostic: print SSM parameter evolution (skip if fixed)
            if not args.fix_ssm and epoch % max(1, args.max_epochs // 5) == 0:
                rho = float(jnp.max(jnp.abs(jnp.linalg.eigvals(model.F))))
                Q_diag = jnp.diag(model.Q)
                pbar.write(
                    f"  [epoch {epoch}] F rho={rho:.4f}  Q=[{float(Q_diag.min()):.3f}..{float(Q_diag.max()):.3f}]"
                )

            postfix = dict(
                loss=f"{mean_loss:.4f}", val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}", pat=patience_counter,
            )
            if args.lambda_ent > 0:
                postfix["ent"] = f"{mean_ent:.4f}"
                postfix["lam"] = f"{float(eff_lam):.3f}"
            pbar.set_postfix(**postfix)

            try:
                import wandb
                log_dict = {"loss": mean_loss, "val_loss": val_loss}
                if args.lambda_ent > 0:
                    log_dict["ent_reg"]   = mean_ent
                    log_dict["lam_ent"]   = float(eff_lam)
                    log_dict["nll"]       = mean_loss - float(eff_lam) * mean_ent
                wandb.log(log_dict)
            except ImportError:
                pass

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Best model saved to {save_path}  (val NLL = {best_val_loss:.4f})")

else:
    model = eqx.tree_deserialise_leaves(save_path, model)
    print(f"Loaded checkpoint from {save_path}")


# ---------------------------------------------------------------------------
# Test evaluation — posterior sample RMSE
# ---------------------------------------------------------------------------

if args.test or args.train:
    print("\nEvaluating on test set …")
    inf_model = eqx.nn.inference_mode(model)

    @eqx.filter_jit
    def test_samples(model, x, y, u, x0, key):
        """Multiple-shooting posterior samples for a batch.

        Each sequence is split into num_patches patches of seg_len steps.
        Patches run in parallel (vmap inside sample_posterior).
        Within each patch:
          - Step 0: true first state is encoded → z_0  (anchor).
          - Steps 1…: SSM propagates from z_0 with noise; decoded autoregressively.

        Returns (M, S, T, d_x).
        """
        keys = jr.split(key, y.shape[0])
        return jax.vmap(
            lambda xi, yi, ui, x0i, ki: model.sample_posterior(
                yi, ui, x0i, ki, n_samples=5, x=xi,
            )
        )(x, y, u, x0, keys)

    samps = np.array(test_samples(
        inf_model,
        test_data["x"], test_data["y"], test_data["u"], test_data["x0"],
        key=jr.PRNGKey(0),
    ))  # (M_test, S, T, d_x)
    # state_rmse expects (S, M, T, d_x) or (M, T, d_x)
    samps_t = samps.transpose(1, 0, 2, 3)   # (S, M_test, T, d_x)
    test_rmse = float(jnp.mean(state_rmse(test_data["x"], jnp.array(samps_t))))
    print(f"Test RMSE (posterior mean vs true): {test_rmse:.4f}")

    try:
        import wandb
        wandb.log({"test_rmse": test_rmse})
    except ImportError:
        pass