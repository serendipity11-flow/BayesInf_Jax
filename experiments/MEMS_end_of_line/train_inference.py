"""Train conditional posterior models on MEMS end-of-line data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.models.networks import FrozenBatchNormEncoder, TimeSeriesEncoder
from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_mems


MODEL_NAMES = ("lstm_bn_flow", "lstm_flow", "transformer_flow", "bnn", "vae",
               "nsf", "fm", "ddpm", "sbm")


def build_model(name: str, param_dim: int, key, width: int, diffusion_steps: int,
                encoder_layers: int = 2, transformer_heads: int = 4,
                flow_dropout: float = 0.0, transformer_model_dim: int | None = None,
                transformer_ff_dim: int | None = None,
                pretrained_lstm_bn_encoder_dir: str | Path | None = None):
    latent, hidden, feature_dim = width, width, 4
    keys = jr.split(key, 2)
    if name == "lstm_bn_flow":
        from bayesinf.models.normalizing_flow import BNNormFlow
        model = BNNormFlow(
            param_dim, feature_dim, keys[0], n_coupling=4,
            hidden_dims=(width,) * 4, encoded_dim=latent,
            encoder_hidden_size=hidden, encoder_layers=encoder_layers,
            clamp=1.0, clamp_activation="I", dropout_rate=flow_dropout)
    elif name in ("lstm_flow", "transformer_flow"):
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(
            param_dim, feature_dim, keys[0], n_coupling=4,
            hidden_nf=width, num_hidden_nf=4,
            encoded_latent_dim=latent,
            hidden_rnn=(transformer_model_dim or hidden)
            if name == "transformer_flow" else hidden,
            flow_type="double_realnvp",
            encoder_type="lstm" if name == "lstm_flow" else "transformer",
            encoder_layers=encoder_layers, sequence_length=96,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            double_hidden_dims=(width,) * 4, clamp=1.0,
            clamp_activation="I", dropout_rate=flow_dropout, use_actnorm=True)
    elif name == "bnn":
        from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator
        model = BNNPosteriorEstimator(feature_dim, param_dim, latent, [width, width], hidden, key=keys[0])
        model = eqx.tree_at(lambda m: m.encoder, model,
                            TimeSeriesEncoder(feature_dim, hidden, latent, keys[1], rnn="lstm"))
    elif name == "vae":
        from bayesinf.models.variational_autoencoder import cVAE
        model = cVAE(feature_dim, param_dim, latent, [width, width], hidden,
                     rnn_type="lstm", key=keys[0])
    elif name == "nsf":
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(param_dim, feature_dim, keys[0], n_coupling=4,
                             hidden_nf=width, num_hidden_nf=4,
                             encoded_latent_dim=latent, hidden_rnn=hidden,
                             rnn_type="lstm", flow_type=name)
    elif name == "fm":
        from bayesinf.models.flow_matching import ConditionalFlowMatching
        model = ConditionalFlowMatching(feature_dim, param_dim, latent, [width, width], hidden, keys[0])
        model = eqx.tree_at(lambda m: m.encoder, model,
                            TimeSeriesEncoder(feature_dim, hidden, latent, keys[1], rnn="lstm"))
    elif name == "ddpm":
        from bayesinf.models.denoising_diffusion_model import ConditionalDDPM
        model = ConditionalDDPM(feature_dim, param_dim, latent, hidden, 16,
                                [width, width], n_steps=diffusion_steps, key=keys[0])
        model = eqx.tree_at(lambda m: m.encoder, model,
                            TimeSeriesEncoder(feature_dim, hidden, latent, keys[1], rnn="lstm"))
    elif name == "sbm":
        from bayesinf.models.score_based_diffusion_model import ConditionalScoreModel
        model = ConditionalScoreModel(feature_dim, param_dim, latent, hidden, 16,
                                      [width, width], key=keys[0])
        model = eqx.tree_at(lambda m: m.encoder, model,
                            TimeSeriesEncoder(feature_dim, hidden, latent, keys[1], rnn="lstm"))
    else:
        raise ValueError(f"Unknown model {name!r}")
    if pretrained_lstm_bn_encoder_dir is not None and name != "lstm_bn_flow":
        from bayesinf.models.normalizing_flow import BNNormFlow
        encoder_dir = Path(pretrained_lstm_bn_encoder_dir)
        source_config = json.loads((encoder_dir / "config.json").read_text())
        source_width = int(source_config.get("width", 64))
        source_layers = int(source_config.get("encoder_layers", 2))
        if source_width != latent:
            raise ValueError(
                "Pretrained encoder output width must match --width: "
                f"{source_width} != {latent}")
        source = BNNormFlow(
            param_dim, feature_dim, keys[1], n_coupling=4,
            hidden_dims=(source_width,) * 4, encoded_dim=source_width,
            encoder_hidden_size=source_width, encoder_layers=source_layers,
            clamp=1.0, clamp_activation="I",
            dropout_rate=float(source_config.get("flow_dropout", 0.0)))
        source = eqx.tree_deserialise_leaves(
            encoder_dir / "lstm_bn_flow.eqx", source)
        state = source.initial_batchnorm_state()
        state = eqx.tree_deserialise_leaves(
            encoder_dir / "lstm_bn_flow.state.eqx", state)
        frozen_encoder = FrozenBatchNormEncoder(source.conditioner, state)
        if name == "vae":
            model = eqx.tree_at(lambda item: item.ts_encoder, model, frozen_encoder)
        else:
            model = eqx.tree_at(lambda item: item.encoder, model, frozen_encoder)
    return model


def loss_and_step(name):
    if name == "bnn":
        from bayesinf.models.bayesian_neural_network import batched_elbo_loss, make_step
        return lambda m, x, y, k: batched_elbo_loss(m, x, y, k, kl_weight=1e-3), make_step
    if name == "vae":
        from bayesinf.models.variational_autoencoder import cvae_loss, make_step
        return lambda m, x, y, k: cvae_loss(m, x, y, k, beta=0.05), make_step
    if name in ("lstm_bn_flow", "lstm_flow", "transformer_flow", "nsf"):
        from bayesinf.models.normalizing_flow import cnf_loss, make_step
        return lambda m, x, y, k: cnf_loss(m, x, y), make_step
    if name == "fm":
        from bayesinf.models.flow_matching import cfm_loss, make_step
        return cfm_loss, make_step
    if name == "ddpm":
        from bayesinf.models.denoising_diffusion_model import make_step
        return lambda m, x, y, k: m.loss(x, y, k), make_step
    from bayesinf.models.score_based_diffusion_model import make_step
    return lambda m, x, y, k: m.loss(x, y, k), make_step


def mems_flow_objective(model, state, signals, parameters, key, l2_coefficient):
    latent, logdet, new_state = model.forward_with_state(
        parameters, signals, state, key=key, inference=False)
    nll = jnp.mean(0.5 * jnp.sum(latent ** 2, axis=-1) - logdet)
    # Match the notebook's small L2 regularizer while leaving fixed ActNorm
    # statistics out through their stop-gradient forward path.
    penalty = sum(jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(model)
                  if eqx.is_inexact_array(leaf) and leaf.ndim > 1)
    return nll + 0.5 * l2_coefficient * penalty, (new_state, nll)


@eqx.filter_jit
def mems_flow_step(model, state, signals, parameters, opt_state, optimizer,
                   key, l2_coefficient):
    (objective, (new_state, nll)), gradients = eqx.filter_value_and_grad(
        mems_flow_objective, has_aux=True
    )(model, state, signals, parameters, key, l2_coefficient)
    updates, opt_state = optimizer.update(
        gradients, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), new_state, opt_state, objective, nll


@eqx.filter_jit
def mems_flow_validation(model, state, signals, parameters):
    latent, logdet, _ = model.forward_with_state(
        parameters, signals, state, inference=True)
    return jnp.mean(0.5 * jnp.sum(latent ** 2, axis=-1) - logdet)


def inference_learning_rate(step, start_lr=1e-4, kernel_gamma=0.1,
                            max_explr_epoch=2000):
    """Exponential schedule, stepped for ``3 * max_explr_epoch``.

    With one full-batch update per epoch this is equivalent to::

        gamma = kernel_gamma ** (1 / max_explr_epoch)
        if epoch < 3 * max_explr_epoch:
            scheduler.step()
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    gamma = jnp.power(kernel_gamma, 1.0 / max_explr_epoch)
    return start_lr * jnp.power(gamma, jnp.minimum(step, 3 * max_explr_epoch))


def lstm_bn_learning_rate(step, *, steps_per_epoch, start_lr=1e-4,
                          lr_decay=0.8, lr_decay_epochs=2000):
    """Notebook scheduler stepped once per epoch and stopped after 3 horizons."""
    step = jnp.asarray(step, dtype=jnp.float32)
    epoch_step = jnp.floor(step / max(1, steps_per_epoch))
    per_epoch_gamma = lr_decay ** (1.0 / lr_decay_epochs)
    maximum_scheduler_steps = 3 * lr_decay_epochs - 1
    scheduler_step = jnp.minimum(epoch_step, maximum_scheduler_steps)
    # Algebraically ``per_epoch_gamma ** scheduler_step``. Evaluating through
    # log(lr_decay) avoids accumulating float32 rounding in gamma near one.
    del per_epoch_gamma
    return start_lr * jnp.exp(
        jnp.log(lr_decay) * scheduler_step / lr_decay_epochs
    )


def transformer_learning_rate(step, *, steps_per_epoch, total_epochs,
                              start_lr=1e-4, hold_epochs=500,
                              min_learning_rate_ratio=0.01):
    """Hold the initial LR, then apply smooth per-epoch cosine decay."""
    step = jnp.asarray(step, dtype=jnp.float32)
    epoch = jnp.floor(step / max(1, steps_per_epoch))
    decay_epochs = max(1, total_epochs - hold_epochs)
    progress = jnp.clip((epoch - hold_epochs) / decay_epochs, 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    multiplier = min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine
    return start_lr * multiplier


def warmup_cosine_learning_rate(step, *, steps_per_epoch, start_lr,
                                warmup_epochs=0, decay_epochs=2000,
                                min_learning_rate_ratio=0.01):
    """Linear warmup followed by cosine decay on an explicit epoch horizon.

    The decay horizon is deliberately independent of ``--epochs``. This makes
    the schedule useful with early stopping: a long maximum training budget no
    longer prevents the optimizer from reaching its refinement learning rate.
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    epoch = step / max(1, steps_per_epoch)
    warmup_epochs = max(0, warmup_epochs)
    decay_epochs = max(1, decay_epochs)
    if warmup_epochs:
        warmup = jnp.minimum(1.0, (epoch + 1.0) / warmup_epochs)
    else:
        warmup = jnp.asarray(1.0, dtype=jnp.float32)
    progress = jnp.clip((epoch - warmup_epochs) / decay_epochs, 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    decay = min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine
    return start_lr * warmup * decay


def train_one(name, data, args, out_dir):
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"{args.wandb_group}-{name}-seed{args.seed}",
            job_type="parameter-identification",
            mode=args.wandb_mode,
            config={
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model": name,
                "train_samples": 200,
                "validation_samples": 50,
                "test_samples": 50,
                "condition_encoder": ("lstm_bn" if name == "lstm_bn_flow" else
                                      "lstm" if name == "lstm_flow" else
                                      "transformer" if name == "transformer_flow" else "lstm"),
                "observation": "sig_c1",
                "parameter_dim": int(data.train_theta.shape[1]),
            },
        )
    key = jr.PRNGKey(args.seed + MODEL_NAMES.index(name) * 1000)
    model = build_model(name, data.train_theta.shape[1], key, args.width,
                        args.diffusion_steps, args.encoder_layers,
                        args.transformer_heads, args.flow_dropout,
                        args.transformer_model_dim, args.transformer_ff_dim,
                        args.pretrained_lstm_bn_encoder_dir)
    is_mems_flow = name in ("lstm_bn_flow", "lstm_flow", "transformer_flow")
    train_x, train_y = jnp.asarray(data.train_x), jnp.asarray(data.train_theta)
    val_x, val_y = jnp.asarray(data.val_x), jnp.asarray(data.val_theta)
    batchnorm_state = model.initial_batchnorm_state() if is_mems_flow else None
    if is_mems_flow:
        model = model.initialize_actnorm(train_y, train_x, batchnorm_state)
    steps_per_epoch = (len(data.train_x) + args.batch_size - 1) // args.batch_size
    if is_mems_flow:
        learning_rate = lambda count: transformer_learning_rate(
            count, steps_per_epoch=steps_per_epoch, total_epochs=args.epochs,
            start_lr=args.learning_rate,
            hold_epochs=args.lr_hold_epochs,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
        )
    elif args.lr_scheduler:
        learning_rate = lambda count: warmup_cosine_learning_rate(
            count, steps_per_epoch=steps_per_epoch,
            start_lr=args.learning_rate,
            warmup_epochs=args.lr_warmup_epochs,
            decay_epochs=args.lr_decay_epochs,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
        )
    else:
        learning_rate = args.learning_rate
    if is_mems_flow:
        optimizer = optax.adam(learning_rate)
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adamw(learning_rate, weight_decay=1e-4),
        )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_fn, make_step = loss_and_step(name)
    best_loss, best_train_loss = np.inf, np.inf
    best_model, best_state, stale, history = model, batchnorm_state, 0, []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        key, perm_key, step_key, val_key = jr.split(key, 4)
        perm = np.asarray(jr.permutation(perm_key, len(train_x)))
        losses = []
        for begin in range(0, len(perm), args.batch_size):
            idx = perm[begin:begin + args.batch_size]
            step_key, batch_key = jr.split(step_key)
            if is_mems_flow:
                model, batchnorm_state, opt_state, _, loss = mems_flow_step(
                    model, batchnorm_state, train_x[idx], train_y[idx], opt_state,
                    optimizer, batch_key, args.l2_coefficient)
            else:
                kwargs = {"beta": 0.05} if name == "vae" else ({"kl_weight": 1e-3} if name == "bnn" else {})
                model, opt_state, loss = make_step(model, train_x[idx], train_y[idx], batch_key,
                                                   opt_state, optimizer, **kwargs)
            losses.append(float(loss))
        val_loss = float(mems_flow_validation(model, batchnorm_state, val_x, val_y)
                         if is_mems_flow else loss_fn(model, val_x, val_y, val_key))
        history.append((epoch + 1, float(np.mean(losses)), val_loss))
        current_lr = (float(learning_rate(epoch * steps_per_epoch))
                      if callable(learning_rate) else float(learning_rate))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "train/loss": history[-1][1],
                           "validation/loss": val_loss,
                           "optimization/learning_rate": current_lr}, step=epoch + 1)
        train_loss = history[-1][1]
        if np.isfinite(val_loss) and val_loss < best_loss - args.min_delta:
            best_loss, best_model, best_state, stale = (
                val_loss, model, batchnorm_state, 0)
            eqx.tree_serialise_leaves(out_dir / f"{name}.eqx", best_model)
            if best_state is not None:
                eqx.tree_serialise_leaves(out_dir / f"{name}.state.eqx", best_state)
        elif np.isfinite(val_loss):
            stale += 1
        if np.isfinite(train_loss) and train_loss < best_train_loss - args.min_delta:
            best_train_loss = train_loss
        if (epoch + 1) % args.log_every == 0:
            print(f"{name:8s} epoch {epoch + 1:4d} train={history[-1][1]:.5f} val={val_loss:.5f}", flush=True)
        if stale >= args.patience:
            break

    # Make the restored validation checkpoint explicit, including BatchNorm state.
    model, batchnorm_state = best_model, best_state
    eqx.tree_serialise_leaves(out_dir / f"{name}.eqx", model)
    if batchnorm_state is not None:
        eqx.tree_serialise_leaves(out_dir / f"{name}.state.eqx", batchnorm_state)
    with (out_dir / f"{name}_history.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(("epoch", "train_loss", "val_loss")); writer.writerows(history)
    metrics = {"model": name, "epochs_trained": len(history),
               "best_validation_loss": best_loss,
               "best_training_loss": best_train_loss,
               "early_stopping_monitor": (
                   "validation_nll" if name in (
                       "lstm_bn_flow", "lstm_flow", "transformer_flow",
                       "nsf") else "validation_loss"),
               "train_seconds": time.perf_counter() - start}
    (out_dir / f"{name}_training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.log({f"train/{key}": value for key, value in metrics.items()
                       if isinstance(value, (int, float))})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb_run.summary[f"train/{key}"] = value
        wandb_run.finish()
    print(name, metrics, flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("results/MEMS_end_of_line"))
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=2000)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--kernel-gamma", type=float, default=0.1,
                   help="Total LR multiplier over max-explr-epoch epochs.")
    p.add_argument("--max-explr-epoch", type=int, default=2000,
                   help="Exponential scheduler horizon; stepping stops after three horizons.")
    p.add_argument("--lr-decay", type=float, default=0.8,
                   help="LSTM-BN total LR multiplier over each decay horizon.")
    p.add_argument("--lr-decay-epochs", type=int, default=2000,
                   help="Cosine-decay horizon for non-MEMS inference heads.")
    p.add_argument("--lr-warmup-epochs", type=int, default=0,
                   help="Linear-warmup epochs for non-MEMS inference heads.")
    p.add_argument("--lr-scheduler", action="store_true",
                   help="Use the legacy schedules for non-MEMS comparison models; "
                        "the LSTM-BN notebook schedule is always enabled.")
    p.add_argument("--min-learning-rate-ratio", type=float, default=0.01,
                   help="Final LR as a fraction of the initial LR for cosine decay.")
    p.add_argument("--lr-hold-epochs", type=int, default=500,
                   help="Epochs at constant initial LR before MEMS-flow cosine decay.")
    p.add_argument("--transformer-hold-epochs", type=int, default=500,
                   help="Epochs at constant initial LR before Transformer cosine decay.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-model-dim", type=int, default=None,
                   help="Internal Transformer width; output remains --width dimensional.")
    p.add_argument("--transformer-ff-dim", type=int, default=None,
                   help="Transformer feed-forward width; defaults to 4x model width.")
    p.add_argument("--l2-coefficient", type=float, default=1e-5)
    p.add_argument("--flow-dropout", type=float, default=0.1,
                   help="Dropout applied to flow conditioning features during training.")
    p.add_argument("--pretrained-lstm-bn-encoder-dir", type=Path, default=None,
                   help="Use and freeze the LSTM-BN encoder/state from this result directory.")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--flow-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-signal-standardization", action="store_true",
                   help="Train directly on source signal values without centering/scaling.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    p.add_argument("--wandb-project", default="MEMS_end_of_line")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="mems-simulated-lstm")
    args = p.parse_args()
    if not 0.0 <= args.flow_dropout < 1.0:
        p.error("--flow-dropout must be in [0, 1)")
    if not 0.0 < args.min_learning_rate_ratio <= 1.0:
        p.error("--min-learning-rate-ratio must be in (0, 1]")
    if args.lr_hold_epochs < 0:
        p.error("--lr-hold-epochs must be non-negative")
    if args.lr_warmup_epochs < 0:
        p.error("--lr-warmup-epochs must be non-negative")
    if args.lr_decay_epochs <= 0:
        p.error("--lr-decay-epochs must be positive")
    if args.patience <= 0:
        p.error("--patience must be positive")
    transformer_model_dim = args.transformer_model_dim or args.width
    if transformer_model_dim % args.transformer_heads:
        p.error("Transformer model dimension must be divisible by --transformer-heads")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_mems(args.data_dir, seed=args.seed,
                     standardize_signals=not args.no_signal_standardization)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in args.models:
        train_one(name, data, args, args.output_dir)


if __name__ == "__main__":
    main()
