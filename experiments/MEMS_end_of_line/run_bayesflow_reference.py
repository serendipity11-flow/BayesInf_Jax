"""JAX reproduction of the legacy BayesFlow MEMS RealNVP experiment."""

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

from bayesinf.models.networks import StackedBidirectionalLSTMEncoder
from bayesinf.models.normalizing_flow import CondNormFlow, cnf_loss, make_step
from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_bayesflow_reference
from experiments.MEMS_end_of_line.plots import plot_parameter_regression


def reference_lr(epoch, start_lr=1e-3):
    """Match StepLR(500, 0.1), then ExponentialLR to 10% every 2000 epochs."""
    epoch = jnp.asarray(epoch, dtype=jnp.float32)
    decayed = start_lr * 0.1 * jnp.power(0.5, (epoch - 500.0) / 2000.0)
    return jnp.where(epoch < 500, start_lr,
                     jnp.where(epoch < 8500, decayed, start_lr * 0.1 * (0.5 ** 4)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/results_parameter_identification/MEMS_end_of_line/bayesflow_reference_jax"))
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--posterior-samples", type=int, default=100)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    p.add_argument("--wandb-project", default="MEMS_end_of_line")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="bayesflow-reference-jax")
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_bayesflow_reference(args.data_dir)
    x_train, y_train = jnp.asarray(data.train_x), jnp.asarray(data.train_theta)
    x_val, y_val = jnp.asarray(data.val_x), jnp.asarray(data.val_theta)

    key, encoder_key = jr.split(jr.PRNGKey(args.seed))
    model = CondNormFlow(6, 4, key, n_coupling=4, hidden_nf=64,
                         num_hidden_nf=4, encoded_latent_dim=64,
                         hidden_rnn=64, rnn_type="lstm", flow_type="realnvp")
    reference_encoder = StackedBidirectionalLSTMEncoder(4, 64, 64, encoder_key, n_layers=2)
    model = eqx.tree_at(lambda m: m.encoder, model, reference_encoder)
    schedule = optax.inject_hyperparams(optax.adamw)(
        learning_rate=lambda count: reference_lr(count), weight_decay=1e-5)
    optimizer = optax.chain(optax.clip_by_global_norm(100.0), schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, name=f"bayesflow-reference-realnvp-seed{args.seed}",
                         mode=args.wandb_mode, config={
                             **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                             "architecture": "conditional_realnvp",
                             "flow_blocks": 4, "coupling_hidden_layers": 4,
                             "coupling_width": 64, "summary_network": "2_layer_bidirectional_lstm",
                             "summary_dim": 64, "batch_size": 200, "weight_decay": 1e-5,
                             "gradient_clip": 100.0, "split": "ordered_200_50_50",
                             "signal_shape": "96x4", "legacy_dropout_rate": 0.1,
                             "note": "JAX flow has deterministic coupling MLPs; dropout metadata retained from reference.",
                         })

    best_loss, best_model, stale, history = np.inf, model, 0, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        key, train_key, val_key = jr.split(key, 3)
        model, opt_state, train_loss = make_step(model, x_train, y_train, train_key,
                                                 opt_state, optimizer)
        val_loss = cnf_loss(model, x_val, y_val)
        train_value, val_value = float(train_loss), float(val_loss)
        lr = reference_lr(epoch)
        history.append((epoch + 1, train_value, val_value, lr))
        if run is not None:
            run.log({"epoch": epoch + 1, "train/loss": train_value,
                     "validation/loss": val_value, "optimization/learning_rate": lr},
                    step=epoch + 1)
        if val_value < best_loss - 1e-4:
            best_loss, best_model, stale = val_value, model, 0
            eqx.tree_serialise_leaves(args.output_dir / "realnvp.eqx", best_model)
        elif epoch >= 8500:
            stale += 1
        if (epoch + 1) % args.log_every == 0:
            print(f"epoch={epoch + 1} train={train_value:.6f} val={val_value:.6f} lr={lr:.3e}", flush=True)
        if epoch >= 8500 and stale >= 1000:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    with (args.output_dir / "history.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(("epoch", "train_loss", "val_loss", "learning_rate")); writer.writerows(history)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    samples = np.asarray(best_model.sample(jnp.asarray(data.test_x), jr.fold_in(key, 99), args.posterior_samples))
    np.save(args.output_dir / "realnvp_test_samples.npy", samples.astype(np.float32))
    plot_parameter_regression(samples, data.test_theta, "BayesFlow-reference JAX RealNVP",
                              args.output_dir / "realnvp_parameter_regression.png")
    mean = samples.mean(axis=0)
    rmse = float(np.sqrt(np.mean((mean - data.test_theta) ** 2)))
    mae = float(np.mean(np.abs(mean - data.test_theta)))
    if run is not None:
        run.log({"test/rmse": rmse, "test/mae": mae,
                 "train/seconds": time.perf_counter() - start})
        run.finish()
    print(f"test_rmse={rmse:.6f} test_mae={mae:.6f}")


if __name__ == "__main__":
    main()
