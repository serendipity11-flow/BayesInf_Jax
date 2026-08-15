"""Train and compare conditional posterior models on MEMS end-of-line data."""

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
import matplotlib.pyplot as plt
import numpy as np
import optax

from bayesinf.models.networks import TimeSeriesEncoder
from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_mems
from experiments.MEMS_end_of_line.plots import plot_parameter_regression


MODEL_NAMES = ("bnn", "vae", "realnvp", "nsf", "fm", "ddpm", "sbm")


def build_model(name: str, param_dim: int, key, width: int, diffusion_steps: int):
    latent, hidden, feature_dim = width, width, 1
    keys = jr.split(key, 2)
    if name == "bnn":
        from bayesinf.models.bayesian_neural_network import BNNPosteriorEstimator
        model = BNNPosteriorEstimator(feature_dim, param_dim, latent, [width, width], hidden, key=keys[0])
        model = eqx.tree_at(lambda m: m.encoder, model,
                            TimeSeriesEncoder(feature_dim, hidden, latent, keys[1], rnn="lstm"))
    elif name == "vae":
        from bayesinf.models.variational_autoencoder import cVAE
        model = cVAE(feature_dim, param_dim, latent, [width, width], hidden,
                     rnn_type="lstm", key=keys[0])
    elif name in ("realnvp", "nsf"):
        from bayesinf.models.normalizing_flow import CondNormFlow
        model = CondNormFlow(param_dim, feature_dim, keys[0], n_coupling=6,
                             hidden_nf=width, num_hidden_nf=2,
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
    return model


def loss_and_step(name):
    if name == "bnn":
        from bayesinf.models.bayesian_neural_network import batched_elbo_loss, make_step
        return lambda m, x, y, k: batched_elbo_loss(m, x, y, k, kl_weight=1e-3), make_step
    if name == "vae":
        from bayesinf.models.variational_autoencoder import cvae_loss, make_step
        return lambda m, x, y, k: cvae_loss(m, x, y, k, beta=0.05), make_step
    if name in ("realnvp", "nsf"):
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


def posterior_metrics(samples: np.ndarray, truth: np.ndarray):
    mean = samples.mean(axis=0)
    err = mean - truth
    lo, hi = np.quantile(samples, [0.05, 0.95], axis=0)
    # Multivariate energy score, averaged over observations.
    term1 = np.linalg.norm(samples - truth[None], axis=-1).mean(axis=0)
    rolled = np.roll(samples, len(samples) // 2, axis=0)
    term2 = 0.5 * np.linalg.norm(samples - rolled, axis=-1).mean(axis=0)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "coverage_90": float(np.mean((truth >= lo) & (truth <= hi))),
        "interval_width_90": float(np.mean(hi - lo)),
        "energy_score": float(np.mean(term1 - term2)),
    }


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
                "condition_encoder": "lstm",
                "observation": "sig_c1",
                "parameter_dim": int(data.train_theta.shape[1]),
            },
        )
    key = jr.PRNGKey(args.seed + MODEL_NAMES.index(name) * 1000)
    model = build_model(name, data.train_theta.shape[1], key, args.width, args.diffusion_steps)
    steps_per_epoch = (len(data.train_x) + args.batch_size - 1) // args.batch_size
    if args.lr_scheduler:
        total_steps = max(1, args.epochs * steps_per_epoch)
        learning_rate = optax.cosine_decay_schedule(
            init_value=args.learning_rate,
            decay_steps=total_steps,
            alpha=args.min_learning_rate_ratio,
        )
    else:
        learning_rate = args.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(learning_rate, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_fn, make_step = loss_and_step(name)
    train_x, train_y = jnp.asarray(data.train_x), jnp.asarray(data.train_theta)
    val_x, val_y = jnp.asarray(data.val_x), jnp.asarray(data.val_theta)
    best_loss, best_model, stale, history = np.inf, model, 0, []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        key, perm_key, step_key, val_key = jr.split(key, 4)
        perm = np.asarray(jr.permutation(perm_key, len(train_x)))
        losses = []
        for begin in range(0, len(perm), args.batch_size):
            idx = perm[begin:begin + args.batch_size]
            step_key, batch_key = jr.split(step_key)
            kwargs = {"beta": 0.05} if name == "vae" else ({"kl_weight": 1e-3} if name == "bnn" else {})
            model, opt_state, loss = make_step(model, train_x[idx], train_y[idx], batch_key,
                                               opt_state, optimizer, **kwargs)
            losses.append(float(loss))
        val_loss = float(loss_fn(model, val_x, val_y, val_key))
        history.append((epoch + 1, float(np.mean(losses)), val_loss))
        current_lr = (float(learning_rate(epoch * steps_per_epoch))
                      if args.lr_scheduler else float(args.learning_rate))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "train/loss": history[-1][1],
                           "validation/loss": val_loss,
                           "optimization/learning_rate": current_lr}, step=epoch + 1)
        if np.isfinite(val_loss) and val_loss < best_loss - args.min_delta:
            best_loss, best_model, stale = val_loss, model, 0
        else:
            stale += 1
        if (epoch + 1) % args.log_every == 0:
            print(f"{name:8s} epoch {epoch + 1:4d} train={history[-1][1]:.5f} val={val_loss:.5f}", flush=True)
        if stale >= args.patience:
            break

    eqx.tree_serialise_leaves(out_dir / f"{name}.eqx", best_model)
    with (out_dir / f"{name}_history.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(("epoch", "train_loss", "val_loss")); writer.writerows(history)
    key, sample_key = jr.split(key)
    test_x = jnp.asarray(data.test_x)
    if name == "sbm":
        normalized = best_model.sample(test_x, sample_key, args.posterior_samples, steps=args.diffusion_steps)
    elif name == "fm":
        normalized = best_model.sample(test_x, sample_key, args.posterior_samples, steps=args.flow_steps)
    else:
        normalized = best_model.sample(test_x, sample_key, args.posterior_samples)
    normalized = np.asarray(normalized)
    samples = normalized * data.theta_std[None] + data.theta_mean[None]
    np.save(out_dir / f"{name}_test_samples.npy", samples.astype(np.float32))
    plot_parameter_regression(samples, data.test_theta, name,
                              out_dir / f"{name}_parameter_regression.png")
    metrics = posterior_metrics(samples, data.test_theta)
    metrics.update(model=name, epochs_trained=len(history), best_val_loss=best_loss,
                   train_seconds=time.perf_counter() - start)
    if wandb_run is not None:
        wandb_run.log({f"test/{key}": value for key, value in metrics.items()
                       if isinstance(value, (int, float))})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb_run.summary[f"test/{key}"] = value
        wandb_run.finish()
    print(name, metrics, flush=True)
    return metrics


def plot_results(rows, out_dir):
    ordered = sorted(rows, key=lambda r: r["rmse"])
    names = [r["model"].upper() for r in ordered]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, metric, title in zip(axes, ("rmse", "energy_score", "coverage_90"),
                                 ("Posterior-mean RMSE", "Energy score", "90% coverage")):
        vals = [r[metric] for r in ordered]
        ax.bar(names, vals, color="#3b82f6")
        ax.set_title(title); ax.tick_params(axis="x", rotation=35)
        if metric == "coverage_90": ax.axhline(0.9, color="black", ls="--", lw=1)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("MEMS end-of-line parameter identification (50 held-out tests)")
    fig.tight_layout(); fig.savefig(out_dir / "model_comparison.png", dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("results/MEMS_end_of_line"))
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", action="store_true",
                   help="Use cosine learning-rate decay across all optimizer steps.")
    p.add_argument("--min-learning-rate-ratio", type=float, default=0.01,
                   help="Final LR as a fraction of the initial LR for cosine decay.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--posterior-samples", type=int, default=100)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--flow-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    p.add_argument("--wandb-project", default="MEMS_end_of_line")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="mems-simulated-lstm")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_mems(args.data_dir, args.seed)
    config = vars(args).copy(); config["data_dir"] = str(args.data_dir); config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = [train_one(name, data, args, args.output_dir) for name in args.models]
    fields = ("model", "rmse", "mae", "r2", "coverage_90", "interval_width_90",
              "energy_score", "best_val_loss", "epochs_trained", "train_seconds")
    with (args.output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    plot_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
