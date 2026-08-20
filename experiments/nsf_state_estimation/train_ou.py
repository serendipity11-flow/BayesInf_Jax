"""Train a solver-free NSF on analytic Ornstein--Uhlenbeck transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFConfig,
    NSFTrainingConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    evaluate_ou_moments,
    fit_normalization,
    sample_ou_transition_data,
    train_nsf_nll,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--num-train", type=int, default=32_768)
    parser.add_argument("--num-validation", type=int, default=8_192)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--coupling-layers", type=int, default=4)
    parser.add_argument("--dt-min", type=float, default=0.05)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--ou-mean", type=float, default=0.7)
    parser.add_argument("--ou-kappa", type=float, default=0.8)
    parser.add_argument("--ou-diffusion", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    root_key = jax.random.key(args.seed)
    model_key, train_key, validation_key, metric_key = jax.random.split(root_key, 4)
    process = OrnsteinUhlenbeck(
        mean=args.ou_mean, kappa=args.ou_kappa, diffusion=args.ou_diffusion
    )
    training = sample_ou_transition_data(
        train_key,
        process,
        args.num_train,
        args.state_dim,
        delta_t_min=args.dt_min,
        delta_t_max=args.dt_max,
    )
    validation = sample_ou_transition_data(
        validation_key,
        process,
        args.num_validation,
        args.state_dim,
        delta_t_min=args.dt_min,
        delta_t_max=args.dt_max,
    )
    state_mean, state_std, _, _ = fit_normalization(
        jnp.concatenate((training["x_current"], training["x_next"]), axis=0)
    )
    model_config = NSFConfig(
        state_dim=args.state_dim,
        hidden_size=args.hidden_size,
        hidden_depth=args.hidden_depth,
        num_coupling_layers=args.coupling_layers,
    )
    training_config = NSFTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
    )
    model = NeuralStochasticFlow.from_config(
        model_config,
        key=model_key,
        state_mean=state_mean,
        state_std=state_std,
    )

    def progress(step, loss):
        if step == 0 or (step + 1) % max(1, args.steps // 20) == 0:
            print(f"step={step + 1} nll={float(loss):.6f}", flush=True)

    model, losses = train_nsf_nll(
        model, training, training_config, progress=progress
    )
    validation_nll = -jnp.mean(model.log_prob(**validation))
    analytic_validation_nll = -jnp.mean(process.log_prob(
        validation["x_next"], validation["x_current"], validation["delta_t"]
    ))
    moments = evaluate_ou_moments(
        model,
        process,
        metric_key,
        jnp.full((args.state_dim,), -1.0),
        args.dt_max,
        num_samples=20_000,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
    report = {
        "redistribution_status": "private-not-approved",
        "seed": args.seed,
        "model_config": vars(model_config),
        "training_config": vars(training_config),
        "normalization": {
            "state_mean": np_list(state_mean),
            "state_std": np_list(state_std),
        },
        "ou": {
            "mean": args.ou_mean,
            "kappa": args.ou_kappa,
            "diffusion": args.ou_diffusion,
        },
        "data": {
            "num_train": args.num_train,
            "num_validation": args.num_validation,
            "delta_t_min": args.dt_min,
            "delta_t_max": args.dt_max,
        },
        "metrics": {
            "first_batch_nll": float(losses[0]),
            "last_batch_nll": float(losses[-1]),
            "validation_nll": float(validation_nll),
            "analytic_validation_nll": float(analytic_validation_nll),
            "long_gap_mean_absolute_error": float(moments["mean_absolute_error"]),
            "long_gap_variance_absolute_error": float(
                moments["variance_absolute_error"]
            ),
        },
    }
    (args.output / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


def np_list(value):
    """Convert a small device vector to a JSON-compatible list."""
    return [float(item) for item in value]


if __name__ == "__main__":
    main()
