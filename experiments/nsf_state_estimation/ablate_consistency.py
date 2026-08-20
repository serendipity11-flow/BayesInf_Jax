"""Run a reproducible OU ablation over NSF consistency weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFBridge,
    NSFConsistencyConfig,
    NSFTrainingConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    fit_normalization,
    flow_consistency_loss,
    sample_ou_transition_data,
    sample_uniform_consistency_triplets,
    train_nsf_with_consistency,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    parser.add_argument(
        "--mode",
        choices=["one_to_two", "two_to_one", "bidirectional"],
        default="bidirectional",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int, default=8_192)
    parser.add_argument("--num-evaluation", type=int, default=1_024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bridge-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bridge-inner-steps", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=2)
    parser.add_argument("--evaluation-mc-samples", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--coupling-layers", type=int, default=2)
    parser.add_argument("--dt-min", type=float, default=0.05)
    parser.add_argument("--dt-max", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(weight < 0 for weight in args.weights):
        raise ValueError("consistency weights must be non-negative")
    root_key = jax.random.key(args.seed)
    data_key, triplet_key, evaluation_key = jax.random.split(root_key, 3)
    process = OrnsteinUhlenbeck(mean=0.7, kappa=0.8, diffusion=0.45)
    transitions = sample_ou_transition_data(
        data_key,
        process,
        args.num_train,
        1,
        delta_t_min=args.dt_min,
        delta_t_max=args.dt_max,
    )
    state_mean, state_std, _, _ = fit_normalization(
        jnp.concatenate((transitions["x_current"], transitions["x_next"]))
    )
    triplets = sample_uniform_consistency_triplets(
        triplet_key,
        transitions["x_current"],
        total_time_min=max(2.0 * args.dt_min, 1e-4),
        total_time_max=args.dt_max,
    )
    evaluation_count = min(args.num_evaluation, args.num_train)
    training_config = NSFTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_index, weight in enumerate(args.weights):
        model = NeuralStochasticFlow(
            1,
            hidden_size=args.hidden_size,
            hidden_depth=args.hidden_depth,
            num_coupling_layers=args.coupling_layers,
            state_mean=state_mean,
            state_std=state_std,
            key=jax.random.key(args.seed + 10),
        )
        bridge = NSFBridge.from_transition(
            model,
            hidden_size=args.hidden_size,
            hidden_depth=args.hidden_depth,
            num_coupling_layers=args.coupling_layers,
            key=jax.random.key(args.seed + 11),
        )
        consistency_config = NSFConsistencyConfig(
            mode=args.mode,
            consistency_weight=weight,
            bridge_inner_steps=args.bridge_inner_steps,
            bridge_learning_rate=args.bridge_learning_rate,
            num_mc_samples=args.mc_samples,
        )
        result = train_nsf_with_consistency(
            model,
            bridge,
            transitions,
            triplets,
            training_config,
            consistency_config,
        )
        evaluation_loss = flow_consistency_loss(
            result.model,
            result.bridge,
            jax.random.fold_in(evaluation_key, run_index),
            triplets["x_initial"][:evaluation_count],
            triplets["delta_t_before"][:evaluation_count],
            triplets["delta_t_after"][:evaluation_count],
            mode=args.mode,
            num_mc_samples=args.evaluation_mc_samples,
        )
        validation_nll = result.model.nll_loss(**transitions)
        label = f"weight_{weight:g}".replace(".", "p")
        eqx.tree_serialise_leaves(args.output / f"{label}_model.eqx", result.model)
        eqx.tree_serialise_leaves(args.output / f"{label}_bridge.eqx", result.bridge)
        rows.append(
            {
                "weight": weight,
                "validation_nll": float(validation_nll),
                "flow": float(evaluation_loss.total),
                "one_to_two": float(evaluation_loss.one_to_two),
                "two_to_one": float(evaluation_loss.two_to_one),
                "last_training_total": float(result.history["total"][-1]),
            }
        )

    report = {
        "redistribution_status": "private-not-approved",
        "seed": args.seed,
        "mode": args.mode,
        "training_config": vars(training_config),
        "bridge_inner_steps": args.bridge_inner_steps,
        "bridge_learning_rate": args.bridge_learning_rate,
        "mc_samples": args.mc_samples,
        "evaluation_mc_samples": args.evaluation_mc_samples,
        "results": rows,
    }
    (args.output / "consistency_ablation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
