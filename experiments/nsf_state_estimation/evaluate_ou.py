"""Evaluate a trained OU NSF across horizons and Monte Carlo seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    evaluate_ou_moments,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing model.eqx and run.json")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--horizons", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = json.loads((args.run / "run.json").read_text())
    model_config = NSFConfig(**metadata["model_config"])
    model = NeuralStochasticFlow.from_config(
        model_config,
        key=jax.random.key(metadata["seed"]),
        state_mean=jnp.asarray(metadata["normalization"]["state_mean"]),
        state_std=jnp.asarray(metadata["normalization"]["state_std"]),
    )
    model = eqx.tree_deserialise_leaves(args.run / "model.eqx", model)
    process = OrnsteinUhlenbeck(**metadata["ou"])
    horizons = [float(value) for value in args.horizons.split(",")]
    initial = jnp.full((model.state_dim,), -1.0)

    rows = []
    for horizon in horizons:
        for seed in range(args.seeds):
            metrics = evaluate_ou_moments(
                model,
                process,
                jax.random.key(seed),
                initial,
                horizon,
                num_samples=args.samples,
            )
            rows.append({
                "horizon": horizon,
                "seed": seed,
                "mean_absolute_error": float(metrics["mean_absolute_error"]),
                "variance_absolute_error": float(
                    metrics["variance_absolute_error"]
                ),
            })

    summary = []
    for horizon in horizons:
        selected = [row for row in rows if row["horizon"] == horizon]
        summary.append({
            "horizon": horizon,
            "seeds": args.seeds,
            "mean_absolute_error_mean": float(np.mean(
                [row["mean_absolute_error"] for row in selected]
            )),
            "mean_absolute_error_std": float(np.std(
                [row["mean_absolute_error"] for row in selected], ddof=1
            )) if args.seeds > 1 else 0.0,
            "variance_absolute_error_mean": float(np.mean(
                [row["variance_absolute_error"] for row in selected]
            )),
            "variance_absolute_error_std": float(np.std(
                [row["variance_absolute_error"] for row in selected], ddof=1
            )) if args.seeds > 1 else 0.0,
        })
    report = {
        "redistribution_status": "private-not-approved",
        "samples_per_seed": args.samples,
        "rows": rows,
        "summary": summary,
    }
    output = args.output or args.run / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
