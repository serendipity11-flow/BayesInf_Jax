"""Cross-solver evaluation of the DDPM -> reverse VP-SDE limit.

For each fixed trained score network, this evaluates both samplers. Holding the
network fixed removes optimization variance, so the remaining discrepancy is
sampling discretization plus finite Monte Carlo error.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from experiments.ddpm_sbm_comparison.core import MatchedDiffusion, sample
from experiments.ddpm_sbm_comparison.data import make_dataset
from experiments.ddpm_sbm_comparison.metrics import sliced_wasserstein, summarize


def arguments():
    parser = argparse.ArgumentParser(description="Evaluate DDPM/SBM samplers on identical trained score networks")
    parser.add_argument("--input", type=Path, required=True, help="One case_seed result directory")
    parser.add_argument("--steps", nargs="+", type=int, default=[25, 50, 100, 250, 500, 1000])
    parser.add_argument("--posterior-samples", type=int, default=1000)
    return parser.parse_args()


def load_context(directory, steps):
    with (directory / "config.json").open() as handle:
        config = json.load(handle)
    case = config["cases"][0]
    seed = int(config["seeds"][0])
    key = jr.PRNGKey(seed)
    dataset = make_dataset(
        case,
        jr.split(jr.fold_in(key, 10), 3),
        int(config["train_size"]),
        int(config["validation_size"]),
        int(config["test_size"]),
        float(config["dt"]),
        float(config["duration"]),
        float(config["noise_std"]),
    )
    models = {}
    for trained_as in ("ddpm", "sbm"):
        template = MatchedDiffusion(
            dataset.train_observations[0].size,
            dataset.train_parameters.shape[-1],
            int(config["width"]),
            int(config["depth"]),
            jr.fold_in(key, 20),
            trained_as,
            max(int(x) for x in config["sample_steps"]),
        )
        models[trained_as] = eqx.tree_deserialise_leaves(
            directory / f"{case}_{trained_as}_seed{seed}.eqx", template
        )
    return case, seed, dataset, models


def main():
    args = arguments()
    case, seed, dataset, models = load_context(args.input, args.steps)
    truth = np.asarray(dataset.test_parameters)
    rows = []
    for trained_as, model in models.items():
        for steps in args.steps:
            outputs = {}
            # The identical key gives both samplers the same initial Gaussian draw.
            common_key = jr.fold_in(jr.PRNGKey(seed), 10_000 + steps)
            for sampler in ("ddpm", "sbm"):
                normalized = sample(model, sampler, dataset.test_observations, common_key,
                                    args.posterior_samples, steps)
                jax.block_until_ready(normalized)
                outputs[sampler] = np.asarray(
                    normalized * dataset.parameter_std[:, None, :] + dataset.parameter_mean[:, None, :]
                )
            distance = sliced_wasserstein(outputs["ddpm"], outputs["sbm"], seed=seed)
            for sampler in ("ddpm", "sbm"):
                rows.append({
                    "case": case,
                    "seed": seed,
                    "trained_as": trained_as,
                    "sampler": sampler,
                    "steps": steps,
                    "same_network_sliced_wasserstein": distance,
                    **summarize(outputs[sampler], truth, case),
                })
            print(f"case={case} seed={seed} trained_as={trained_as} steps={steps} SW={distance:.6f}")

    columns = sorted({name for row in rows for name in row})
    with (args.input / "convergence_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for trained_as in ("ddpm", "sbm"):
        subset = [row for row in rows if row["trained_as"] == trained_as and row["sampler"] == "ddpm"]
        ax.plot([row["steps"] for row in subset],
                [row["same_network_sliced_wasserstein"] for row in subset],
                marker="o", label=f"network trained as {trained_as.upper()}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reverse steps")
    ax.set_ylabel("DDPM–SBM sliced Wasserstein distance")
    ax.set_title(f"Same-network convergence: {case}, seed {seed}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.input / "theoretical_convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
