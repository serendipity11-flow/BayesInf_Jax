from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments.ddpm_sbm_comparison.core import MatchedDiffusion, sample, train_step
from experiments.ddpm_sbm_comparison.data import make_dataset
from experiments.ddpm_sbm_comparison.metrics import sliced_wasserstein, summarize


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled DDPM/SBM parameter-identification comparison")
    parser.add_argument("--cases", nargs="+", choices=("basic", "linear", "multimodal"), default=["basic", "linear", "multimodal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--sample-steps", nargs="+", type=int, default=[25, 50, 100, 250, 500, 1000])
    parser.add_argument("--posterior-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("experiments/ddpm_sbm_comparison/results"))
    parser.add_argument("--smoke", action="store_true", help="Tiny end-to-end validation run")
    return parser.parse_args()


def plot_samples(path, ddpm, sbm, truth, names, case):
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), squeeze=False)
    for index, name in enumerate(names):
        ax = axes[0, index]
        ax.hist(ddpm[0, :, index], bins=35, density=True, alpha=0.5, label="DDPM")
        ax.hist(sbm[0, :, index], bins=35, density=True, alpha=0.5, label="SBM")
        ax.axvline(truth[0, index], color="black", linestyle="--", label="truth")
        ax.set_xlabel(name)
    axes[0, 0].legend()
    fig.suptitle(f"{case}: posterior for test observation 0")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if args.smoke:
        args.cases, args.seeds = ["basic"], [0]
        args.train_size, args.validation_size, args.test_size = 32, 16, 4
        args.epochs, args.batch_size, args.sample_steps, args.posterior_samples = 2, 16, [4], 8
        args.width, args.depth, args.patience = 16, 1, 2
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "config.json").open("w") as handle:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, handle, indent=2)

    rows = []
    for case in args.cases:
        for seed in args.seeds:
            key = jr.PRNGKey(seed)
            data_keys = jr.split(jr.fold_in(key, 10), 3)
            dataset = make_dataset(case, data_keys, args.train_size, args.validation_size, args.test_size, args.dt, args.duration, args.noise_std)
            models, training_info = {}, {}
            initialization_key = jr.fold_in(key, 20)
            for method_index, method in enumerate(("ddpm", "sbm")):
                model = MatchedDiffusion(dataset.train_observations[0].size, dataset.train_parameters.shape[-1],
                                         args.width, args.depth, initialization_key, method, max(args.sample_steps))
                optimizer = optax.adamw(args.learning_rate, weight_decay=1e-4)
                state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
                best_model, best_loss, stale = model, np.inf, 0
                start = time.perf_counter()
                for epoch in range(args.epochs):
                    # Both methods see the same minibatch ordering; only time sampling differs.
                    permutation_key = jr.fold_in(jr.fold_in(key, 100), epoch)
                    loss_key = jr.fold_in(jr.fold_in(key, 200 + method_index), epoch)
                    indices = jr.permutation(permutation_key, args.train_size)[: args.batch_size]
                    model, state, loss = train_step(model, dataset.train_observations[indices], dataset.train_parameters[indices], loss_key, optimizer, state)
                    if epoch % 10 == 0 or epoch == args.epochs - 1:
                        validation_loss = float(model.loss(dataset.validation_observations, dataset.validation_parameters,
                                                           jr.fold_in(loss_key, 1)))
                        if validation_loss < best_loss:
                            best_model, best_loss, stale = model, validation_loss, 0
                        else:
                            stale += 10
                        if stale >= args.patience:
                            break
                training_seconds = time.perf_counter() - start
                models[method] = best_model
                training_info[method] = (epoch + 1, best_loss, training_seconds)
                eqx.tree_serialise_leaves(args.output / f"{case}_{method}_seed{seed}.eqx", best_model)

            method_samples = {}
            for steps in args.sample_steps:
                for method_index, method in enumerate(("ddpm", "sbm")):
                    sample_key = jr.fold_in(key, steps * 2 + method_index)
                    start = time.perf_counter()
                    normalized = sample(models[method], method, dataset.test_observations, sample_key, args.posterior_samples, steps)
                    jax.block_until_ready(normalized)
                    sampling_seconds = time.perf_counter() - start
                    physical = np.asarray(normalized * dataset.parameter_std[:, None, :] + dataset.parameter_mean[:, None, :])
                    method_samples[(method, steps)] = physical
                    values = summarize(physical, np.asarray(dataset.test_parameters), case)
                    epochs_run, best_loss, training_seconds = training_info[method]
                    rows.append({"case": case, "seed": seed, "method": method, "steps": steps,
                                 "epochs": epochs_run, "validation_loss": best_loss,
                                 "training_seconds": training_seconds, "sampling_seconds": sampling_seconds, **values})
                rows[-2]["ddpm_sbm_sliced_wasserstein"] = rows[-1]["ddpm_sbm_sliced_wasserstein"] = sliced_wasserstein(
                    method_samples[("ddpm", steps)], method_samples[("sbm", steps)], seed=seed
                )
            last = args.sample_steps[-1]
            plot_samples(args.output / f"{case}_seed{seed}.png", method_samples[("ddpm", last)], method_samples[("sbm", last)],
                         np.asarray(dataset.test_parameters), dataset.parameter_names, case)
            print(f"finished case={case} seed={seed} "
                  f"ddpm_val={training_info['ddpm'][1]:.5f} sbm_val={training_info['sbm'][1]:.5f}")

    columns = sorted({key for row in rows for key in row})
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
