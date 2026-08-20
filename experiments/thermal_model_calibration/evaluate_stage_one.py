"""Evaluate and plot the trained first-stage thermal BayesFlow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.model_calibration import load_fmu_dataset
from bayesinf.applications.sensitivity_analysis.thermal_model import PARAMETER_NAMES
from bayesinf.models.normalizing_flow import BNNormFlow
from experiments.thermal_model_calibration.run import DEFAULT_DATA, DEFAULT_RESULTS


def posterior_metrics(samples, truth):
    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, (0.05, 0.95), axis=0)
    error = mean - truth
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "coverage_90": float(np.mean((truth >= lower) & (truth <= upper))),
        "mean_interval_width_90": float(np.mean(upper - lower)),
        "per_parameter_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "per_parameter_mae": np.mean(np.abs(error), axis=0).tolist(),
        "per_parameter_coverage_90": np.mean(
            (truth >= lower) & (truth <= upper), axis=0
        ).tolist(),
    }


def plot_regression(samples, truth, names, output):
    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, (0.05, 0.95), axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for index, axis in enumerate(axes.flat):
        error = np.stack((mean[:, index] - lower[:, index],
                          upper[:, index] - mean[:, index]))
        axis.errorbar(truth[:, index], mean[:, index], yerr=error, fmt="o", ms=3,
                      alpha=.6, capsize=1.5, color="#2563eb")
        limits = (min(truth[:, index].min(), lower[:, index].min()),
                  max(truth[:, index].max(), upper[:, index].max()))
        axis.plot(limits, limits, color="black", ls="--", lw=1)
        axis.set_title(names[index])
        axis.set_xlabel("True parameter")
        axis.set_ylabel("Posterior mean and 90% CI")
        axis.grid(alpha=.25)
    fig.suptitle("Stage-one BayesFlow posterior regression — fixed test set")
    fig.tight_layout()
    fig.savefig(output / "stage_one_parameter_regression.png", dpi=180)
    fig.savefig(output / "stage_one_parameter_regression.pdf")
    plt.close(fig)


def plot_target_posterior(samples, truth, names, output, target=0):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for index, axis in enumerate(axes.flat):
        axis.hist(samples[:, target, index], bins=35, density=True,
                  color="#60a5fa", alpha=.8)
        axis.axvline(truth[target, index], color="black", ls="--", lw=2, label="Truth")
        axis.axvline(samples[:, target, index].mean(), color="#dc2626", lw=1.5,
                     label="Posterior mean")
        axis.set_title(names[index])
        axis.set_xlabel("Parameter value")
        axis.set_ylabel("Density")
        axis.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"Stage-one posterior — fixed test sample {target}")
    fig.tight_layout()
    fig.savefig(output / f"stage_one_posterior_test_{target:03d}.png", dpi=180)
    fig.savefig(output / f"stage_one_posterior_test_{target:03d}.pdf")
    plt.close(fig)


def plot_history(path, output):
    values = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(values["epoch"], values["train_loss"], alpha=.45, lw=.8, label="Training")
    axis.plot(values["epoch"], values["validation_loss"], alpha=.45, lw=.8,
              label="Validation")
    window = min(100, max(1, len(values) // 10))
    kernel = np.ones(window) / window
    for field, color, label in (("train_loss", "#1d4ed8", "Training moving mean"),
                                ("validation_loss", "#dc2626", "Validation moving mean")):
        smooth = np.convolve(values[field], kernel, mode="valid")
        axis.plot(values["epoch"][window - 1:], smooth, color=color, lw=1.6, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Negative log likelihood")
    axis.set_title("Stage-one BayesFlow learning curves")
    axis.grid(alpha=.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "stage_one_training_history.png", dpi=180)
    fig.savefig(output / "stage_one_training_history.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS / "stage_one_bayesflow")
    parser.add_argument("--posterior-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    stage = args.result_dir / "stage_01"
    output = args.result_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.result_dir / "config.json").read_text())
    group = tuple(config["groups"][0])
    normalization = np.load(stage / "normalization.npz")
    data = load_fmu_dataset(args.data_dir, seed=args.seed, shuffle=True)
    if "x_min" in normalization.files:
        test_x = 2.0 * (data.test.signals - normalization["x_min"]) / (
            normalization["x_max"] - normalization["x_min"]
        ) - 1.0
    else:  # Backward compatibility with existing mean/std checkpoints.
        test_x = (data.test.signals - normalization["x_mean"]) / normalization["x_std"]
    truth = data.test.parameters[:, list(group)]
    template = BNNormFlow(
        len(group), test_x.shape[-1], jr.PRNGKey(args.seed),
        n_coupling=config["n_coupling"], hidden_dims=(config["width"],) * 4,
        encoded_dim=config["summary_dim"], encoder_hidden_size=config["rnn_hidden"],
        encoder_layers=config["recurrent_layers"], sequence_length=test_x.shape[1],
        activation=jnp.tanh, clamp=config["clamp"], clamp_activation="I",
    )
    model = eqx.tree_deserialise_leaves(stage / "bayesflow.eqx", template)
    state = eqx.tree_deserialise_leaves(
        stage / "bayesflow_state.eqx", template.initial_batchnorm_state()
    )
    predictions, key = [], jr.PRNGKey(args.seed + 100)
    for begin in range(0, len(test_x), args.batch_size):
        key, sample_key = jr.split(key)
        normalized = model.sample(
            jnp.asarray(test_x[begin:begin + args.batch_size]), sample_key,
            args.posterior_samples, state,
        )
        predictions.append(np.asarray(normalized))
    normalized_samples = np.concatenate(predictions, axis=1)
    if "theta_min" in normalization.files:
        samples = 0.5 * (normalized_samples + 1.0) * (
            normalization["theta_max"][None] - normalization["theta_min"][None]
        ) + normalization["theta_min"][None]
    else:
        samples = (normalized_samples * normalization["theta_std"][None] +
                   normalization["theta_mean"][None])
    metrics = posterior_metrics(samples, truth)
    metrics.update({"group": list(group), "parameter_names": [PARAMETER_NAMES[i] for i in group],
                    "test_samples": len(truth), "posterior_samples": args.posterior_samples})
    np.save(output / "stage_one_test_posterior_samples.npy", samples.astype(np.float32))
    (output / "stage_one_metrics.json").write_text(json.dumps(metrics, indent=2))
    names = [PARAMETER_NAMES[index] for index in group]
    plot_regression(samples, truth, names, output)
    plot_target_posterior(samples, truth, names, output)
    plot_history(stage / "history.csv", output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
