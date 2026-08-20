"""Evaluate trained MEMS conditional posterior models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_mems
from experiments.MEMS_end_of_line.plots import plot_parameter_regression
from experiments.MEMS_end_of_line.train_inference import MODEL_NAMES, build_model


def posterior_metrics(samples, truth):
    mean = samples.mean(axis=0)
    error = mean - truth
    low, high = np.quantile(samples, (0.05, 0.95), axis=0)
    term1 = np.linalg.norm(samples - truth[None], axis=-1).mean(axis=0)
    paired = np.roll(samples, len(samples) // 2, axis=0)
    term2 = 0.5 * np.linalg.norm(samples - paired, axis=-1).mean(axis=0)
    residual = np.sum(error ** 2)
    total = np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - residual / max(total, 1e-12)),
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
        "interval_width_90": float(np.mean(high - low)),
        "energy_score": float(np.mean(term1 - term2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path("results/MEMS_end_of_line"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=["lstm_bn_flow", "lstm_flow", "transformer_flow"])
    parser.add_argument("--posterior-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = args.results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    config = json.loads(config_path.read_text())
    seed = config.get("seed", 0) if args.seed is None else args.seed
    data = load_mems(
        args.data_dir, seed=seed,
        standardize_signals=not config.get("no_signal_standardization", False),
    )
    rows = []
    for name in args.models:
        checkpoint = args.results_dir / f"{name}.eqx"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        key = jr.PRNGKey(seed + MODEL_NAMES.index(name) * 1000)
        model = build_model(name, data.train_theta.shape[1], key,
                            int(config.get("width", 64)),
                            int(config.get("diffusion_steps", 100)),
                            int(config.get("encoder_layers", 2)),
                            int(config.get("transformer_heads", 4)),
                            float(config.get("flow_dropout", 0.0)),
                            config.get("transformer_model_dim"),
                            config.get("transformer_ff_dim"),
                            config.get("pretrained_lstm_bn_encoder_dir"))
        model = eqx.tree_deserialise_leaves(checkpoint, model)
        sample_key = jr.fold_in(key, 1)
        test_signals = jnp.asarray(data.test_x)
        if name in ("lstm_bn_flow", "lstm_flow", "transformer_flow"):
            state = model.initial_batchnorm_state()
            state_path = args.results_dir / f"{name}.state.eqx"
            if state is not None:
                if not state_path.exists():
                    raise FileNotFoundError(f"BatchNorm state not found: {state_path}")
                state = eqx.tree_deserialise_leaves(state_path, state)
            samples = model.sample(test_signals, sample_key,
                                   args.posterior_samples, state=state)
        elif name == "sbm":
            samples = model.sample(test_signals, sample_key, args.posterior_samples,
                                   steps=int(config.get("diffusion_steps", 100)))
        elif name == "fm":
            samples = model.sample(test_signals, sample_key, args.posterior_samples,
                                   steps=int(config.get("flow_steps", 100)))
        else:
            samples = model.sample(test_signals, sample_key, args.posterior_samples)
        samples = np.asarray(samples) * data.theta_std[None] + data.theta_mean[None]
        expected = (args.posterior_samples, len(data.test_x), 6)
        if samples.shape != expected:
            raise ValueError(f"Expected posterior samples {expected}, got {samples.shape}")
        np.save(args.results_dir / f"{name}_test_samples.npy", samples.astype(np.float32))
        plot_parameter_regression(samples, data.test_theta, name,
                                  args.results_dir / f"{name}_parameter_regression.png")
        metrics = posterior_metrics(samples, data.test_theta)
        metrics["model"] = name
        (args.results_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        rows.append(metrics)
        print(name, metrics, flush=True)

    fields = ("model", "rmse", "mae", "r2", "coverage_90",
              "interval_width_90", "energy_score")
    with (args.results_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
