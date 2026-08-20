"""Train the JAX PELS-VAE forward surrogate on FMU-generated trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr

from bayesinf.applications.model_calibration import (
    PELSTrainingConfig,
    load_fmu_dataset,
    train_pels_vae,
)
from bayesinf.models.PELS_VAE import PELS_VAE
from experiments.thermal_model_calibration.run import DEFAULT_DATA, DEFAULT_RESULTS


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "pels_vae")
    result.add_argument("--epochs", type=int, default=10_000)
    result.add_argument("--patience", type=int, default=1_000)
    result.add_argument("--batch-size", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--latent-dim", type=int, default=64)
    result.add_argument("--hidden-dim", type=int, default=512)
    result.add_argument("--seed", type=int, default=0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    print("JAX devices:", jax.devices(), flush=True)
    data = load_fmu_dataset(args.data_dir, seed=args.seed, shuffle=True)
    sequence_length, channels = data.train.signals.shape[1:]
    model = PELS_VAE(
        data.train.parameters.shape[1], args.latent_dim, sequence_length, channels,
        args.hidden_dim, args.hidden_dim, n_lstm_layers=1, n_linear_layers=4,
        dropout=0.0, key=jr.PRNGKey(args.seed),
    )
    state = model.initial_state()
    config = PELSTrainingConfig(
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(json.dumps({
        **asdict(config), "data_dir": str(args.data_dir), "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim, "sequence_length": sequence_length,
        "channels": channels,
    }, indent=2))
    best_loss, history = train_pels_vae(model, state, data, args.output_dir, config)
    print(json.dumps({"best_validation_loss": best_loss, "epochs_trained": len(history),
                      "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
