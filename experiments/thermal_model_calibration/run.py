"""Run iterative thermal model calibration with JAX/Equinox."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.random as jr

from bayesinf.applications.model_calibration import (
    CalibrationNormalization,
    CalibrationConfig,
    IterativeModelCalibrator,
    PELSVAEForwardModel,
    load_fmu_dataset,
)
from bayesinf.models.PELS_VAE import PELS_VAE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "simulation_data"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "results" / "results_parameter_identification" / "thermal_model_calibration"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--vae-checkpoint", type=Path, required=True,
                        help="Equinox PELS-VAE checkpoint (PyTorch .pt files are not compatible).")
    result.add_argument("--vae-state", type=Path,
                        help="Optional serialized BatchNorm state; defaults to freshly initialized state.")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    result.add_argument("--real-test-index", type=int, default=0,
                        help="Index in the 50-sample noisy real-test proxy partition.")
    result.add_argument("--max-stages", type=int,
                        help="Stop after this many sensitivity-ordered stages.")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=1000)
    result.add_argument("--patience", type=int, default=1000)
    result.add_argument("--batch-size", type=int, default=200)
    result.add_argument("--posterior-samples", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=5e-4)
    result.add_argument("--min-learning-rate", type=float, default=5e-6)
    result.add_argument("--lr-hold-epochs", type=int, default=500)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--rnn-hidden", type=int, default=64)
    result.add_argument("--encoder-layers", type=int, default=2)
    result.add_argument("--latent-dim", type=int, default=64)
    result.add_argument("--vae-hidden", type=int, default=512)
    result.add_argument("--vae-passes", type=int, default=1)
    result.add_argument("--intermediate-eval-every", type=int, default=100)
    result.add_argument("--intermediate-posterior-samples", type=int, default=100)
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    result.add_argument("--wandb-project", default="thermal_model_calibration")
    result.add_argument("--wandb-group", default="iterative-double-realnvp")
    result.add_argument("--wandb-entity", default=None)
    result.add_argument("--wandb-loss-only", action="store_true",
                        help="Transmit only per-epoch training loss to W&B.")
    result.add_argument("--no-shuffle", action="store_true")
    return result


def load_surrogate(args, data):
    if args.vae_checkpoint.suffix == ".pt":
        raise ValueError("A PyTorch .pt checkpoint cannot be loaded into Equinox; provide a JAX .eqx checkpoint")
    sequence_length, channels = data.train.signals.shape[1:]
    parameter_dim = data.train.parameters.shape[1]
    template = PELS_VAE(
        parameter_dim, args.latent_dim, sequence_length, channels,
        args.vae_hidden, args.vae_hidden, n_lstm_layers=1,
        n_linear_layers=4, dropout=0.0, key=jr.PRNGKey(args.seed + 10),
    )
    model = eqx.tree_deserialise_leaves(args.vae_checkpoint, template)
    state = model.initial_state()
    if args.vae_state is not None:
        state = eqx.tree_deserialise_leaves(args.vae_state, state)
    normalization = CalibrationNormalization.load(
        args.vae_checkpoint.parent / "normalization.npz")
    return PELSVAEForwardModel(
        model, state, normalization, passes=args.vae_passes)


def main(argv=None):
    args = parser().parse_args(argv)
    data = load_fmu_dataset(args.data_dir, seed=args.seed, shuffle=not args.no_shuffle)
    forward_model = load_surrogate(args, data)
    config = CalibrationConfig(
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        posterior_samples=args.posterior_samples, width=args.width,
        rnn_hidden=args.rnn_hidden, learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        lr_hold_epochs=args.lr_hold_epochs, seed=args.seed,
        recurrent_layers=args.encoder_layers,
        intermediate_eval_every=args.intermediate_eval_every,
        intermediate_posterior_samples=args.intermediate_posterior_samples,
    )
    tracker = None
    if args.wandb_mode != "disabled":
        import wandb
        init_kwargs = dict(
            project=args.wandb_project, entity=args.wandb_entity,
            group=args.wandb_group, name=f"iterative-double-realnvp-seed{args.seed}",
            job_type="iterative-parameter-identification", mode=args.wandb_mode,
        )
        if not args.wandb_loss_only:
            init_kwargs["config"] = {
                **{key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
                "calibration_config": asdict(config),
            }
        tracker = wandb.init(**init_kwargs)
    try:
        result = IterativeModelCalibrator(
            config, args.output_dir, tracker=tracker,
            tracker_loss_only=args.wandb_loss_only,
        ).run(
            data, forward_model, real_test_index=args.real_test_index,
            max_stages=args.max_stages
        )
    finally:
        if tracker is not None:
            tracker.finish()
    summary = {
        "estimate": result.estimate.tolist(),
        "truth": result.truth.tolist(),
        "relative_error_percent": result.relative_error_percent.tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
