"""Evaluate a trained PELS-VAE and plot held-out FMU reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.model_calibration import CalibrationNormalization, load_fmu_dataset
from bayesinf.models.PELS_VAE import PELS_VAE
from experiments.thermal_model_calibration.run import DEFAULT_DATA, DEFAULT_RESULTS


PARAMETER_NAMES = (
    "CF_heatUp", "CF_coolDown", "CF_zetaFBT", "CF_zetaRBT", "CF_zetaTFR/BFR",
    "CF_zetaFLR/RLR", "CF_airExchangeFBT", "CF_airExchangeRBT",
    "CF_airExchangeTFR/BFR", "CF_airExchangeFLR/RLR", "CF_alpha_external",
    "CF_alpha_internal", "CF_hull_capacity", "CF_hull_insulation", "ratioBottomTop",
    "ratioFrontRear", "A_internal_mass", "CF_alpha_mass", "CF_v_air_mass",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_RESULTS / "pels_vae")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "pels_vae_evaluation")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--passes", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=25)
    result.add_argument("--plot-indices", type=int, nargs="+", default=(0, 50, 100, 150))
    return result


def _predict(model, state, parameters, key, passes, batch_size):
    predictions = []
    for begin in range(0, len(parameters), batch_size):
        batch = jnp.asarray(parameters[begin:begin + batch_size])
        key, batch_key = jr.split(key)
        prediction = model.predict(batch, state, key=batch_key, passes=passes)
        predictions.append(np.asarray(prediction, dtype=np.float32))
    return np.concatenate(predictions), key


def _metrics(prediction, truth):
    error = prediction - truth
    residual_sum = float(np.sum(error ** 2))
    total_sum = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse_celsius": float(np.sqrt(np.mean(error ** 2))),
        "mae_celsius": float(np.mean(np.abs(error))),
        "max_absolute_error_celsius": float(np.max(np.abs(error))),
        "r2": float(1.0 - residual_sum / max(total_sum, 1e-12)),
        "per_channel_rmse_celsius": np.sqrt(np.mean(error ** 2, axis=(0, 1))).tolist(),
        "per_channel_mae_celsius": np.mean(np.abs(error), axis=(0, 1)).tolist(),
    }


def _plot_one(truth, prediction, index, output):
    time_minutes = np.arange(truth.shape[1]) * 18.0 / 60.0
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for channel, axis in enumerate(axes.flat):
        axis.plot(time_minutes, truth[index, :, channel], color="black", lw=1.8, label="FMU")
        axis.plot(time_minutes, prediction[index, :, channel], color="#2563eb", lw=1.4,
                  ls="--", label="PELS-VAE")
        axis.set_title(f"Temperature sensor {2 * channel}")
        axis.set_ylabel("Temperature [°C]")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Time [min]")
    fig.suptitle(f"Held-out thermal trajectory reconstruction — test sample {index}")
    fig.tight_layout()
    fig.savefig(output / f"reconstruction_test_{index:03d}.png", dpi=180)
    fig.savefig(output / f"reconstruction_test_{index:03d}.pdf")
    plt.close(fig)


def main(argv=None):
    args = parser().parse_args(argv)
    config_path = args.checkpoint_dir / "training_config.json"
    training = json.loads(config_path.read_text())
    data = load_fmu_dataset(args.data_dir, seed=args.seed, shuffle=True)
    sequence_length, channels = data.train.signals.shape[1:]
    template = PELS_VAE(
        data.train.parameters.shape[1], training["latent_dim"], sequence_length, channels,
        training["hidden_dim"], training["hidden_dim"], n_lstm_layers=1,
        n_linear_layers=4, dropout=0.0, key=jr.PRNGKey(args.seed),
    )
    model = eqx.nn.inference_mode(
        eqx.tree_deserialise_leaves(args.checkpoint_dir / "pels_vae.eqx", template)
    )
    state = eqx.tree_deserialise_leaves(
        args.checkpoint_dir / "pels_vae_state.eqx", template.initial_state())
    normalization = CalibrationNormalization.load(args.checkpoint_dir / "normalization.npz")
    normalized_parameters = normalization.normalize_parameters(data.test.parameters)
    prediction, _ = _predict(model, state, normalized_parameters, jr.PRNGKey(args.seed + 1),
                             args.passes, args.batch_size)
    prediction = normalization.denormalize_signals(prediction)
    metrics = _metrics(prediction, data.test.signals)
    metrics.update({"test_samples": len(prediction), "latent_passes": args.passes,
                    "checkpoint": str(args.checkpoint_dir / "pels_vae.eqx")})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "test_reconstruction.npy", prediction)
    (args.output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2))
    for index in args.plot_indices:
        if 0 <= index < len(prediction):
            _plot_one(data.test.signals, prediction, index, args.output_dir)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
