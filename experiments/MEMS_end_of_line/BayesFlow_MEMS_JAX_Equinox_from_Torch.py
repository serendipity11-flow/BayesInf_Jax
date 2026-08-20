"""Compatibility entry point for the former all-in-one MEMS notebook.

Model components live in :mod:`bayesinf.models.networks` and
:mod:`bayesinf.models.normalizing_flow`; data, training, evaluation, and plots
live in this experiment package. New runs should normally invoke
``train_inference`` directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax

from experiments.MEMS_end_of_line.train_inference import main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def notebook_compatible_arguments():
    output = os.getenv(
        "BAYESFLOW_JAX_OUTPUT_DIR",
        str(PROJECT_ROOT / "results/results_parameter_identification/"
            "MEMS_end_of_line/notebook_lstm_bn"),
    )
    return [
        "--data-dir", os.getenv(
            "BAYESFLOW_JAX_DATA_DIR", "/home/zhangyi/data/MEMS/MEMS_simulated"),
        "--output-dir", output,
        "--models", "lstm_bn_flow",
        "--epochs", os.getenv("BAYESFLOW_JAX_EPOCHS", "10000"),
        "--patience", os.getenv("BAYESFLOW_JAX_PATIENCE", "2000"),
        "--batch-size", "200",
        "--learning-rate", "1e-4",
        "--lr-decay", "0.8",
        "--lr-decay-epochs", "2000",
        "--width", "64",
        "--encoder-layers", "2",
        "--no-signal-standardization",
        "--seed", os.getenv("BAYESFLOW_JAX_SEED", "42"),
        "--log-every", "25",
    ]


if __name__ == "__main__":
    if os.getenv("BAYESFLOW_JAX_REQUIRE_GPU", "0") == "1" and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, but JAX selected {jax.default_backend()!r}")
    if len(sys.argv) == 1:
        sys.argv.extend(notebook_compatible_arguments())
    main()
