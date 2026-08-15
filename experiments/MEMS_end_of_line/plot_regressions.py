"""Create per-method six-parameter regression figures from saved samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.MEMS_end_of_line.data import DEFAULT_DATA_DIR, load_mems
from experiments.MEMS_end_of_line.plots import plot_parameter_regression

MODEL_NAMES = ("bnn", "vae", "realnvp", "nsf", "fm", "ddpm", "sbm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/MEMS_end_of_line"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    truth = load_mems(args.data_dir, args.seed).test_theta
    for model in args.models:
        sample_path = args.results_dir / f"{model}_test_samples.npy"
        if not sample_path.exists():
            print(f"Skipping {model}: {sample_path} does not exist")
            continue
        output_path = args.results_dir / f"{model}_parameter_regression.png"
        plot_parameter_regression(np.load(sample_path), truth, model, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
