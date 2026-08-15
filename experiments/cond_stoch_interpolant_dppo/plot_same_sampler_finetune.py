"""Merge unchanged CSI results with sampler-controlled DDPM results and plot."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .evaluate_across_datasets import plot_learning_curves, write_csv


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(csi: Path, ddpm: Path, output: Path) -> None:
    csi_rows = [row for row in read_rows(csi) if row["method"] == "CSI-analytic"]
    ddpm_rows = read_rows(ddpm)
    if not csi_rows or not ddpm_rows:
        raise ValueError("both CSI and sampler-controlled DDPM rows are required")
    rows = csi_rows + ddpm_rows
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "checkpoints_combined.csv", rows)
    plot_learning_curves(rows, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi", type=Path, required=True)
    parser.add_argument("--ddpm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.csi, args.ddpm, args.output)
