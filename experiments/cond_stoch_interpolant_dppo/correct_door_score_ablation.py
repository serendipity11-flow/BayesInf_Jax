"""Replace the legacy noisy DDPM bar with the official BC-sampler result."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .evaluate_across_datasets import write_csv


def main(score_results: Path, ddpm_results: Path, output: Path) -> None:
    with score_results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with ddpm_results.open(newline="") as handle:
        ddpm = [
            row for row in csv.DictReader(handle)
            if row["stage"] == "pretrain"
        ]
    if len(ddpm) != 1:
        raise ValueError(f"expected one aggregate DDPM pretrain row, got {len(ddpm)}")
    replacement = dict(ddpm[0])
    replacement["method"] = "DDPM-BC"
    corrected = [row for row in rows if row["method"] != "DDPM"] + [replacement]
    write_csv(output, corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-results", type=Path, required=True)
    parser.add_argument("--ddpm-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.score_results, args.ddpm_results, args.output)
