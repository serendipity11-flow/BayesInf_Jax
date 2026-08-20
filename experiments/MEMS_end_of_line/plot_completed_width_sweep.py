"""Generate missing feature diagnostics for completed width-sweep runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(
        "results/results_parameter_identification/MEMS_end_of_line/encoder_width_sweep")
    for width in (8, 16, 32, 64):
        result_dir = root / f"width_{width}"
        subprocess.run(
            [sys.executable, "-m", "experiments.MEMS_end_of_line.plot_encoder_pca",
             "--results-dir", str(result_dir), "--models", "lstm_bn_flow"],
            check=True,
        )


if __name__ == "__main__":
    main()
