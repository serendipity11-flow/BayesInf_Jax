"""Run CPU Sobol analysis on the saved thermal FMU sensitivity simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesinf.applications.sensitivity_analysis.thermal_model import run_saved_fmu_analysis
from experiments.thermal_model_calibration.run import DEFAULT_RESULTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "sensitivity_analysis")
    args = parser.parse_args(argv)
    run_saved_fmu_analysis(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
