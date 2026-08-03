"""Collect the five saved ALOHA state-estimation evaluation metrics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


METHODS = (
    ("amortized_realnvp", "amortized_realnvp"),
    ("flow_matching", "amortized_fm"),
    ("latent_ssm_flow", "latent_ssm_flow"),
    ("realnvp_flow_sde", "realnvp_flow_sde"),
    (
        "conditional_stochastic_interpolant",
        "conditional_stochastic_interpolant",
    ),
)


root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "results/results_state_estimation/Aloha_sim_insertion")
variant = sys.argv[2].strip("_") if len(sys.argv) > 2 else ""
variant_tag = f"_{variant}" if variant else ""
rows = []
for method, stem in METHODS:
    path = root / f"{stem}{variant_tag}_eval_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation metrics: {path}")
    with path.open(newline="") as handle:
        saved = next(csv.DictReader(handle))
    rows.append(
        {
            "method": method,
            "rmse": saved["rmse"],
            "mae": saved["mae"],
            "coverage_90": saved["coverage_90"],
            "metrics_file": str(path),
        }
    )

output = root / f"evaluation_summary{variant_tag}.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(output)
