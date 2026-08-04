"""Plot aggregate ALOHA insertion state-estimation evaluation metrics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "amortized_realnvp": "Amortized\nRealNVP",
    "flow_matching": "Flow\nmatching",
    "latent_ssm_flow": "Latent SSM\nflow",
    "realnvp_flow_sde": "RealNVP\nFlowSDE",
    "conditional_stochastic_interpolant": "Conditional stochastic\ninterpolant",
}


def number(value: str) -> float:
    return float(value) if value else np.nan


root = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "results/results_state_estimation/Aloha_sim_insertion"
)
variant = sys.argv[2].strip("_") if len(sys.argv) > 2 else ""
variant_tag = f"_{variant}" if variant else ""
with (root / f"evaluation_summary{variant_tag}.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))

labels = [LABELS[row["method"]] for row in rows]
rmse = np.asarray([number(row["rmse"]) for row in rows])
mae = np.asarray([number(row["mae"]) for row in rows])
coverage = np.asarray([number(row["coverage_90"]) for row in rows])
colors = ["#4C78A8", "#59A14F", "#B279A2", "#F28E2B", "#E15759"]

fig, axes = plt.subplots(
    1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.45, 1.0]}
)

order = np.argsort(rmse)[::-1]
bars = axes[0].barh(
    np.arange(len(rows)), rmse[order], color=np.asarray(colors)[order], alpha=0.9
)
axes[0].set_yticks(np.arange(len(rows)), np.asarray(labels)[order])
axes[0].set_xlabel("Posterior-mean RMSE (normalized state)")
axes[0].set_title("State-estimation accuracy")
axes[0].grid(axis="x", alpha=0.25)
axes[0].bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
axes[0].set_xlim(0, max(rmse) * 1.18)

reported = np.flatnonzero(np.isfinite(coverage))
x = np.arange(len(reported))
width = 0.36
axes[1].bar(
    x - width / 2,
    mae[reported],
    width,
    color=np.asarray(colors)[reported],
    label="MAE",
)
reported_coverage = coverage[reported]
coverage_fraction = (
    reported_coverage
    if np.nanmax(reported_coverage) <= 1.0
    else reported_coverage / 100.0
)
axes[1].bar(
    x + width / 2,
    coverage_fraction,
    width,
    color=np.asarray(colors)[reported],
    alpha=0.45,
    hatch="//",
    label="90% CI coverage",
)
axes[1].axhline(0.9, color="#333333", linestyle="--", linewidth=1.2, label="Target coverage")
axes[1].set_xticks(x, np.asarray(labels)[reported])
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel("Metric value")
axes[1].set_title("Error and uncertainty calibration")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(frameon=False, fontsize=8, loc="upper left")

fig.suptitle(
    "ALOHA simulated insertion — "
    + ("GMM-noise" if variant == "gmm" else "noisy")
    + " proprioceptive state estimation",
    fontsize=14,
)
fig.text(
    0.5,
    0.01,
    "250-step trajectories · observation noise σ=0.1 · episode-level test split",
    ha="center",
    fontsize=9,
    color="#555555",
)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))

for suffix in ("png", "pdf"):
    fig.savefig(
        root / f"evaluation_comparison{variant_tag}.{suffix}",
        dpi=220,
        bbox_inches="tight",
    )
plt.close(fig)
