"""Aggregate and plot the inferred-source sigma_0 sweep."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt


root = Path("results/results_state_estimation/lorenz96_N12/si_inferred_sigma_sweep")
sigmas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
rows = []
for sigma in sigmas:
    tag = "0" if sigma == 0 else str(sigma)
    path = root / f"sig{tag}" / "eval_100x200" / "metrics.csv"
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    row["sigma_0"] = sigma
    rows.append(row)

summary = root / "sigma_sweep_100x200.csv"
fields = list(rows[0])
with summary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(sigmas, [float(r["rmse_all"]) for r in rows], "o-", label="All")
axes[0].plot(sigmas, [float(r["rmse_unobs"]) for r in rows], "s--", label="Unobserved")
axes[0].set_ylabel("RMSE")
axes[0].legend()
axes[1].plot(sigmas, [float(r["coverage_all"]) for r in rows], "o-")
axes[1].axhline(.9, color="red", linestyle="--", label="Nominal 90%")
axes[1].set_ylabel("Coverage")
axes[1].legend()
axes[2].plot(sigmas, [float(r["mciw_all"]) for r in rows], "o-")
axes[2].set_ylabel("Mean CI width")
for axis in axes:
    axis.set_xlabel(r"$\sigma_0$")
    axis.grid(alpha=.25)
figure.tight_layout()
figure.savefig(root / "sigma_sweep_100x200.png", dpi=240, bbox_inches="tight")
