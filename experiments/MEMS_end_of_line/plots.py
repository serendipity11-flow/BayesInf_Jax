"""Regression and uncertainty plots for MEMS parameter identification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PARAM_NAMES = (r"$D_L$", r"$f_0$", r"$m$", r"$d_{off}$", r"$p_1$", r"$p_2$")


def plot_parameter_regression(
    samples: np.ndarray,
    truth: np.ndarray,
    model_name: str,
    output_path: Path | str,
) -> None:
    """Plot posterior estimates, 90% CIs, truth, and identity for six parameters.

    ``samples`` must have shape ``(posterior_samples, test_cases, parameters)``
    and ``truth`` shape ``(test_cases, parameters)``.
    """
    samples, truth = np.asarray(samples), np.asarray(truth)
    if samples.ndim != 3 or truth.ndim != 2 or samples.shape[1:] != truth.shape:
        raise ValueError(f"Incompatible sample/truth shapes: {samples.shape}, {truth.shape}")
    if truth.shape[1] != 6:
        raise ValueError(f"MEMS regression plot expects six parameters, got {truth.shape[1]}")

    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    for parameter, ax in enumerate(axes.flat):
        x, y = truth[:, parameter], mean[:, parameter]
        order = np.argsort(x)
        x, y = x[order], y[order]
        lo, hi = lower[order, parameter], upper[order, parameter]
        bounds = np.array([x.min(), x.max(), lo.min(), hi.max(), y.min(), y.max()])
        margin = max(0.03 * np.ptp(bounds), 1e-4)
        limit = (bounds.min() - margin, bounds.max() + margin)

        ax.plot(limit, limit, color="black", linestyle="--", linewidth=1.2,
                label="Ground truth / ideal")
        ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), fmt="o", markersize=3.8,
                    linewidth=0.8, capsize=1.8, alpha=0.72, color="#2563eb",
                    ecolor="#93c5fd", label="Posterior mean ± 90% CI")
        parameter_name = PARAM_NAMES[parameter]
        ax.set(xlabel=f"Ground truth {parameter_name}",
               ylabel=f"Estimated {parameter_name}",
               title=parameter_name, xlim=limit, ylim=limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        if parameter == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{model_name.upper()}: posterior regression on 50 held-out signals",
                 fontsize=15)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
