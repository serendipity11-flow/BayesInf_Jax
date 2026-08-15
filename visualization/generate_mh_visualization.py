"""Generate the two-parameter Metropolis--Hastings illustration.

Run from the repository root with:

    py -3.12 scripts/generate_mh_visualization.py

The example targets a correlated bivariate Gaussian posterior and uses a
symmetric Gaussian random-walk proposal.  A fixed seed makes the four figures
fully reproducible.  The output directory can be changed with ``--output-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 7
MEAN = np.array([1.0, -0.5])
COVARIANCE = np.array([[1.0, 0.75], [0.75, 1.25]])
PROPOSAL_STD = 0.65
INITIAL_STATE = np.array([-3.5, 2.8])
N_ITERATIONS = 2_500
BURN_IN = 500


def log_posterior(theta: np.ndarray) -> float:
    """Unnormalised log density of the illustrative posterior."""
    delta = theta - MEAN
    return -0.5 * float(delta @ np.linalg.solve(COVARIANCE, delta))


def metropolis_hastings() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw a random-walk MH chain and retain proposal decisions."""
    rng = np.random.default_rng(SEED)
    chain = np.empty((N_ITERATIONS + 1, 2))
    proposals = np.empty((N_ITERATIONS, 2))
    accepted = np.zeros(N_ITERATIONS, dtype=bool)
    chain[0] = INITIAL_STATE

    for i in range(N_ITERATIONS):
        proposal = chain[i] + rng.normal(scale=PROPOSAL_STD, size=2)
        log_alpha = min(0.0, log_posterior(proposal) - log_posterior(chain[i]))
        proposals[i] = proposal
        accepted[i] = np.log(rng.random()) < log_alpha
        chain[i + 1] = proposal if accepted[i] else chain[i]

    return chain, proposals, accepted, float(accepted.mean())


def posterior_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-4.5, 5.0, 240)
    y = np.linspace(-4.0, 4.5, 240)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    delta = points - MEAN
    precision = np.linalg.inv(COVARIANCE)
    density = np.exp(-0.5 * np.einsum("...i,ij,...j->...", delta, precision, delta))
    return xx, yy, density


def style_parameter_panel(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, density: np.ndarray) -> None:
    ax.contour(xx, yy, density, levels=[0.08, 0.25, 0.50, 0.75], colors="#9aa0a6", linewidths=0.65)
    ax.set_xlim(-4.5, 5.0)
    ax.set_ylim(-4.0, 4.5)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$", rotation=0, labelpad=8)
    ax.tick_params(labelsize=6, length=2)
    ax.grid(alpha=0.12, linewidth=0.4)


def draw_panel(
    ax: plt.Axes,
    stage: str,
    chain: np.ndarray,
    proposals: np.ndarray,
    accepted: np.ndarray,
    acceptance_rate: float,
    xx: np.ndarray,
    yy: np.ndarray,
    density: np.ndarray,
) -> None:
    """Draw one stage of the MH example on an independent axis."""
    colors = {"chain": "#2166ac", "accepted": "#1b9e77", "rejected": "#d95f02"}
    style_parameter_panel(ax, xx, yy, density)

    if stage == "setup":
        ax.scatter(*chain[0], marker="s", s=24, color="black", zorder=5)
        ax.plot([chain[0, 0], proposals[0, 0]], [chain[0, 1], proposals[0, 1]],
                "--", color=colors["rejected"], lw=1)
        ax.scatter(*proposals[0], marker="x", s=28, color=colors["rejected"], zorder=5)
        settings = (
            r"target: $\mathcal{N}([1,-0.5]^\mathsf{T},\,\Sigma)$" "\n"
            r"$\Sigma=[[1,0.75],[0.75,1.25]]$" "\n"
            rf"proposal: $\mathcal{{N}}(\theta_i,{PROPOSAL_STD**2:.2f}I)$; seed: {SEED}"
        )
        ax.text(0.98, 0.96, settings, transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92})
    elif stage == "early_chain":
        early = 25
        ax.plot(chain[: early + 1, 0], chain[: early + 1, 1], "-o", ms=2.2, lw=0.8, color=colors["chain"])
        rejected = np.flatnonzero(~accepted[:early])
        ax.scatter(proposals[rejected, 0], proposals[rejected, 1], marker="x", s=15,
                   color=colors["rejected"], linewidths=0.7)
    elif stage == "burn_in":
        ax.plot(chain[: BURN_IN + 1, 0], chain[: BURN_IN + 1, 1],
                color=colors["chain"], lw=0.45, alpha=0.65)
        ax.scatter(chain[BURN_IN, 0], chain[BURN_IN, 1], s=22, color=colors["accepted"], zorder=5)
    elif stage == "retained_samples":
        retained = chain[BURN_IN + 1 :]
        ax.scatter(retained[:, 0], retained[:, 1], s=3, color=colors["chain"], alpha=0.28, rasterized=True)
        ax.scatter(*MEAN, marker="+", s=45, linewidths=1.2, color="black", zorder=5)
    else:
        raise ValueError(f"Unknown panel stage: {stage}")


def generate_figures(output_dir: Path) -> list[Path]:
    """Generate one tightly bounded PDF for each algorithm stage."""
    chain, proposals, accepted, acceptance_rate = metropolis_hastings()
    xx, yy, density = posterior_grid()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = ("setup", "early_chain", "burn_in", "retained_samples")
    outputs: list[Path] = []
    for stage in stages:
        fig, ax = plt.subplots(figsize=(3.25, 2.05), constrained_layout=True)
        draw_panel(ax, stage, chain, proposals, accepted, acceptance_rate, xx, yy, density)
        output = output_dir / f"mh_{stage}.pdf"
        fig.savefig(output, bbox_inches="tight", dpi=300)
        plt.close(fig)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for the four PDF files (default: %(default)s)",
    )
    args = parser.parse_args()
    outputs = generate_figures(args.output_dir)
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
