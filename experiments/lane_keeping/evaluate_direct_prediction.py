"""Evaluate flow control predictions on held-out data without a control rollout."""

import argparse
import csv
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control import AmortizedSeqFlow
from experiments.lane_keeping.dataset import (
    BicycleConfig, generate_dataset, split_dataset,
)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - target
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((target - target.mean())**2)
    return {
        "mae_rad": float(np.mean(np.abs(residual))),
        "rmse_rad": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error_rad": float(np.max(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot),
        "bias_rad": float(np.mean(residual)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--root", type=Path,
                        default=Path("results/results_optimal_control/runs"))
    parser.add_argument("--directory", type=str, default=None,
                        help="Result directory name; defaults to seed<seed>")
    args = parser.parse_args()

    cfg = BicycleConfig()
    kd, ks, _, _ = jr.split(jr.PRNGKey(args.seed), 4)
    _, _, test = split_dataset(generate_dataset(kd, args.samples, cfg), ks)

    output = args.root / (args.directory or f"seed{args.seed}")
    model = AmortizedSeqFlow(cfg.horizon, cfg.delta_max, jr.PRNGKey(0))
    with (output / "lane_flow.eqx").open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)

    prediction = model.sample(
        test["observations"],
        test["reference"],
        jr.PRNGKey(args.seed + 1000),
        deterministic=True,
        past_controls=test["past_controls"],
        previous_steering=test["previous_steering"],
    )[0]
    target = test["actions"]
    prediction, target = np.asarray(prediction), np.asarray(target)

    rows = []
    for scope, y, p in (
        ("first_action", target[:, 0], prediction[:, 0]),
        ("full_chunk", target, prediction),
    ):
        rows.append({"scope": scope, **_metrics(y, p)})
    with (output / "direct_control_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    residual = prediction - target
    horizon_mae = np.mean(np.abs(residual), axis=0)
    example = int(np.argmax(np.abs(target[:, 0])))
    limit = max(np.max(np.abs(target[:, 0])), np.max(np.abs(prediction[:, 0])))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].scatter(np.rad2deg(target[:, 0]), np.rad2deg(prediction[:, 0]),
                       s=9, alpha=0.35, color="#377eb8")
    lim_deg = np.rad2deg(limit) * 1.05
    axes[0, 0].plot([-lim_deg, lim_deg], [-lim_deg, lim_deg], "k--", lw=1)
    axes[0, 0].set(xlabel="Expert first control [deg]",
                   ylabel="Flow first control [deg]", title="First-action agreement")

    step = np.arange(cfg.horizon)
    axes[0, 1].plot(step, np.rad2deg(target[example]), "o-", label="Expert")
    axes[0, 1].plot(step, np.rad2deg(prediction[example]), "s--", label="Flow")
    axes[0, 1].set(xlabel="Horizon step", ylabel="Steering [deg]",
                   title="Largest-control held-out example")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].bar(step, np.rad2deg(horizon_mae), color="#d95f02")
    axes[1, 0].set(xlabel="Horizon step", ylabel="MAE [deg]",
                   title="Error across prediction horizon")
    axes[1, 1].hist(np.rad2deg(residual.ravel()), bins=50, color="#1b9e77", alpha=0.85)
    axes[1, 1].set(xlabel="Flow − expert [deg]", ylabel="Count",
                   title="All control residuals")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Seed {args.seed}: direct held-out control prediction (no rollout)")
    path = output / "direct_control_prediction.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)

    print(f"test examples={target.shape[0]}")
    for row in rows:
        print(row)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
