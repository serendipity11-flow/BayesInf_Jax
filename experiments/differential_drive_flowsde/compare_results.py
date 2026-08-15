"""Plot baseline versus optimized multiple-shooting FlowSDE metrics."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

NAMES = ("p_x", "p_y", "theta", "v", "omega")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {name: json.loads((path / "evaluation" / "metrics.json").read_text())
               for name, path in (("baseline", args.baseline), ("multiple shooting", args.optimized))}
    history = {name: json.loads((path / "joint_stage" / "history.json").read_text())
               for name, path in (("baseline", args.baseline), ("multiple shooting", args.optimized))}
    summary = {"models": metrics, "relative_change_percent": {}}
    for key in ("mean_position_rmse_m", "mean_position_mae_m"):
        old, new = metrics["baseline"]["rollout"][key], metrics["multiple shooting"]["rollout"][key]
        summary["relative_change_percent"][f"rollout_{key}"] = 100 * (new / old - 1)
    for state in NAMES:
        old, new = metrics["baseline"]["rmse"][state], metrics["multiple shooting"]["rmse"][state]
        summary["relative_change_percent"][f"one_step_rmse_{state}"] = 100 * (new / old - 1)
    (args.output / "comparison_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); positions = np.arange(5); width = .36
    for offset, (label, result) in enumerate(metrics.items()):
        axes[0, 0].bar(positions + (offset - .5) * width, [result["rmse"][n] for n in NAMES], width, label=label)
        axes[0, 1].bar(positions + (offset - .5) * width, [result["coverage_90"][n] for n in NAMES], width, label=label)
    axes[0, 0].set_xticks(positions, NAMES); axes[0, 0].set(title="Held-out one-step RMSE", ylabel="native units"); axes[0, 0].legend()
    axes[0, 1].set_xticks(positions, NAMES); axes[0, 1].axhline(.9, color="k", ls="--", label="nominal 90%")
    axes[0, 1].set(title="Predictive interval calibration", ylabel="90% coverage", ylim=(0, 1)); axes[0, 1].legend()
    rollout_keys = ("mean_position_rmse_m", "particle_position_rmse_m")
    labels = ("mean path", "particles")
    for offset, (model_name, result) in enumerate(metrics.items()):
        axes[1, 0].bar(np.arange(2) + (offset - .5) * width,
                       [result["rollout"][key] for key in rollout_keys], width, label=model_name)
    axes[1, 0].set_xticks(np.arange(2), labels); axes[1, 0].set(title="2-second rollout position RMSE", ylabel="m"); axes[1, 0].legend()
    optimized_joint = [row for row in history["multiple shooting"] if row["stage"] == "joint"]
    axes[1, 1].plot([r["stage_epoch"] for r in optimized_joint], [r["validation_shooting_mse"] for r in optimized_joint], label="shooting MSE")
    axes[1, 1].plot([r["stage_epoch"] for r in optimized_joint], [r["validation_energy_score"] for r in optimized_joint], label="energy score")
    axes[1, 1].set(title="Optimized validation path objectives", xlabel="joint epoch", ylabel="normalized score"); axes[1, 1].legend()
    fig.tight_layout(); fig.savefig(args.output / "baseline_vs_multishoot.png", dpi=180); fig.savefig(args.output / "baseline_vs_multishoot.pdf"); plt.close(fig)
    print(json.dumps(summary["relative_change_percent"], indent=2)); return summary


def parser():
    root = Path("results/results_optimal_control/differential_drive")
    p = argparse.ArgumentParser(); p.add_argument("--baseline", type=Path, default=root / "seed0")
    p.add_argument("--optimized", type=Path, default=root / "multishoot_seed0")
    p.add_argument("--output", type=Path, default=root / "comparison"); return p


if __name__ == "__main__": run(parser().parse_args())
