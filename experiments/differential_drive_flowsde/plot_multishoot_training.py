"""Plot all optimized multiple-shooting training objectives."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def run(args):
    history = json.loads((args.result / "joint_stage" / "history.json").read_text())
    drift = [r for r in history if r["stage"] == "drift"]
    joint = [r for r in history if r["stage"] == "joint"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot([r["stage_epoch"] for r in drift], [r["validation_drift_mse"] for r in drift], label="drift MSE")
    axes[0, 0].plot([r["stage_epoch"] for r in drift], [r.get("validation_continuity_mse", r["validation_shooting_mse"]) for r in drift], label="5-step continuity MSE")
    axes[0, 0].set(title="Stage 1 validation objectives", xlabel="epoch", ylabel="normalized score"); axes[0, 0].legend()
    axes[0, 1].plot([r["stage_epoch"] for r in joint], [r["train_nll"] for r in joint], label="train")
    axes[0, 1].plot([r["stage_epoch"] for r in joint], [r["validation_nll"] for r in joint], label="validation")
    axes[0, 1].set(title="Transition likelihood", xlabel="joint epoch", ylabel="NLL"); axes[0, 1].legend()
    axes[1, 0].plot([r["stage_epoch"] for r in joint], [r.get("validation_continuity_mse", r["validation_shooting_mse"]) for r in joint], label="5-step continuity MSE")
    axes[1, 0].plot([r["stage_epoch"] for r in joint], [r["validation_energy_score"] for r in joint], label="whole-path energy score")
    axes[1, 0].set(title="Stage 2 path objectives", xlabel="joint epoch", ylabel="normalized score"); axes[1, 0].legend()
    axes[1, 1].plot([r["global_epoch"] for r in history], [r["learning_rate"] for r in history])
    axes[1, 1].axvline(len(drift), color="k", ls="--", label="joint stage")
    axes[1, 1].set(title="Warmup-cosine schedule", xlabel="global epoch", ylabel="learning rate"); axes[1, 1].legend()
    figure.tight_layout(); args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=180); figure.savefig(args.output.with_suffix(".pdf")); plt.close(figure)


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--result", type=Path, default=Path("results/results_optimal_control/differential_drive/multishoot_seed0"))
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/differential_drive/multishoot_seed0/evaluation/multishoot_objectives")); return p


if __name__ == "__main__": run(parser().parse_args())
