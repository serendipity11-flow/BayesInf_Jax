"""Compare direct XY states with trajectories reconstructed from psi and body velocity."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def reconstruct(states, dt):
    """Trapezoid-integrate world velocity derived from psi, v_long and v_lat."""
    psi, v_long, v_lat = states[..., 2], states[..., 3], states[..., 4]
    vx = v_long * np.cos(psi) - v_lat * np.sin(psi)
    vy = v_long * np.sin(psi) + v_lat * np.cos(psi)
    velocity = np.stack((vx, vy), axis=-1)
    increments = .5 * dt * (velocity[..., 1:, :] + velocity[..., :-1, :])
    return np.concatenate((states[..., :1, :2],
        states[..., :1, :2] + np.cumsum(increments, axis=-2)), axis=-2)


def run(args):
    with np.load(args.trajectory_data) as result:
        truth = result["truth"]
        prediction = result["recursive_mean"]
        episode_ids = result["test_episode_ids"]
    count = min(args.episodes, len(episode_ids))
    true_reconstruction = reconstruct(truth[:count], args.dt)
    predicted_reconstruction = reconstruct(prediction[:count], args.dt)
    true_gap = np.linalg.norm(true_reconstruction - truth[:count, :, :2], axis=-1)
    predicted_gap = np.linalg.norm(predicted_reconstruction - prediction[:count, :, :2], axis=-1)

    fig, axes = plt.subplots(2, count, figsize=(5 * count, 8.5), squeeze=False)
    time = np.arange(truth.shape[1]) * args.dt
    for index in range(count):
        ax = axes[0, index]
        ax.plot(truth[index, :, 0], truth[index, :, 1], color="k", lw=2,
                label="MuJoCo direct x,y")
        ax.plot(true_reconstruction[index, :, 0], true_reconstruction[index, :, 1],
                color="tab:green", lw=1.8, ls="--", label="MuJoCo reconstructed")
        ax.plot(prediction[index, :, 0], prediction[index, :, 1], color="tab:orange",
                lw=2, label="FlowSDE direct x,y")
        ax.plot(predicted_reconstruction[index, :, 0], predicted_reconstruction[index, :, 1],
                color="tab:blue", lw=1.8, ls="--", label="FlowSDE reconstructed")
        ax.set(title=f"episode {episode_ids[index]}", xlabel="x [m]", ylabel="y [m]")
        ax.grid(alpha=.25); ax.legend(fontsize=8)

        ax = axes[1, index]
        ax.plot(time, true_gap[index], color="tab:green", label="MuJoCo consistency gap")
        ax.plot(time, predicted_gap[index], color="tab:red", label="FlowSDE consistency gap")
        ax.set(title="kinematic reconstruction error", xlabel="time [s]",
               ylabel="|| reconstructed xy - state xy || [m]")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Trajectory reconstruction from ψ, longitudinal and lateral velocity", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180); plt.close(fig)
    metrics = {
        int(episode_ids[i]): {
            "mujoco_final_consistency_gap_m": float(true_gap[i, -1]),
            "flowsde_final_consistency_gap_m": float(predicted_gap[i, -1]),
        } for i in range(count)
    }
    print(metrics)


def parser():
    root = Path("results/results_optimal_control/offroad_flowsde/seg10_stride10_tw1_energy0_seed0/evaluation")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-data", type=Path, default=root / "trajectory_data.npz")
    p.add_argument("--output", type=Path, default=root / "state_reconstructed_trajectories.png")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--dt", type=float, default=.05)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
