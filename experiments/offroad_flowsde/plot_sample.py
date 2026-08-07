"""Plot one held-out off-road trajectory with every state and control input."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STATE_NAMES = ("x [m]", "y [m]", "psi [rad]", "v_long [m/s]", "v_lat [m/s]",
               "yaw_rate [rad/s]", "wheel_slip [-]")


def run(args):
    with np.load(args.trajectory_data) as result:
        truth = result["truth"]
        mean = result["recursive_mean"]
        particles = result["recursive_particles"]
        ids = result["test_episode_ids"]
    index = int(np.flatnonzero(ids == args.episode)[0]) if args.episode is not None else 0
    if index >= particles.shape[0]:
        raise ValueError(f"particle paths were saved only for episodes {ids[:particles.shape[0]].tolist()}")
    episode_id = int(ids[index])
    with np.load(args.data) as data:
        mask = data["episode_index"] == episode_id
        controls = data["action"][mask][:truth.shape[1] - 1]
    time = np.arange(truth.shape[1]) * args.dt
    control_time = time[:-1]
    lo, hi = np.quantile(particles[index], (.05, .95), axis=0)

    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    axes = axes.ravel()
    ax = axes[0]
    for particle in particles[index]:
        ax.plot(particle[:, 0], particle[:, 1], color="tab:blue", alpha=.08, lw=.6)
    ax.plot(truth[index, :, 0], truth[index, :, 1], color="k", lw=2, label="MuJoCo")
    ax.plot(mean[index, :, 0], mean[index, :, 1], color="tab:orange", lw=2, label="FlowSDE zero-noise")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="XY trajectory"); ax.legend(); ax.grid(alpha=.25)

    for state_index, name in enumerate(STATE_NAMES):
        ax = axes[state_index + 1]
        ax.fill_between(time, lo[:, state_index], hi[:, state_index], color="tab:blue", alpha=.18,
                        label="FlowSDE 90%" if state_index == 0 else None)
        ax.plot(time, truth[index, :, state_index], color="k", lw=1.8, label="MuJoCo")
        ax.plot(time, mean[index, :, state_index], color="tab:orange", lw=1.6, label="FlowSDE zero-noise")
        ax.set(xlabel="time [s]", ylabel=name, title=name.split(" [")[0]); ax.grid(alpha=.25)
        if state_index == 0: ax.legend(fontsize=8)

    for action_index, name in enumerate(("throttle", "steering")):
        ax = axes[8 + action_index]
        ax.step(control_time, controls[:, action_index], where="post", color="tab:purple", lw=1.6)
        ax.set(xlabel="time [s]", ylabel=name, title=f"input: {name}", ylim=(-1.08, 1.08)); ax.grid(alpha=.25)
    axes[10].axis("off"); axes[11].axis("off")
    fig.suptitle(f"Off-road FlowSDE held-out sample — episode {episode_id}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180); plt.close(fig)
    print(args.output)


def parser():
    root = Path("results/results_optimal_control/offroad_flowsde/seg10_stride10_tw1_energy0p1_seed0/evaluation")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-data", type=Path, default=root / "trajectory_data.npz")
    p.add_argument("--data", type=Path, default=Path("data/offroad_flowsde/transitions_psi.npz"))
    p.add_argument("--episode", type=int, default=221)
    p.add_argument("--dt", type=float, default=.05)
    p.add_argument("--output", type=Path, default=root / "sample_episode_221_states_inputs.png")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
