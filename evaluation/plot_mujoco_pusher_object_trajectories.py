"""Compare baseline and contact-supervised FlowSDE object trajectories."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.aloha_optimal_control.data import load_episodes
from experiments.aloha_optimal_control.train_flow_sde import build_model


OBJECT_INDICES = (3, 4, 5, 9, 10, 11)
OBJECT_LABELS = ("object x [m]", "object y [m]", "object yaw [rad]",
                 "object vx [m/s]", "object vy [m/s]", "object yaw rate [rad/s]")


def load_rollout(config_path, data_path, episode_id, particles, seed):
    config = json.loads(config_path.read_text())
    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(data_path)
    matches = np.flatnonzero(episode_ids == episode_id)
    if len(matches) != 1:
        raise ValueError(f"episode {episode_id} is not uniquely present")
    index = int(matches[0])
    stats = np.load(config["normalization"])
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    u_mean, u_std = stats["u_mean"], stats["u_std"]
    model = build_model(SimpleNamespace(**config), state_dim, control_dim)
    model = eqx.tree_deserialise_leaves(Path(config["checkpoint"]), model)
    x0 = jnp.asarray((episodes[index][0] - x_mean) / x_std)
    controls = jnp.asarray((actions[index] - u_mean) / u_std)
    samples_n = eqx.filter_jit(model.sample_rollout)(
        x0, controls, jr.PRNGKey(seed), particles,
        diffusion_temperature=1.0,
    )
    samples = (np.asarray(samples_n) * x_std[None, None]
               + x_mean[None, None])
    return episodes[index][1:], samples


def main(args):
    baseline_config = json.loads(args.baseline_config.read_text())
    contact_config = json.loads(args.contact_config.read_text())
    baseline_ids = baseline_config["test_episode_ids"]
    contact_ids = contact_config["test_episode_ids"]
    if baseline_ids != contact_ids:
        raise ValueError("baseline and contact checkpoints use different test splits")
    episode_id = baseline_ids[0] if args.episode is None else args.episode
    truth, baseline = load_rollout(
        args.baseline_config, args.data, episode_id, args.particles, args.seed
    )
    contact_truth, contact = load_rollout(
        args.contact_config, args.data, episode_id, args.particles, args.seed
    )
    np.testing.assert_allclose(truth, contact_truth)

    frame = pd.read_parquet(
        args.data,
        columns=["episode_index", "frame_index", "object_contact_force"],
    )
    episode = frame[frame.episode_index == episode_id].sort_values("frame_index")
    wrench = np.linalg.norm(np.stack(episode.object_contact_force), axis=-1)
    seconds = (np.arange(len(truth)) + 1) * baseline_config["dt"]
    baseline_mean = baseline.mean(0)
    contact_mean = contact.mean(0)
    baseline_lo, baseline_hi = np.quantile(baseline, [0.05, 0.95], axis=0)
    contact_lo, contact_hi = np.quantile(contact, [0.05, 0.95], axis=0)

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=False)
    for axis, state_index, label in zip(axes.flat[:6], OBJECT_INDICES, OBJECT_LABELS):
        axis.fill_between(seconds, baseline_lo[:, state_index],
                          baseline_hi[:, state_index], color="C0", alpha=0.14)
        axis.fill_between(seconds, contact_lo[:, state_index],
                          contact_hi[:, state_index], color="C1", alpha=0.14)
        axis.plot(seconds, truth[:, state_index], color="black", lw=1.5,
                  label="truth")
        axis.plot(seconds, baseline_mean[:, state_index], color="C0", lw=1.1,
                  label="baseline mean")
        axis.plot(seconds, contact_mean[:, state_index], color="C1", lw=1.1,
                  label="contact mean")
        axis.set(ylabel=label, xlabel="time [s]")
        axis.grid(alpha=0.2)

    axes[3, 0].plot(seconds, wrench, color="C3", lw=1.1)
    axes[3, 0].fill_between(seconds, 0, wrench, color="C3", alpha=0.15)
    axes[3, 0].set(xlabel="time [s]", ylabel="wrench norm", title="Measured contact")
    axes[3, 0].grid(alpha=0.2)

    axes[3, 1].plot(truth[:, 3], truth[:, 4], color="black", lw=1.8,
                    label="truth")
    axes[3, 1].plot(baseline_mean[:, 3], baseline_mean[:, 4], color="C0",
                    lw=1.2, label="baseline mean")
    axes[3, 1].plot(contact_mean[:, 3], contact_mean[:, 4], color="C1",
                    lw=1.2, label="contact mean")
    axes[3, 1].scatter(truth[0, 3], truth[0, 4], color="black", s=25, zorder=3)
    axes[3, 1].set(xlabel="object x [m]", ylabel="object y [m]",
                   title="Planar object path", aspect="equal")
    axes[3, 1].grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=9)
    axes[3, 1].legend(fontsize=9)
    fig.suptitle(
        f"Held-out episode {episode_id}: baseline vs contact-supervised FlowSDE\n"
        "shading = 90% particle interval",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez(
        args.output.with_suffix(".npz"), episode_id=episode_id, time=seconds,
        truth=truth[:, OBJECT_INDICES], baseline_samples=baseline[:, :, OBJECT_INDICES],
        contact_samples=contact[:, :, OBJECT_INDICES], contact_wrench_norm=wrench,
    )
    print(args.output)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--baseline-config", type=Path, required=True)
    p.add_argument("--contact-config", type=Path, required=True)
    p.add_argument("--episode", type=int)
    p.add_argument("--particles", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
