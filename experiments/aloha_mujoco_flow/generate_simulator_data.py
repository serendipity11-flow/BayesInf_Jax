"""Generate compact, episode-safe training data from Menagerie ALOHA 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mujoco_aloha import (
    DEFAULT_XML,
    CorrelatedJointDisturbance,
    JointDisturbanceConfig,
    MujocoAlohaPlant,
)


def smooth_random_actions(rng, steps, neutral, low, high, scale=0.12, rho=0.96):
    span = high - low
    noise = np.zeros_like(neutral)
    result = []
    for _ in range(steps):
        noise = rho * noise + np.sqrt(1 - rho ** 2) * rng.normal(size=len(neutral))
        result.append(np.clip(neutral + scale * span * noise, low, high))
    return np.asarray(result, dtype=np.float32)


def nominal_actions(rng, steps, neutral, low, high, perturb=0.0):
    time = np.arange(steps)[:, None]
    span = high - low
    frequency = rng.uniform(0.004, 0.018, size=(1, len(neutral)))
    phase = rng.uniform(0, 2 * np.pi, size=(1, len(neutral)))
    target = neutral + 0.10 * span * np.sin(2 * np.pi * frequency * time + phase)
    if perturb:
        target += perturb * span * rng.normal(size=target.shape)
    return np.clip(target, low, high).astype(np.float32)


def cpu_mppi_action(plant, goal, nominal, rng, horizon=8, candidates=24,
                    exploration=0.04, temperature=0.5):
    """Small exact-MuJoCo MPPI update used only to broaden data coverage."""
    span = plant.action_high - plant.action_low
    noise = exploration * span * rng.normal(size=(candidates, horizon, plant.action_dim))
    sequences = np.clip(nominal[None] + noise, plant.action_low, plant.action_high)
    snapshot = plant.copy_full_state()
    costs = np.empty(candidates)
    for candidate in range(candidates):
        plant.restore_full_state(snapshot)
        cost = 0.0
        for action in sequences[candidate]:
            state = plant.step(action)
            cost += np.mean(((state[:14] - goal) / np.maximum(span, 1e-3)) ** 2)
        costs[candidate] = cost
    plant.restore_full_state(snapshot)
    scaled = (costs - costs.min()) / max(costs.std(), 1e-8)
    weights = np.exp(-scaled / temperature); weights /= weights.sum()
    return np.sum(weights[:, None, None] * sequences, axis=0)[0]


def generate(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, episode_meta = [], []
    modes = (["smooth"] * args.smooth_episodes + ["nominal"] * args.nominal_episodes
             + ["mppi_perturbed"] * args.mppi_episodes)
    disturbance_config = JointDisturbanceConfig(
        rho=args.disturbance_rho,
        sigma_z=args.sigma_z,
        sigma_ind=args.sigma_ind,
        innovation=args.innovation,
        mixture_probability=args.mixture_probability,
        mixture_scale=args.mixture_scale,
        student_df=args.student_df,
        sigma_free=args.sigma_free,
        sigma_contact=args.sigma_contact,
        contact_force_reference=args.contact_force_reference,
    )
    for episode_id, mode in enumerate(modes):
        gain = rng.uniform(*args.gain_scale_range)
        damping = rng.uniform(*args.damping_scale_range)
        plant = MujocoAlohaPlant(
            args.xml, args.control_dt, gain_scale=gain, damping_scale=damping
        )
        disturbance = (
            CorrelatedJointDisturbance(disturbance_config, rng)
            if args.joint_disturbance
            else None
        )
        initial_scale = args.initial_noise_fraction * (plant.action_high - plant.action_low)
        state = plant.reset(position_noise=initial_scale, rng=rng)
        if mode == "smooth":
            commands = smooth_random_actions(
                rng, args.steps, plant.neutral_action, plant.action_low, plant.action_high
            )
        else:
            commands = nominal_actions(
                rng, args.steps, plant.neutral_action, plant.action_low, plant.action_high,
                args.perturbation_fraction if mode == "mppi_perturbed" else 0.0,
            )
        plan = np.broadcast_to(commands[0], (args.mppi_horizon, plant.action_dim)).copy()
        goal = commands[min(args.steps - 1, args.mppi_horizon * 2)]
        for step in range(args.steps):
            action = commands[step]
            if mode == "mppi_perturbed" and step % args.mppi_every == 0:
                plan = np.concatenate([plan[1:], plan[-1:]])
                action = cpu_mppi_action(
                    plant, goal, plan, rng, args.mppi_horizon,
                    args.mppi_candidates, args.mppi_exploration,
                )
                plan[0] = action
            contact_intensity = plant.contact_intensity(args.contact_force_reference)
            disturbance_torque = (
                disturbance.sample(contact_intensity)
                if disturbance is not None
                else np.zeros(plant.action_dim, dtype=np.float32)
            )
            rows.append({
                "observation.state": (
                    state[:14] if args.positions_only else state
                ).astype(np.float32, copy=True),
                "action": np.asarray(action, dtype=np.float32),
                "disturbance_torque": disturbance_torque,
                "contact_intensity": np.float32(contact_intensity),
                "episode_index": episode_id, "frame_index": step,
            })
            state = plant.step(action, disturbance_torque)
        # Terminal row keeps the existing episode loader's state/action alignment;
        # its action is discarded by load_episodes.
        rows.append({
            "observation.state": (
                state[:14] if args.positions_only else state
            ).astype(np.float32, copy=True),
            "action": np.asarray(commands[-1], dtype=np.float32),
            "disturbance_torque": np.zeros(plant.action_dim, dtype=np.float32),
            "contact_intensity": np.float32(
                plant.contact_intensity(args.contact_force_reference)
            ),
            "episode_index": episode_id, "frame_index": args.steps,
        })
        episode_meta.append({"episode": episode_id, "mode": mode,
                             "gain_scale": gain, "damping_scale": damping})
        print(f"episode {episode_id + 1}/{len(modes)} mode={mode}")
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    metadata = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "state": (
            "14 actuator-associated qpos"
            if args.positions_only
            else "14 actuator-associated qpos followed by 14 qvel"
        ),
        "action": "14 Menagerie joint-position actuator controls",
        "disturbance_torque": (
            "14 generalized joint torques subtracted through qfrc_applied before mj_step"
        ),
        "disturbance_modes": disturbance.L.tolist() if args.joint_disturbance else None,
        "disturbance": {
            **vars(disturbance_config),
            "enabled": bool(args.joint_disturbance),
            "contact_scale": "sigma_free + r*(sigma_contact-sigma_free)",
        },
        "episodes": episode_meta,
    }
    with args.output.with_suffix(".json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved {len(modes)} episodes to {args.output}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--output", type=Path, default=Path("data/aloha_mujoco/trajectories.parquet"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-dt", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--smooth-episodes", type=int, default=48)
    p.add_argument("--nominal-episodes", type=int, default=24)
    p.add_argument("--mppi-episodes", type=int, default=8)
    p.add_argument(
        "--positions-only",
        action="store_true",
        help="Export only the 14 actuator joint positions, omitting qvel.",
    )
    p.add_argument("--initial-noise-fraction", type=float, default=0.03)
    p.add_argument("--perturbation-fraction", type=float, default=0.015)
    p.add_argument("--gain-scale-range", type=float, nargs=2, default=(1.0, 1.0))
    p.add_argument("--damping-scale-range", type=float, nargs=2, default=(1.0, 1.0))
    p.add_argument("--mppi-horizon", type=int, default=8)
    p.add_argument("--mppi-candidates", type=int, default=24)
    p.add_argument("--mppi-every", type=int, default=5)
    p.add_argument("--mppi-exploration", type=float, default=0.04)
    p.add_argument("--joint-disturbance", action="store_true")
    p.add_argument(
        "--innovation",
        choices=("gaussian", "gaussian_mixture", "student_t"),
        default="gaussian_mixture",
    )
    p.add_argument("--disturbance-rho", type=float, default=0.98)
    p.add_argument("--sigma-z", type=float, default=1.0)
    p.add_argument("--sigma-ind", type=float, default=0.05)
    p.add_argument("--mixture-probability", type=float, default=0.05)
    p.add_argument("--mixture-scale", type=float, default=5.0)
    p.add_argument("--student-df", type=float, default=4.0)
    p.add_argument("--sigma-free", type=float, default=1.0)
    p.add_argument("--sigma-contact", type=float, default=3.0)
    p.add_argument("--contact-force-reference", type=float, default=50.0)
    return p


if __name__ == "__main__":
    generate(parser().parse_args())
