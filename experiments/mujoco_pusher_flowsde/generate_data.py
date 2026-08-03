#!/usr/bin/env python3
"""Generate stochastic MuJoCo pusher data for controlled FlowSDE models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


def state_vector(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate((data.qpos.copy(), data.qvel.copy()))


def object_contact_wrench(data: mujoco.MjData, object_dofs: np.ndarray) -> np.ndarray:
    """Return the contact-generated generalized wrench on (x, y, yaw).

    MuJoCo accumulates contact constraint forces in ``qfrc_constraint`` after
    ``mj_step``. The object's three planar DoFs therefore give the net force
    and torque caused by all contacts, excluding ``qfrc_applied``.
    """
    return data.qfrc_constraint[object_dofs].copy()


def object_pose(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.array([
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_x")]],
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_y")]],
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_yaw")]],
    ])


def fingertip_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    return data.xpos[body, :2].copy()


def controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    previous: np.ndarray,
    rng: np.random.Generator,
    exploration: float,
) -> np.ndarray:
    """Operational-space contact controller with smooth torque exploration."""
    obj = object_pose(model, data)
    tip = fingertip_xy(model, data)
    goal = np.array([0.68, 0.0])
    direction = goal - obj[:2]
    direction /= max(np.linalg.norm(direction), 1e-6)
    behind = obj[:2] - 0.12 * direction
    waypoint = behind if np.linalg.norm(tip - behind) > 0.08 else obj[:2] + 0.10 * direction

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body)
    jac = jacp[:2, :3]
    desired_velocity = np.clip(7.0 * (waypoint - tip), -1.2, 1.2)
    desired_qvel = np.linalg.pinv(jac, rcond=1e-3) @ desired_velocity
    torque = 2.8 * (desired_qvel - data.qvel[:3]) - 0.25 * data.qvel[:3]
    proposal = torque + exploration * rng.standard_normal(3)
    smoothed = 0.82 * previous + 0.18 * proposal
    return np.clip(smoothed, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def stage_cost(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    previous_action: np.ndarray,
) -> tuple[float, dict[str, float]]:
    pose = object_pose(model, data)
    goal_cost = 12.0 * np.sum((pose[:2] - np.array([0.68, 0.0])) ** 2) + 0.8 * pose[2] ** 2
    # Penalize approaching either wall while the object crosses the barrier.
    x_gate = np.exp(-0.5 * ((pose[0] - 0.28) / 0.11) ** 2)
    clearance = 0.15 - abs(pose[1])
    obstacle_cost = 18.0 * x_gate * max(0.0, -clearance) ** 2
    control_cost = 0.015 * np.sum(action ** 2)
    smoothness_cost = 0.12 * np.sum((action - previous_action) ** 2)
    terms = {
        "goal_cost": float(goal_cost),
        "obstacle_cost": float(obstacle_cost),
        "control_cost": float(control_cost),
        "smoothness_cost": float(smoothness_cost),
    }
    return float(sum(terms.values())), terms


def reset_episode(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    mujoco.mj_resetData(model, data)
    # Fold the arm so that the fingertip starts behind (negative x side of)
    # the cuboid; a nearly straight reset can accidentally push it backwards.
    data.qpos[:3] = np.array((1.0, -2.0, 1.0)) + rng.normal(0.0, 0.05, 3)
    data.qpos[3:6] = np.array((-0.05, rng.uniform(-0.09, 0.09), rng.uniform(-0.25, 0.25)))
    data.qvel[:] = rng.normal(0.0, 0.01, model.nv)
    mujoco.mj_forward(model, data)


def generate(args: argparse.Namespace) -> dict:
    xml = Path(args.model)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    object_dofs = np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ("object_x", "object_y", "object_yaw")
    ])
    rows: list[dict] = []

    for episode in range(args.episodes):
        reset_episode(model, data, rng)
        action = np.zeros(model.nu)
        disturbance = np.zeros(3)
        regime = 1.0
        for step in range(args.steps):
            if step and step % args.regime_steps == 0:
                regime = rng.choice((-1.5, -0.5, 0.5, 1.5))
            innovation = args.process_sigma * np.array((1.0, 1.0, 0.35)) * rng.standard_normal(3)
            disturbance = args.process_rho * disturbance + np.sqrt(
                1.0 - args.process_rho ** 2
            ) * innovation
            biased_disturbance = disturbance + regime * np.array((0.0, args.lateral_bias, 0.0))

            previous_action = action.copy()
            action = controller(model, data, previous_action, rng, args.exploration)
            before = state_vector(data)
            pose_before = object_pose(model, data)
            cost, terms = stage_cost(model, data, action, previous_action)
            data.ctrl[:] = action
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[object_dofs] = biased_disturbance
            mujoco.mj_step(model, data)
            after = state_vector(data)
            contact_wrench = object_contact_wrench(data, object_dofs)

            rows.append({
                "episode_index": episode,
                "frame_index": step,
                "timestamp": step * model.opt.timestep,
                "observation.state": before.tolist(),
                "action": action.tolist(),
                "next_state": after.tolist(),
                "object_pose": pose_before.tolist(),
                "object_contact_force": contact_wrench.tolist(),
                "object_contact_force_norm": float(np.linalg.norm(contact_wrench)),
                "pseudo_process_noise": biased_disturbance.tolist(),
                "disturbance_regime": float(regime),
                "cost": cost,
                **terms,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    metadata = {
        "model": str(xml),
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "dt": model.opt.timestep,
        "state_dim": model.nq + model.nv,
        "action_dim": model.nu,
        "object_contact_force": {
            "column": "object_contact_force",
            "components": ["force_x", "force_y", "torque_yaw"],
            "units": ["N", "N", "N m"],
            "timing": "constraint force over observation.state -> next_state",
        },
        "pseudo_process_noise": {
            "rho": args.process_rho,
            "sigma": args.process_sigma,
            "lateral_bias": args.lateral_bias,
            "regime_steps": args.regime_steps,
        },
        "cost": {
            "goal_position": [0.68, 0.0],
            "goal_weight": 12.0,
            "yaw_weight": 0.8,
            "obstacle_weight": 18.0,
            "control_weight": 0.015,
            "smoothness_weight": 0.12,
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here / "pusher_passage.xml")
    parser.add_argument("--output", type=Path, default=Path("data/mujoco_pusher/pseudo_process.parquet"))
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration", type=float, default=0.7)
    parser.add_argument("--process-rho", type=float, default=0.97)
    parser.add_argument("--process-sigma", type=float, default=0.35)
    parser.add_argument("--lateral-bias", type=float, default=0.08)
    parser.add_argument("--regime-steps", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(generate(args), indent=2))


if __name__ == "__main__":
    main()
