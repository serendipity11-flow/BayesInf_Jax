"""Generate trajectory-safe train/validation/test data for the 8-D robot."""
import argparse, json
from pathlib import Path
import h5py
import numpy as np
from .env import DifferentialDriveConfig, DifferentialDriveEnv

BASE_STATE_ORDER = ("p_x", "p_y", "theta", "v_forward", "omega")
EXTENDED_STATE_ORDER = ("p_x", "p_y", "theta", "v_forward", "v_lateral", "omega",
                        "wheel_left", "wheel_right")
FULL_STATE_ORDER = ("p_x", "p_y", "theta", "v_forward", "v_lateral", "v_vertical",
                    "roll", "pitch", "roll_rate", "pitch_rate", "omega",
                    "wheel_left", "wheel_right", "actuator_left", "actuator_right",
                    "slip_left", "slip_right")
COMPACT_PHYSICAL_STATE_ORDER = ("p_x", "p_y", "p_z", "cos_theta", "sin_theta",
                                "v_forward", "v_lateral", "v_vertical", "omega",
                                "wheel_left", "wheel_right", "slip_left", "slip_right")
ACTION_ORDER = ("v_cmd", "omega_cmd")


def collect(args):
    if sum(args.split_trajectories) != args.trajectories:
        raise ValueError("split counts must sum to trajectories")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = args.trajectories * args.steps
    state_order = (COMPACT_PHYSICAL_STATE_ORDER if args.compact_physical_state else
                   FULL_STATE_ORDER if args.full_state else
                   EXTENDED_STATE_ORDER if args.extended_state else BASE_STATE_ORDER)
    state_dim = len(state_order)
    arrays = {"state": np.empty((total, state_dim), np.float32), "action": np.empty((total, 2), np.float32),
              "next_state": np.empty((total, state_dim), np.float32), "friction": np.empty(total, np.float32),
              "target": np.empty((total, 2), np.float32),
              "trajectory_id": np.empty(total, np.int32), "split": np.empty(total, np.uint8)}
    rng = np.random.default_rng(args.seed)
    boundaries = np.cumsum((0, *args.split_trajectories))
    for episode in range(args.trajectories):
        split = int(np.searchsorted(boundaries[1:], episode, side="right"))
        env = DifferentialDriveEnv(DifferentialDriveConfig(
            control_dt=args.dt, extended_state=args.extended_state,
            full_state=args.full_state,
            compact_physical_state=args.compact_physical_state), int(rng.integers(2**32)))
        state = env.reset((rng.uniform(-4, 4), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)))
        command = np.zeros(2)
        target = rng.uniform((-4, -3), (4, 3))
        start = episode * args.steps
        for k in range(args.steps):
            if k % args.waypoint_steps == 0 or np.linalg.norm(target - state[:2]) < .35:
                target = rng.uniform((-4, -3), (4, 3))
            bearing = np.arctan2(target[1] - state[1], target[0] - state[0])
            theta = (np.arctan2(state[4], state[3]) if args.compact_physical_state
                     else state[2])
            heading_error = np.arctan2(np.sin(bearing - theta), np.cos(bearing - theta))
            desired = np.asarray((np.clip(1.1 * np.linalg.norm(target - state[:2]), 0, 1.8),
                                  np.clip(2.2 * heading_error, -2.5, 2.5)))
            # Correlated excitation preserves broad dynamics coverage around a
            # stabilizing navigation policy.
            command += args.command_smoothing * (desired - command)
            command += np.asarray((.12, .25)) * rng.normal(size=2)
            command = np.clip(command, (0.0, -3.0), (2.2, 3.0)).astype(np.float32)
            nxt = env.step(command)
            i = start + k
            arrays["state"][i], arrays["action"][i], arrays["next_state"][i] = state, command, nxt
            arrays["friction"][i], arrays["trajectory_id"][i], arrays["split"][i] = env.hidden_context["friction"], episode, split
            arrays["target"][i] = target
            state = nxt
    with h5py.File(args.output, "w") as f:
        for name, value in arrays.items(): f.create_dataset(name, data=value, compression="gzip")
        f.attrs["dt"] = args.dt; f.attrs["state_order"] = json.dumps(state_order); f.attrs["action_order"] = json.dumps(ACTION_ORDER)
        f.attrs["friction_is_observed"] = False; f.attrs["task"] = "smooth_terrain_waypoint_navigation"
    metadata = {"state_order": state_order, "action_order": ACTION_ORDER, "frames": total,
                "splits": dict(zip(("train", "validation", "test"), args.split_trajectories)), "dt": args.dt,
                "task": "smooth_terrain_waypoint_navigation", "waypoint_steps": args.waypoint_steps}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=Path("data/differential_drive/smooth_terrain_navigation.h5"))
    p.add_argument("--trajectories", type=int, default=300); p.add_argument("--split-trajectories", type=int, nargs=3, default=(210, 45, 45))
    p.add_argument("--steps", type=int, default=500); p.add_argument("--dt", type=float, default=.02); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--waypoint-steps", type=int, default=125)
    p.add_argument("--command-smoothing", type=float, default=.08)
    p.add_argument("--extended-state", action="store_true",
                   help="Observe lateral velocity and both wheel angular velocities")
    p.add_argument("--full-state", action="store_true",
                   help="Also observe body attitude/rates, actuator forces, and wheel slip")
    p.add_argument("--compact-physical-state", action="store_true",
                   help="Use the selected 13D pose, velocity, wheel, and slip state")
    return p


if __name__ == "__main__": print(json.dumps(collect(parser().parse_args()), indent=2))
