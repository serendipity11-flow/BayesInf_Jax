"""Generate the 100 Hz, wheel-resolved off-road FlowSDE dataset.

The output stores trajectories (rather than materialising overlapping windows) and
stores valid ``window_start`` rows.  A training loader can recover a window as
``state[i:i + horizon + 1]`` and ``action[i:i + horizon + 1]``.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .env import OffroadConfig, OffroadRoverEnv


STATE_ORDER = ("p_x", "p_y", "theta", "v_x", "v_y", "omega",
               "omega_fl", "omega_fr", "omega_rl", "omega_rr")
ACTION_ORDER = ("throttle_brake", "steer")
POLICIES = ("ou", "chirp", "racing")
SPLITS = ("train", "validation", "test")
POLICY_FRACTIONS = np.asarray((0.4, 0.2, 0.4))


def _policy_counts(count):
    raw = POLICY_FRACTIONS * count
    result = np.floor(raw).astype(int)
    result[np.argsort(raw - result)[::-1][:count - result.sum()]] += 1
    return result


class HybridPolicy:
    def __init__(self, name, rng, dt, delta_max):
        self.name, self.rng, self.dt, self.delta_max = name, rng, dt, delta_max
        self.command = np.zeros(2)
        self.phase = rng.uniform(0, 2 * np.pi)
        self.throttle = rng.uniform(0.35, 0.95)
        self.chirp_f0 = rng.uniform(0.2, 0.8)
        self.chirp_f1 = rng.uniform(3.0, 8.0)
        self.radius = rng.uniform(3.0, 7.0)
        self.direction = rng.choice((-1.0, 1.0))

    def __call__(self, frame, state, steps):
        if self.name == "ou":
            # OU stationary standard deviations are approximately 0.45 and 0.3 rad.
            mean = np.asarray((0.35, 0.0))
            theta, sigma = np.asarray((1.0, 1.6)), np.asarray((0.64, 0.54))
            self.command += theta * (mean - self.command) * self.dt
            self.command += sigma * np.sqrt(self.dt) * self.rng.normal(size=2)
            action = self.command
        elif self.name == "chirp":
            t, duration = frame * self.dt, steps * self.dt
            rate = self.chirp_f0 + (self.chirp_f1 - self.chirp_f0) * t / duration
            phase = self.phase + 2 * np.pi * (self.chirp_f0 * t +
                0.5 * (self.chirp_f1 - self.chirp_f0) * t * t / duration)
            action = np.asarray((self.throttle + 0.12 * np.sin(2 * np.pi * 0.25 * t),
                                 0.9 * self.delta_max * np.sin(phase)))
        else:
            # A lightweight racing controller tracks an episode-specific circle.
            x, y, yaw, vx, vy, yaw_rate = state[:6]
            target_yaw = np.arctan2(self.direction * x, -self.direction * y)
            yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            radial_error = np.hypot(x, y) - self.radius
            steer = 1.25 * yaw_error + 0.18 * self.direction * radial_error - 0.08 * yaw_rate
            target_speed = 3.0 + 1.5 * self.rng.random()
            throttle = 0.48 + 0.22 * (target_speed - vx) - 0.04 * abs(vy)
            action = np.asarray((throttle, steer))
        return np.clip(action, (-1.0, -self.delta_max), (1.0, self.delta_max)).astype(np.float32)


def _episode_plan(counts, seed):
    rng, rows, episode = np.random.default_rng(seed), [], 0
    for split_id, count in enumerate(counts):
        policies = np.repeat(np.arange(3), _policy_counts(count))
        rng.shuffle(policies)
        for policy_id in policies:
            rows.append((episode, split_id, int(policy_id), 1 if split_id == 2 else 0,
                         int(rng.integers(0, np.iinfo(np.uint32).max))))
            episode += 1
    return rows


def _datasets(handle, frames, trajectories, windows, chunk_frames):
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {
        "state": handle.create_dataset("state", (frames, 10), "f4", chunks=(min(chunk_frames, frames), 10), **compression),
        "action": handle.create_dataset("action", (frames, 2), "f4", chunks=(min(chunk_frames, frames), 2), **compression),
        "next_state": handle.create_dataset("next_state", (frames, 10), "f4", chunks=(min(chunk_frames, frames), 10), **compression),
        "trajectory_id": handle.create_dataset("trajectory_id", (frames,), "i4", chunks=(min(chunk_frames, frames),), **compression),
        "split": handle.create_dataset("split", (frames,), "u1", chunks=(min(chunk_frames, frames),), **compression),
        "policy": handle.create_dataset("policy", (frames,), "u1", chunks=(min(chunk_frames, frames),), **compression),
        "trajectory_start": handle.create_dataset("trajectory_start", (trajectories,), "i8"),
        "window_start": handle.create_dataset("window_start", (windows,), "i8", **compression),
        "window_split": handle.create_dataset("window_split", (windows,), "u1", **compression),
    }


def collect(args):
    if sum(args.split_trajectories) != args.trajectories:
        raise ValueError("--split-trajectories must sum to --trajectories")
    if args.steps <= args.horizon:
        raise ValueError("steps must be greater than horizon")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    plan = _episode_plan(args.split_trajectories, args.seed)
    windows_per_trajectory = (args.steps - args.horizon - 1) // args.stride + 1
    total_frames = args.trajectories * args.steps
    config = OffroadConfig(control_dt=args.dt, delta_max=args.delta_max)
    with h5py.File(temporary, "w") as handle:
        data = _datasets(handle, total_frames, args.trajectories,
                         args.trajectories * windows_per_trajectory, args.chunk_frames)
        window_cursor = 0
        for episode, split_id, policy_id, map_id, episode_seed in plan:
            env = OffroadRoverEnv(config=config, seed=episode_seed, terrain_map=map_id)
            rng = np.random.default_rng(episode_seed)
            pose = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-np.pi, np.pi))
            state = env.reset(pose=pose, randomize=False)
            policy = HybridPolicy(POLICIES[policy_id], rng, args.dt, args.delta_max)
            start, stop = episode * args.steps, (episode + 1) * args.steps
            states = np.empty((args.steps, 10), np.float32)
            actions = np.empty((args.steps, 2), np.float32)
            next_states = np.empty((args.steps, 10), np.float32)
            for frame in range(args.steps):
                action = policy(frame, state, args.steps)
                next_state = env.step(action)
                states[frame], actions[frame], next_states[frame] = state, action, next_state
                state = next_state
            data["state"][start:stop], data["action"][start:stop] = states, actions
            data["next_state"][start:stop] = next_states
            data["trajectory_id"][start:stop] = episode
            data["split"][start:stop], data["policy"][start:stop] = split_id, policy_id
            data["trajectory_start"][episode] = start
            starts = start + np.arange(0, args.steps - args.horizon, args.stride)
            end = window_cursor + len(starts)
            data["window_start"][window_cursor:end] = starts
            data["window_split"][window_cursor:end] = split_id
            window_cursor = end
            if args.progress_every and (episode + 1) % args.progress_every == 0:
                print(f"generated {episode + 1}/{args.trajectories} trajectories", flush=True)
        for key, value in {
            "dt": args.dt, "horizon": args.horizon, "stride": args.stride,
            "state_order": json.dumps(STATE_ORDER), "action_order": json.dumps(ACTION_ORDER),
            "split_names": json.dumps(SPLITS), "policy_names": json.dumps(POLICIES),
            "terrain_maps": json.dumps(("A", "B")), "seed": args.seed,
        }.items():
            handle.attrs[key] = value
    temporary.replace(args.output)
    metadata = {
        "format_version": 2, "trajectories": args.trajectories, "steps_per_trajectory": args.steps,
        "frames": total_frames, "duration_seconds": args.steps * args.dt, "dt": args.dt,
        "state_dim": 10, "state_order": STATE_ORDER, "action_dim": 2, "action_order": ACTION_ORDER,
        "delta_max": args.delta_max, "splits": dict(zip(SPLITS, args.split_trajectories)),
        "split_terrain": {"train": "A", "validation": "A", "test": "B"},
        "policies": dict(zip(POLICIES, POLICY_FRACTIONS.tolist())),
        "terrain": {"map_A": "4 Gaussian friction patches", "map_B": "unseen 4-patch map with sharper gradients"},
        "windows": {"horizon": args.horizon, "stride": args.stride,
                    "count": args.trajectories * windows_per_trajectory}, "seed": args.seed,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/offroad_flowsde/offroad_v2.h5"))
    p.add_argument("--trajectories", type=int, default=2000)
    p.add_argument("--split-trajectories", type=int, nargs=3, default=(1400, 300, 300),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--delta-max", type=float, default=0.6)
    p.add_argument("--chunk-frames", type=int, default=8192)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    return p


if __name__ == "__main__":
    print(json.dumps(collect(parser().parse_args()), indent=2))
