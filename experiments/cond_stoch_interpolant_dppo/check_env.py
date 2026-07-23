"""Machine-readable environment compatibility check."""
from __future__ import annotations
import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="AdroitHandDoor-v1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env)
    obs, info = env.reset(seed=args.seed)
    _, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
    print(json.dumps({
        "env_id": args.env,
        "gymnasium": gym.__version__,
        "gymnasium_robotics": gymnasium_robotics.__version__,
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "max_episode_steps": env.spec.max_episode_steps,
        "reward_sample": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info_keys": sorted(step_info),
    }, indent=2))
    env.close()


if __name__ == "__main__":
    main()

