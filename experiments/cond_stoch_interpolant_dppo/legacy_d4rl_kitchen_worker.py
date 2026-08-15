"""Run the unmodified legacy D4RL Kitchen environment behind a pipe.

This file is intentionally dependency-light: it is launched with the isolated
Python 3.10 D4RL environment while policy inference remains in Python 3.12.
"""
from __future__ import annotations

import contextlib
import pickle
import struct
import sys


def receive(stream):
    size_bytes = stream.read(8)
    if not size_bytes:
        raise EOFError
    size = struct.unpack("!Q", size_bytes)[0]
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("truncated command")
    return pickle.loads(payload)


def send(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    # D4RL and the original Franka environment print during import and model
    # construction. Keep stdout exclusively for the binary protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import gym
        import d4rl  # noqa: F401 - registers kitchen-complete-v0
        import numpy as np

        env = gym.make("kitchen-complete-v0")
    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    initial_tasks = set(env.unwrapped.TASK_ELEMENTS)
    elapsed_steps = 0
    try:
        while True:
            command, payload = receive(source)
            if command == "reset":
                seed = int(payload)
                with contextlib.redirect_stdout(sys.stderr):
                    env.seed(seed)
                    observation = env.reset()
                elapsed_steps = 0
                send(sink, (observation, {"success": False, "completed_tasks": {}}))
            elif command == "step":
                with contextlib.redirect_stdout(sys.stderr):
                    observation, reward, done, info = env.step(
                        np.asarray(payload, dtype=np.float32)
                    )
                elapsed_steps += 1
                remaining = set(env.unwrapped.tasks_to_complete)
                completed = initial_tasks - remaining
                info = dict(info)
                info["completed_tasks"] = {
                    name.replace(" ", "_"): name in completed
                    for name in initial_tasks
                }
                info["success"] = not remaining
                time_limit = bool(info.get("TimeLimit.truncated", False))
                terminated = bool(done and not time_limit)
                truncated = bool(time_limit or (elapsed_steps >= 280 and not terminated))
                send(sink, (observation, reward, terminated, truncated, info))
            elif command == "close":
                with contextlib.redirect_stdout(sys.stderr):
                    env.close()
                send(sink, None)
                return
            else:
                raise ValueError(f"unknown command: {command}")
    except EOFError:
        env.close()


if __name__ == "__main__":
    main()
