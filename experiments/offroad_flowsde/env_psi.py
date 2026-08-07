"""Seven-state off-road environment with continuous, unwrapped yaw psi."""

import mujoco
import numpy as np

from .env import OffroadRoverEnv


class OffroadRoverPsiEnv(OffroadRoverEnv):
    """State: ``[x, y, psi, v_long, v_lat, yaw_rate, wheel_slip]``."""

    state_dim = 7

    def reset(self, pose=(0.0, 0.0, 0.0), randomize=True):
        self._raw_yaw = None
        self._unwrapped_yaw = float(pose[2])
        return super().reset(pose=pose, randomize=randomize)

    def state(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        raw_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if self._raw_yaw is None:
            self._unwrapped_yaw = raw_yaw
        else:
            change = (raw_yaw - self._raw_yaw + np.pi) % (2 * np.pi) - np.pi
            self._unwrapped_yaw += change
        self._raw_yaw = raw_yaw
        world_velocity = self.data.qvel[:2]
        forward, lateral = rotation[:2, 0], rotation[:2, 1]
        v_long, v_lat = world_velocity @ forward, world_velocity @ lateral
        wheel_speed = self.data.qvel[6:10]
        surface_speed = 0.10 * np.mean(np.abs(wheel_speed))
        slip = (surface_speed - abs(v_long)) / max(surface_speed, abs(v_long), 0.1)
        return np.asarray((self.data.qpos[0], self.data.qpos[1], self._unwrapped_yaw,
                           v_long, v_lat, self.data.qvel[5], slip), dtype=np.float32)
