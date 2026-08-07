"""Observable context plus a 14-D physical dynamics core for the rover."""

import numpy as np

from .env import OffroadRoverEnv


class OffroadRoverCoreEnv(OffroadRoverEnv):
    """State is ``[x, y, core]``; only ``core`` should be learned.

    Core order: psi, v_long, v_lat, yaw_rate, four wheel angular velocities,
    z, roll, pitch, v_z, roll_rate, pitch_rate.
    """

    state_dim, core_dim = 16, 14

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
            self._unwrapped_yaw += (raw_yaw - self._raw_yaw + np.pi) % (2*np.pi) - np.pi
        self._raw_yaw = raw_yaw
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
        body_linear = rotation.T @ self.data.qvel[:3]
        body_angular = rotation.T @ self.data.qvel[3:6]
        core = np.asarray((self._unwrapped_yaw, body_linear[0], body_linear[1],
            body_angular[2], *self.data.qvel[6:10], self.data.qpos[2], roll, pitch,
            body_linear[2], body_angular[0], body_angular[1]), dtype=np.float32)
        return np.concatenate((self.data.qpos[:2].astype(np.float32), core))
