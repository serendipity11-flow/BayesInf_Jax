"""MuJoCo differential-drive plant on a smooth heterogeneous traction field."""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class DifferentialDriveConfig:
    control_dt: float = 0.02
    wheel_radius: float = 0.075
    track_width: float = 0.42
    friction_mean: float = 0.72
    friction_amplitude: float = 0.16
    friction_min: float = 0.48
    velocity_noise: float = 0.035
    yaw_noise: float = 0.05
    aggressive_threshold: float = 0.7
    aggressive_gain: float = 0.16
    extended_state: bool = False
    full_state: bool = False
    compact_physical_state: bool = False


class DifferentialDriveEnv:
    """Observed state optionally includes lateral and wheel velocities."""
    state_dim, action_dim = 5, 2

    def __init__(self, config=DifferentialDriveConfig(), seed=0):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("robot.xml")))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        ratio = config.control_dt / self.model.opt.timestep
        self.substeps = int(round(ratio))
        if self.substeps < 1 or not np.isclose(ratio, self.substeps):
            raise ValueError("control_dt must be an integer multiple of the MuJoCo timestep")
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_x_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")]
        self.base_y_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")]
        self.base_yaw_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")]
        self.base_x_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")]
        self.base_y_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")]
        self.base_yaw_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")]
        self.left_wheel_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel")]
        self.right_wheel_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel")]
        self.floor_ids = (mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_floor"),)

    def reset(self, pose=(-2.0, 0.0, 0.0)):
        mujoco.mj_resetData(self.model, self.data)
        x, y, yaw = pose
        self.data.qpos[self.base_x_qpos] = x
        self.data.qpos[self.base_y_qpos] = y
        self.data.qpos[self.base_yaw_qpos] = yaw
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        planar_velocity = self.data.qvel[[self.base_x_dof, self.base_y_dof]]
        v_forward = float(planar_velocity @ rotation[:2, 0])
        v_lateral = float(planar_velocity @ rotation[:2, 1])
        base = (self.data.qpos[0], self.data.qpos[1], theta, v_forward)
        if self.config.compact_physical_state:
            wheel_left = float(self.data.qvel[self.left_wheel_dof])
            wheel_right = float(self.data.qvel[self.right_wheel_dof])
            half_track = .5 * self.config.track_width
            yaw_rate = float(self.data.qvel[self.base_yaw_dof])
            slip_left = self.config.wheel_radius * wheel_left - (v_forward - half_track * yaw_rate)
            slip_right = self.config.wheel_radius * wheel_right - (v_forward + half_track * yaw_rate)
            return np.asarray((self.data.xpos[self.body_id, 0], self.data.xpos[self.body_id, 1],
                               self.data.xpos[self.body_id, 2],
                               np.cos(theta), np.sin(theta), v_forward, v_lateral,
                               0.0, yaw_rate, wheel_left, wheel_right,
                               slip_left, slip_right), dtype=np.float32)
        if self.config.full_state:
            # These variables expose body attitude, actuator loading, and tire slip,
            # which determine the otherwise partially observed v/omega acceleration.
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
            angular_body = np.asarray((0.0, 0.0, self.data.qvel[self.base_yaw_dof]))
            wheel_left = float(self.data.qvel[self.left_wheel_dof])
            wheel_right = float(self.data.qvel[self.right_wheel_dof])
            half_track = .5 * self.config.track_width
            slip_left = self.config.wheel_radius * wheel_left - (v_forward - half_track * self.data.qvel[self.base_yaw_dof])
            slip_right = self.config.wheel_radius * wheel_right - (v_forward + half_track * self.data.qvel[self.base_yaw_dof])
            return np.asarray((*base, v_lateral, 0.0, roll, pitch,
                               angular_body[0], angular_body[1], self.data.qvel[self.base_yaw_dof],
                               wheel_left, wheel_right, *self.data.actuator_force,
                               slip_left, slip_right), dtype=np.float32)
        if self.config.extended_state:
            return np.asarray((*base, v_lateral, self.data.qvel[self.base_yaw_dof],
                               self.data.qvel[self.left_wheel_dof],
                               self.data.qvel[self.right_wheel_dof]), dtype=np.float32)
        return np.asarray((*base, self.data.qvel[self.base_yaw_dof]), dtype=np.float32)

    def friction_at(self, x, y=0.0):
        """Continuous multi-scale terrain map; no hidden regime boundary."""
        x, y = float(x), float(y)
        variation = (0.55 * np.sin(0.72 * x) * np.cos(0.51 * y)
                     + 0.30 * np.sin(1.31 * y + 0.37 * x)
                     + 0.15 * np.cos(1.73 * x - 0.44 * y))
        return float(np.clip(self.config.friction_mean + self.config.friction_amplitude * variation,
                             self.config.friction_min, 1.0))

    def step(self, action, stochastic=True):
        v_cmd, omega_cmd = np.asarray(action, dtype=float)
        mu = self.friction_at(self.data.xpos[self.body_id, 0], self.data.xpos[self.body_id, 1])
        for geom_id in self.floor_ids:
            self.model.geom_friction[geom_id] = (mu, .01, .001)
        half_track = .5 * self.config.track_width
        # The wheel hinge axes point along global -y after the 90-degree tire
        # rotation, so negative joint velocity produces positive body-forward motion.
        self.data.ctrl[:] = (-(v_cmd - half_track * omega_cmd) / self.config.wheel_radius,
                             -(v_cmd + half_track * omega_cmd) / self.config.wheel_radius)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        if stochastic:
            severity = max(0.0, abs(v_cmd * omega_cmd) - self.config.aggressive_threshold)
            surface_gain = self.config.friction_mean / mu
            scale = np.sqrt(self.config.control_dt) * (1 + self.config.aggressive_gain * severity) * surface_gain
            rotation = self.data.xmat[self.body_id].reshape(3, 3)
            heading = np.arctan2(rotation[1, 0], rotation[0, 0])
            forward = np.asarray((np.cos(heading), np.sin(heading)))
            self.data.qvel[[self.base_x_dof, self.base_y_dof]] += forward * self.rng.normal(
                0, self.config.velocity_noise * scale)
            self.data.qvel[self.base_yaw_dof] += self.rng.normal(0, self.config.yaw_noise * scale)
            mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def hidden_context(self):
        mu = self.friction_at(self.data.xpos[self.body_id, 0], self.data.xpos[self.body_id, 1])
        band = "low" if mu < .64 else ("medium" if mu < .76 else "high")
        return {"friction": mu, "terrain_band": band, "surface": band}
