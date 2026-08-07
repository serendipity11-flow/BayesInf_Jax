"""MuJoCo ground-truth plant with an unobserved spatial friction field."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class OffroadConfig:
    control_dt: float = 0.01
    drive_gain: float = 1.0
    steer_gain: float = 0.75
    delta_max: float = 0.6
    friction_bounds: tuple[float, float] = (0.12, 1.25)
    base_friction: float = 0.95
    patch_count: int = 7
    patch_amplitude: tuple[float, float] = (-0.75, 0.25)
    patch_width: tuple[float, float] = (0.7, 2.2)
    roughness_std: float = 0.018
    wind_std: float = 0.012
    mode_probabilities: tuple[float, float, float] = (0.72, 0.22, 0.06)
    mode_slip: tuple[float, float, float] = (0.0, 0.20, 0.62)


class OffroadRoverEnv:
    """Wheel-resolved rover; friction and slip mode are deliberately hidden.

    State is ``[x, y, yaw, v_long, v_lat, yaw_rate, omega_fl, omega_fr,
    omega_rl, omega_rr]``. Action is throttle/brake in ``[-1, 1]`` and steering
    angle in ``[-delta_max, delta_max]``.
    """

    state_dim, action_dim = 10, 2

    def __init__(self, config=OffroadConfig(), seed=0, model_path=None, terrain_map=None):
        self.config = config
        path = Path(model_path or Path(__file__).with_name("rover.xml"))
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        ratio = config.control_dt / self.model.opt.timestep
        self.substeps = int(round(ratio))
        if self.substeps < 1 or not np.isclose(ratio, self.substeps):
            raise ValueError("control_dt must be an integer multiple of model timestep")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        self.action_low = np.asarray((-1.0, -config.delta_max), np.float32)
        self.action_high = np.asarray((1.0, config.delta_max), np.float32)
        self.terrain_map = terrain_map
        self._centers = self._amplitudes = self._widths = None
        self._previous_wheel_speed = np.zeros(4)

    def _sample_field(self):
        if self.terrain_map is not None:
            # Fixed maps make train/validation share Map A while test is genuinely
            # OOD on Map B. Narrow Map-B kernels create sharper transitions.
            maps = {
                0: (((-5.5, -2.0), (-1.0, 3.5), (3.0, -3.5), (6.0, 2.5)),
                    (-0.58, -0.38, 0.18, -0.68), (2.2, 1.8, 2.5, 1.6)),
                1: (((-6.0, 3.0), (-2.0, -2.5), (2.5, 2.0), (6.5, -1.0)),
                    (-0.72, 0.22, -0.62, -0.48), (0.65, 0.85, 0.55, 0.75)),
            }
            centers, amplitudes, widths = maps[int(self.terrain_map)]
            self._centers = np.asarray(centers, dtype=float)
            self._amplitudes = np.asarray(amplitudes, dtype=float)
            self._widths = np.asarray(widths, dtype=float)
            return
        n = self.config.patch_count
        self._centers = self.rng.uniform((-10, -7), (10, 7), size=(n, 2))
        self._amplitudes = self.rng.uniform(*self.config.patch_amplitude, size=n)
        self._widths = self.rng.uniform(*self.config.patch_width, size=n)

    def friction_at(self, position):
        """Evaluate the episode-specific Gaussian-mixture traction map."""
        delta = np.asarray(position)[None, :2] - self._centers
        bumps = self._amplitudes * np.exp(-0.5 * np.sum(delta * delta, axis=1) / self._widths**2)
        return float(np.clip(self.config.base_friction + bumps.sum(), *self.config.friction_bounds))

    def reset(self, pose=(0.0, 0.0, 0.0), randomize=True):
        mujoco.mj_resetData(self.model, self.data)
        if randomize or self._centers is None:
            self._sample_field()
        x, y, yaw = pose
        self.data.qpos[:7] = (x, y, 0.30, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        self._previous_wheel_speed[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        world_velocity = self.data.qvel[:2]
        forward, lateral = rotation[:2, 0], rotation[:2, 1]
        v_long, v_lat = world_velocity @ forward, world_velocity @ lateral
        wheel_speed = self.data.qvel[6:10]
        surface_speed = 0.10 * np.mean(np.abs(wheel_speed))
        slip = (surface_speed - abs(v_long)) / max(surface_speed, abs(v_long), 0.1)
        return np.asarray((self.data.qpos[0], self.data.qpos[1], yaw, v_long, v_lat,
                           self.data.qvel[5], *wheel_speed), dtype=np.float32)

    def step(self, action, stochastic=True):
        throttle = float(np.clip(np.asarray(action, dtype=float)[0], -1, 1))
        steering = float(np.clip(np.asarray(action, dtype=float)[1],
                                 -self.config.delta_max, self.config.delta_max))
        steering /= self.config.delta_max
        left = self.config.drive_gain * (throttle - self.config.steer_gain * steering)
        right = self.config.drive_gain * (throttle + self.config.steer_gain * steering)
        traction = self.friction_at(self.data.qpos[:2])
        self.model.geom_friction[self.ground_id, 0] = traction
        mode = self.rng.choice(3, p=self.config.mode_probabilities) if stochastic else 0
        slip = self.config.mode_slip[mode]
        # With the wheel-cylinder frame in rover.xml, negative joint rotation
        # corresponds to positive chassis-x travel.
        self.data.ctrl[:] = (-left * (1 - slip), -right * (1 - slip),
                             -left * (1 - slip), -right * (1 - slip))
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        if stochastic:
            speed = np.linalg.norm(self.data.qvel[:2])
            # Configured standard deviations are continuous-time diffusion
            # intensities. The bounded linear speed term avoids the numerical
            # positive feedback produced by the old per-step speed-squared kick.
            speed_factor = 1.0 + 0.35 * min(float(speed), 10.0)
            scale = self.config.roughness_std * speed_factor * np.sqrt(self.config.control_dt)
            self.data.qvel[:2] += self.rng.normal(0, scale, 2)
            self.data.qvel[5] += self.rng.normal(0, scale * 0.8)
            self.data.qvel[:2] += self.rng.normal(
                0, self.config.wind_std * np.sqrt(self.config.control_dt), 2)
            mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def hidden_context(self):
        """Diagnostics only; never include this in learned-model observations."""
        return {"friction": self.friction_at(self.data.qpos[:2]),
                "patch_centers": self._centers.copy(), "patch_widths": self._widths.copy()}
