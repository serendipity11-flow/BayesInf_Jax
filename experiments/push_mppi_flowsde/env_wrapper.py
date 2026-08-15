"""Noisy MuJoCo pusher with 3-D Cartesian end-effector velocity commands."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .state import STATE_DIM


@dataclass(frozen=True)
class EnvConfig:
    control_dt: float = 0.02
    process_position_std: float = 2e-4
    process_velocity_std: float = 2e-3
    friction_range: tuple[float, float] = (0.75, 1.25)
    solver_range: tuple[float, float] = (0.9, 1.1)
    gate_position: tuple[float, float, float] = (0.28, 0.0, 0.0)
    gate_half_width: float = 0.15
    velocity_gain: float = 2.8
    velocity_limit: float = 1.2


class NoisyPushingEnv:
    def __init__(self, xml_path, config=EnvConfig(), seed=0,
                 ee_body="fingertip", object_body="cuboid"):
        self.xml_path, self.config = Path(xml_path).resolve(), config
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data, self.rng = mujoco.MjData(self.model), np.random.default_rng(seed)
        self.ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if min(self.ee_body, self.object_body) < 0:
            raise ValueError("end-effector/object body name not found")
        ratio = config.control_dt / self.model.opt.timestep
        self.substeps = int(round(ratio))
        if self.substeps < 1 or not np.isclose(ratio, self.substeps):
            raise ValueError("control_dt must be an integer model-timestep multiple")
        self.arm_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]].copy()
        joints = np.flatnonzero(self.model.jnt_bodyid == self.object_body)
        self.object_qpos = self.model.jnt_qposadr[joints]
        self.object_dofs = self.model.jnt_dofadr[joints]
        self.base_friction = self.model.geom_friction.copy()
        self.base_solimp, self.base_solref = self.model.geom_solimp.copy(), self.model.geom_solref.copy()
        self.action_low = np.full(3, -config.velocity_limit, np.float32)
        self.action_high = -self.action_low

    def reset(self, qpos=None, qvel=None, randomize=True):
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[:] = qpos
        elif self.model.nq == 6 and len(self.arm_dofs) == 3:
            self.data.qpos[:3] = (1.0, -2.0, 1.0)
            self.data.qpos[3:6] = (-0.05, self.rng.uniform(-0.09, 0.09),
                                   self.rng.uniform(-0.25, 0.25))
        if qvel is not None:
            self.data.qvel[:] = qvel
        if randomize:
            flo, fhi = self.config.friction_range
            slo, shi = self.config.solver_range
            self.model.geom_friction[:] = self.base_friction * self.rng.uniform(flo, fhi)
            self.model.geom_solimp[:] = self.base_solimp * self.rng.uniform(slo, shi)
            self.model.geom_solref[:] = self.base_solref * self.rng.uniform(slo, shi)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        ep, op = self.data.xpos[self.ee_body].copy(), self.data.xpos[self.object_body].copy()
        ev, ov = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.ee_body, ev, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.object_body, ov, 0)
        rotation = self.data.xmat[self.object_body].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        gate = np.asarray(self.config.gate_position)
        state = np.concatenate((ep, ev[3:], op, [yaw], ov[3:], ov[:3], op - ep,
                                gate - op, [op[1] + self.config.gate_half_width,
                                            self.config.gate_half_width - op[1]])).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"invalid extracted state shape {state.shape}")
        return state

    def step(self, action, noisy=True):
        action = np.clip(np.asarray(action), self.action_low, self.action_high)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body)
        desired = np.linalg.pinv(jacp[:, self.arm_dofs], rcond=1e-4) @ action
        torque = self.config.velocity_gain * (desired - self.data.qvel[self.arm_dofs])
        self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0],
                                    self.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        if noisy:
            self.data.qpos[self.object_qpos] += self.rng.normal(
                0, self.config.process_position_std, len(self.object_qpos))
            self.data.qvel[self.object_dofs] += self.rng.normal(
                0, self.config.process_velocity_std, len(self.object_dofs))
            mujoco.mj_forward(self.model, self.data)
        return self.state()

    def snapshot(self):
        return (self.data.qpos.copy(), self.data.qvel.copy(), self.data.act.copy(),
                self.data.ctrl.copy(), float(self.data.time))

    def restore(self, snapshot):
        qpos, qvel, act, ctrl, time = snapshot
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        if self.model.na: self.data.act[:] = act
        self.data.ctrl[:], self.data.time = ctrl, time
        mujoco.mj_forward(self.model, self.data)
