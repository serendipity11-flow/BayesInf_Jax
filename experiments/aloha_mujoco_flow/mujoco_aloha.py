"""Exact MuJoCo Menagerie ALOHA joint-control plant and coordinate mapping."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


DEFAULT_XML = Path("third_party/mujoco_menagerie/aloha/scene.xml")


class MujocoAlohaPlant:
    """ALOHA 2 Menagerie plant with 14 joint commands and a 28-D Markov state."""

    def __init__(self, xml_path=DEFAULT_XML, control_dt=0.02,
                 gain_scale=1.0, damping_scale=1.0):
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        ratio = control_dt / self.model.opt.timestep
        self.substeps = int(round(ratio))
        if self.substeps < 1 or not np.isclose(ratio, self.substeps):
            raise ValueError("control_dt must be an integer multiple of the MuJoCo timestep")
        self.control_dt = float(control_dt)
        self.qpos_indices = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]].copy()
        self.qvel_indices = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]].copy()
        if self.model.nu != 14 or len(np.unique(self.qpos_indices)) != 14:
            raise ValueError("expected the Menagerie 14-actuator joint-position model")
        self.action_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.action_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.neutral_qpos = self.model.key("neutral_pose").qpos.copy()
        self.neutral_action = self.model.key("neutral_pose").ctrl.copy()
        # Optional domain randomization applies to a fresh model instance only.
        self.model.actuator_gainprm[:, 0] *= float(gain_scale)
        self.model.dof_damping[:] *= float(damping_scale)

    @property
    def state_dim(self):
        return 28

    @property
    def action_dim(self):
        return 14

    def get_state(self):
        return np.concatenate([
            self.data.qpos[self.qpos_indices], self.data.qvel[self.qvel_indices]
        ]).astype(np.float32)

    def reset(self, state=None, position_noise=None, rng=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.neutral_qpos
        self.data.ctrl[:] = self.neutral_action
        if state is not None:
            state = np.asarray(state)
            if state.shape != (self.state_dim,):
                raise ValueError(f"state must have shape {(self.state_dim,)}")
            self.data.qpos[self.qpos_indices] = state[:14]
            self.data.qvel[self.qvel_indices] = state[14:]
            self.data.ctrl[:] = np.clip(state[:14], self.action_low, self.action_high)
        elif position_noise is not None:
            rng = np.random.default_rng() if rng is None else rng
            noise = rng.normal(size=14) * np.asarray(position_noise)
            command = np.clip(self.neutral_action + noise, self.action_low, self.action_high)
            self.data.qpos[self.qpos_indices] = command
            self.data.ctrl[:] = command
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        return self.get_state()

    def copy_full_state(self):
        return (self.data.qpos.copy(), self.data.qvel.copy(), self.data.act.copy(),
                self.data.ctrl.copy(), float(self.data.time))

    def restore_full_state(self, snapshot):
        qpos, qvel, act, ctrl, sim_time = snapshot
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        if self.model.na:
            self.data.act[:] = act
        self.data.ctrl[:], self.data.time = ctrl, sim_time
        mujoco.mj_forward(self.model, self.data)

