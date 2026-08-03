"""Exact MuJoCo Menagerie ALOHA joint-control plant and coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np


DEFAULT_XML = Path("third_party/mujoco_menagerie/aloha/scene.xml")


@dataclass(frozen=True)
class JointDisturbanceConfig:
    """Configuration for correlated actuator-joint torque disturbances."""

    latent_dim: int = 3
    rho: float = 0.98
    sigma_z: float = 1.0
    sigma_ind: float = 0.05
    innovation: Literal["gaussian", "gaussian_mixture", "student_t"] = (
        "gaussian_mixture"
    )
    mixture_probability: float = 0.05
    mixture_scale: float = 5.0
    student_df: float = 4.0
    sigma_free: float = 1.0
    sigma_contact: float = 3.0
    contact_force_reference: float = 50.0

    def __post_init__(self):
        if self.latent_dim != 3:
            raise ValueError("ALOHA disturbance construction requires latent_dim=3")
        if not 0 <= self.rho < 1 or self.sigma_z < 0 or self.sigma_ind < 0:
            raise ValueError("rho must be in [0,1), and noise scales non-negative")
        if self.innovation not in {"gaussian", "gaussian_mixture", "student_t"}:
            raise ValueError(f"unsupported innovation: {self.innovation}")
        if not 0 <= self.mixture_probability <= 1 or self.mixture_scale <= 0:
            raise ValueError("invalid Gaussian-mixture parameters")
        if self.student_df <= 2:
            raise ValueError("student_df must exceed 2 for finite variance")
        if self.sigma_free < 0 or self.sigma_contact < 0:
            raise ValueError("contact scales must be non-negative")
        if self.contact_force_reference <= 0:
            raise ValueError("contact_force_reference must be positive")


def aloha_disturbance_modes(n_joints: int = 14) -> np.ndarray:
    """Return normalized common, arm-differential, and wrist modes."""

    if n_joints != 14:
        raise ValueError("ALOHA disturbance modes are defined for 14 joints")
    modes = np.zeros((n_joints, 3), dtype=np.float64)
    modes[:, 0] = 1.0
    modes[:7, 1], modes[7:, 1] = 1.0, -1.0
    # Wrist rotation/flex and gripper joints on the left and right arms.
    modes[[4, 5, 6, 11, 12, 13], 2] = 1.0
    modes[:, 2] -= modes[:, 2].mean()
    modes /= np.linalg.norm(modes, axis=0, keepdims=True)
    return modes.astype(np.float32)


class CorrelatedJointDisturbance:
    """Stateful AR(1) joint-torque disturbance generator."""

    def __init__(self, config: JointDisturbanceConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.L = aloha_disturbance_modes()
        self.z = config.sigma_z * self._innovation()

    def _innovation(self) -> np.ndarray:
        cfg = self.config
        if cfg.innovation == "gaussian":
            sample = self.rng.normal(size=cfg.latent_dim)
        elif cfg.innovation == "gaussian_mixture":
            outlier = self.rng.random(cfg.latent_dim) < cfg.mixture_probability
            scales = np.where(outlier, cfg.mixture_scale, 1.0)
            sample = scales * self.rng.normal(size=cfg.latent_dim)
            # Unit variance keeps sigma_z comparable across innovation families.
            variance = (
                (1.0 - cfg.mixture_probability)
                + cfg.mixture_probability * cfg.mixture_scale**2
            )
            sample /= np.sqrt(variance)
        else:
            sample = self.rng.standard_t(cfg.student_df, size=cfg.latent_dim)
            sample /= np.sqrt(cfg.student_df / (cfg.student_df - 2.0))
        return np.asarray(sample, dtype=np.float32)

    def sample(self, contact_intensity: float) -> np.ndarray:
        """Sample current torque, then advance the latent state for next step."""

        cfg = self.config
        r_t = float(np.clip(contact_intensity, 0.0, 1.0))
        scale = cfg.sigma_free + r_t * (cfg.sigma_contact - cfg.sigma_free)
        independent = cfg.sigma_ind * self.rng.normal(size=self.L.shape[0])
        torque = scale * (self.L @ self.z + independent)
        self.z = (
            cfg.rho * self.z
            + cfg.sigma_z * np.sqrt(1.0 - cfg.rho**2) * self._innovation()
        )
        return np.asarray(torque, dtype=np.float32)


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

    def contact_intensity(self, force_reference: float = 50.0) -> float:
        """Normalized total normal-contact force for the current state."""

        if force_reference <= 0:
            raise ValueError("force_reference must be positive")
        force = np.zeros(6, dtype=np.float64)
        total_normal_force = 0.0
        for contact_id in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, contact_id, force)
            total_normal_force += max(float(force[0]), 0.0)
        return float(np.clip(total_normal_force / force_reference, 0.0, 1.0))

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

    def step(self, action, disturbance_torque=None):
        """Advance physics after applying control and optional joint torque.

        ``ctrl`` remains the position-controller command. A disturbance is
        applied as generalized joint torque through ``qfrc_applied`` before
        integration, giving net joint torque controller minus disturbance.
        """

        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        self.data.qfrc_applied[:] = 0.0
        if disturbance_torque is not None:
            disturbance_torque = np.asarray(disturbance_torque, dtype=np.float64)
            if disturbance_torque.shape != (self.action_dim,):
                raise ValueError(
                    f"disturbance_torque must have shape {(self.action_dim,)}"
                )
            self.data.qfrc_applied[self.qvel_indices] = -disturbance_torque
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.data.qfrc_applied[:] = 0.0
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
