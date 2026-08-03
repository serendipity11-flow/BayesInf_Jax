import numpy as np

from experiments.aloha_mujoco_flow.generate_simulator_data import (
    nominal_actions, smooth_random_actions,
)
from experiments.aloha_mujoco_flow.mujoco_aloha import (
    CorrelatedJointDisturbance,
    JointDisturbanceConfig,
    MujocoAlohaPlant,
    aloha_disturbance_modes,
)


def test_menagerie_aloha_mapping_and_step_are_finite():
    plant = MujocoAlohaPlant(control_dt=0.02)
    state = plant.reset()
    assert state.shape == (28,)
    assert plant.action_low.shape == plant.action_high.shape == (14,)
    next_state = plant.step(plant.neutral_action)
    assert next_state.shape == state.shape
    assert np.isfinite(next_state).all()


def test_generated_action_families_respect_actuator_ranges():
    plant = MujocoAlohaPlant(control_dt=0.02)
    rng = np.random.default_rng(0)
    for actions in (
        smooth_random_actions(
            rng, 12, plant.neutral_action, plant.action_low, plant.action_high
        ),
        nominal_actions(
            rng, 12, plant.neutral_action, plant.action_low, plant.action_high, 0.01
        ),
    ):
        assert actions.shape == (12, 14)
        assert np.all(actions >= plant.action_low)
        assert np.all(actions <= plant.action_high)


def test_correlated_disturbance_modes_are_normalized_and_structured():
    modes = aloha_disturbance_modes()
    assert modes.shape == (14, 3)
    np.testing.assert_allclose(np.linalg.norm(modes, axis=0), 1.0, atol=1e-6)
    assert np.all(modes[:, 0] > 0)
    np.testing.assert_allclose(modes[:7, 1], -modes[7:, 1])
    assert np.count_nonzero(modes[:, 2]) == 14
    assert np.abs(modes[[4, 5, 6, 11, 12, 13], 2]).mean() > np.abs(
        modes[[0, 1, 2, 3, 7, 8, 9, 10], 2]
    ).mean()


def test_contact_intensity_scales_same_disturbance_realization_by_three():
    cfg = JointDisturbanceConfig(
        innovation="student_t", sigma_free=1.0, sigma_contact=3.0
    )
    free = CorrelatedJointDisturbance(cfg, np.random.default_rng(12))
    contact = CorrelatedJointDisturbance(cfg, np.random.default_rng(12))
    np.testing.assert_allclose(contact.sample(1.0), 3.0 * free.sample(0.0), rtol=1e-6)


def test_joint_torque_disturbance_changes_only_during_physics_step():
    plant = MujocoAlohaPlant(control_dt=0.02)
    initial = plant.reset()
    snapshot = plant.copy_full_state()
    torque = np.ones(plant.action_dim, dtype=np.float32)
    disturbed = plant.step(plant.neutral_action, torque)
    plant.restore_full_state(snapshot)
    nominal = plant.step(plant.neutral_action)
    np.testing.assert_array_equal(initial, np.concatenate([
        snapshot[0][plant.qpos_indices], snapshot[1][plant.qvel_indices]
    ]).astype(np.float32))
    assert not np.allclose(disturbed, nominal)
    assert np.all(plant.data.qfrc_applied == 0)
