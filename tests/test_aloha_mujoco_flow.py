import numpy as np

from experiments.aloha_mujoco_flow.generate_simulator_data import (
    nominal_actions, smooth_random_actions,
)
from experiments.aloha_mujoco_flow.mujoco_aloha import MujocoAlohaPlant


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

