import h5py
import numpy as np
from experiments.differential_drive_flowsde import DifferentialDriveConfig, DifferentialDriveEnv
from experiments.differential_drive_flowsde.dataset import collect, parser
from experiments.differential_drive_flowsde.train_multishoot import (balanced_surface_sample,
    causal_contexts, terrain_stratified_sample)


def test_environment_has_five_observed_states_and_hidden_surface():
    env = DifferentialDriveEnv(seed=4)
    state = env.reset((-1, 0, 0))
    assert state.shape == (5,)
    assert env.hidden_context["terrain_band"] in {"low", "medium", "high"}
    next_state = env.step(np.asarray((1.0, 0.5)), stochastic=False)
    assert next_state.shape == (5,) and np.all(np.isfinite(next_state))
    samples = np.array([env.friction_at(x, y) for x in np.linspace(-4, 4, 20)
                        for y in np.linspace(-3, 3, 12)])
    assert samples.min() >= env.config.friction_min
    assert samples.max() <= 1.0
    assert np.max(np.abs(np.diff([env.friction_at(x, 0) for x in np.linspace(-1, 1, 100)]))) < .02


def test_dataset_keeps_friction_out_of_observations(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path / "dd.h5"), "--trajectories", "3",
                                "--split-trajectories", "1", "1", "1", "--steps", "4"])
    collect(args)
    with h5py.File(args.output) as f:
        assert f["state"].shape == (12, 5) and f["action"].shape == (12, 2)
        assert not bool(f.attrs["friction_is_observed"])
        assert f["target"].shape == (12, 2)
        assert np.bincount(f["split"][:], minlength=3).tolist() == [4, 4, 4]


def test_extended_state_contains_lateral_and_wheel_velocities(tmp_path):
    env = DifferentialDriveEnv(DifferentialDriveConfig(extended_state=True), seed=4)
    state = env.reset((-1, 0, 0))
    assert state.shape == (8,)
    assert np.all(np.isfinite(env.step(np.asarray((1.0, 0.5)), stochastic=False)))
    args = parser().parse_args(["--output", str(tmp_path / "extended.h5"),
                                "--trajectories", "3", "--split-trajectories", "1", "1", "1",
                                "--steps", "4", "--extended-state"])
    collect(args)
    with h5py.File(args.output) as f:
        assert f["state"].shape == (12, 8)
        assert "wheel_left" in f.attrs["state_order"]


def test_full_state_exposes_contact_and_actuator_dynamics():
    env = DifferentialDriveEnv(DifferentialDriveConfig(full_state=True), seed=4)
    env.reset((-1, 0, 0))
    state = env.step(np.asarray((1.0, 0.5)), stochastic=False)
    assert state.shape == (17,)
    assert np.all(np.isfinite(state))


def test_surface_balanced_windows_use_midpoint_and_oversample_ice():
    candidate_windows = np.arange(24).reshape(6, 4)
    friction = np.full(24, .8)
    friction[candidate_windows[:2, 2]] = .15
    sampled = balanced_surface_sample(candidate_windows, friction, 20, .5,
                                      np.random.default_rng(3))
    sampled_ice = friction[sampled[:, 2]] < .5
    assert sampled.shape == (20, 4)
    assert sampled_ice.sum() == 10


def test_terrain_stratified_windows_cover_all_quantiles():
    candidate_windows = np.arange(48).reshape(12, 4)
    friction = np.linspace(.5, .9, 48)
    sampled = terrain_stratified_sample(candidate_windows, friction, 24, 4,
                                        np.random.default_rng(2))
    midpoint_values = friction[sampled[:, 2]]
    source_values = friction[candidate_windows[:, 2]]
    edges = np.quantile(source_values, np.linspace(0, 1, 5))
    counts = np.histogram(midpoint_values, bins=edges)[0]
    assert sampled.shape == (24, 4)
    assert counts.tolist() == [6, 6, 6, 6]


def test_causal_contexts_pad_without_crossing_episodes():
    values = np.arange(10).reshape(5, 2)
    actions = np.arange(5).reshape(5, 1)
    states, previous = causal_contexts(values, actions, np.array([0, 0, 0, 1, 1]), 3,
                                        np.array([-1]))
    assert states[3].tolist() == [values[3].tolist()] * 3
    assert previous[3, :, 0].tolist() == [-1, -1]
    assert states[2].tolist() == values[:3].tolist()
    assert previous[2, :, 0].tolist() == [0, 1]
