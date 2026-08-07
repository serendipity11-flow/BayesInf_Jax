import numpy as np
import h5py

from experiments.offroad_flowsde import OffroadRoverEnv
from experiments.offroad_flowsde.dataset import collect, parser


def test_offroad_state_and_hidden_friction_are_finite():
    env = OffroadRoverEnv(seed=3)
    state = env.reset()
    assert state.shape == (10,)
    assert -np.pi <= state[2] <= np.pi
    friction = env.hidden_context["friction"]
    assert env.config.friction_bounds[0] <= friction <= env.config.friction_bounds[1]
    next_state = env.step(np.array([0.8, 0.3]), stochastic=False)
    assert next_state.shape == (10,)
    assert np.all(np.isfinite(next_state))


def test_friction_field_is_spatial_and_reproducible():
    first, second = OffroadRoverEnv(seed=9), OffroadRoverEnv(seed=9)
    first.reset(); second.reset()
    points = (np.array([-4.0, 1.0]), np.array([3.0, -2.0]))
    values_a = [first.friction_at(point) for point in points]
    values_b = [second.friction_at(point) for point in points]
    assert np.allclose(values_a, values_b)
    assert first.hidden_context["patch_centers"].shape == (first.config.patch_count, 2)


def test_v2_dataset_has_split_safe_windows_and_policy_mix(tmp_path):
    args = parser().parse_args([
        "--output", str(tmp_path / "offroad.h5"), "--trajectories", "10",
        "--split-trajectories", "7", "1", "2", "--steps", "60",
        "--progress-every", "0",
    ])
    metadata = collect(args)
    assert metadata["state_order"][6:] == ("omega_fl", "omega_fr", "omega_rl", "omega_rr")
    with h5py.File(args.output) as data:
        assert data["state"].shape == (600, 10)
        assert data["action"].shape == (600, 2)
        assert np.bincount(data["split"][:], minlength=3).tolist() == [420, 60, 120]
        for start, split in zip(data["window_start"][:], data["window_split"][:]):
            rows = slice(start, start + metadata["windows"]["horizon"] + 1)
            assert np.all(data["trajectory_id"][rows] == data["trajectory_id"][start])
            assert np.all(data["split"][rows] == split)
