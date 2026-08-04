import numpy as np
import pandas as pd

from experiments.aloha_optimal_control.data import (
    load_episodes,
    load_transition_auxiliary,
)
from experiments.aloha_optimal_control.train_flow_sde import (
    contact_temporal_weights,
    trajectory_windows,
)


def test_contact_force_labels_stay_aligned_with_transition_windows(tmp_path):
    rows = []
    for episode in range(2):
        for step in range(4):
            state = np.full(12, 10 * episode + step, dtype=np.float32)
            rows.append({
                "episode_index": episode,
                "frame_index": step,
                "observation.state": state,
                "action": np.full(3, step, dtype=np.float32),
                "next_state": state + 1,
                "object_contact_force": np.array(
                    [100 * episode + step, 0.0, 0.0], dtype=np.float32
                ),
            })
    path = tmp_path / "contacts.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    states, actions, episode_ids, _, _ = load_episodes(path)
    contact = load_transition_auxiliary(path, "object_contact_force", episode_ids)
    x_mean = np.zeros(12, dtype=np.float32)
    x_std = np.ones(12, dtype=np.float32)
    u_mean = np.zeros(3, dtype=np.float32)
    u_std = np.ones(3, dtype=np.float32)
    windows = trajectory_windows(
        states, actions, np.array([0, 1]), x_mean, x_std, u_mean, u_std,
        window_steps=2, stride=2, auxiliary=contact,
    )

    assert len(windows) == 4
    np.testing.assert_array_equal(
        np.asarray(windows[3])[:, :, 0],
        np.array([[0, 1], [2, 3], [100, 101], [102, 103]]),
    )


def test_contact_temporal_weights_emphasize_high_force_and_have_unit_mean():
    force = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]]], dtype=np.float32)
    weights = np.asarray(contact_temporal_weights(force, np.ones(3)))

    assert weights.shape == force.shape[:2]
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.mean(), 1.0, rtol=1e-5)
    assert weights[0, 1] > weights[0, 2] > weights[0, 0]


def test_contact_weight_persists_into_post_contact_motion():
    force = np.zeros((1, 6, 3), dtype=np.float32)
    force[0, 2, 0] = 10.0
    weights = np.asarray(contact_temporal_weights(
        force, np.ones(3), post_contact_steps=2
    ))

    assert weights[0, 2] == weights[0, 3] == weights[0, 4]
    assert weights[0, 4] > weights[0, 5]
