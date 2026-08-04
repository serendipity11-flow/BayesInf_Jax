import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.aloha_state_estimation.utils.dataset import (
    denormalize_aloha_state,
    generate_pseudo_process_trajectory,
    load_aloha_insertion_dataset,
)


class AlohaInsertionDatasetTest(unittest.TestCase):
    def test_episode_safe_windows_and_noise(self):
        n_episodes, episode_len, dim = 5, 12, 4
        episodes = np.repeat(np.arange(n_episodes), episode_len)
        frames = np.tile(np.arange(episode_len), n_episodes)
        state = np.stack(
            [episodes + frames / 100 + joint for joint in range(dim)], axis=1
        ).astype(np.float32)
        action = (state + 0.25).astype(np.float32)
        table = pa.table(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "episode_index": episodes,
                "frame_index": frames,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file-000.parquet"
            pq.write_table(table, path)
            splits, metadata = load_aloha_insertion_dataset(
                path,
                horizon=5,
                stride=3,
                obs_std=0.2,
                dropout_prob=0.25,
                seed=7,
                train_frac=0.6,
                val_frac=0.2,
            )

        episode_sets = [
            set(np.asarray(splits[name]["episode_index"]).tolist())
            for name in ("train", "val", "test")
        ]
        self.assertTrue(episode_sets[0].isdisjoint(episode_sets[1]))
        self.assertTrue(episode_sets[0].isdisjoint(episode_sets[2]))
        self.assertTrue(episode_sets[1].isdisjoint(episode_sets[2]))
        self.assertEqual(splits["train"]["x"].shape[1:], (5, dim))
        self.assertEqual(splits["train"]["u"].shape[1:], (5, dim))
        self.assertEqual(splits["train"]["y"].shape[1:], (5, 2 * dim))
        mask = np.asarray(splits["train"]["y"])[..., dim:]
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

        restored = denormalize_aloha_state(splits["test"]["x"], metadata)
        self.assertTrue(np.isfinite(restored).all())

    def test_default_is_single_proprioceptive_observation(self):
        n_episodes, episode_len, dim = 3, 8, 4
        episodes = np.repeat(np.arange(n_episodes), episode_len)
        frames = np.tile(np.arange(episode_len), n_episodes)
        state = np.arange(n_episodes * episode_len * dim, dtype=np.float32).reshape(
            -1, dim
        )
        table = pa.table(
            {
                "observation.state": state.tolist(),
                "action": (state + 1).tolist(),
                "episode_index": episodes,
                "frame_index": frames,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file-000.parquet"
            pq.write_table(table, path)
            splits, _ = load_aloha_insertion_dataset(
                path, horizon=4, stride=4, obs_std=0.1
            )
        self.assertEqual(splits["train"]["y"].shape[-1], dim)

    def test_pseudo_process_is_reproducible_and_episode_consistent(self):
        n_episodes, episode_len, dim = 3, 12, 4
        episodes = np.repeat(np.arange(n_episodes), episode_len)
        frames = np.tile(np.arange(episode_len), n_episodes)
        state = np.stack(
            [0.1 * frames + j for j in range(dim)], axis=1
        ).astype(np.float32)
        action = state + np.linspace(0.0, 0.4, dim, dtype=np.float32)
        first = generate_pseudo_process_trajectory(
            state, action, episodes, seed=11, disturbance_dim=2
        )
        second = generate_pseudo_process_trajectory(
            state, action, episodes, seed=11, disturbance_dim=2
        )
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        # Every episode starts at its recorded state with a reset disturbance.
        starts = np.arange(0, len(state), episode_len)
        np.testing.assert_allclose(first[0][starts], state[starts])
        np.testing.assert_allclose(first[1][starts], 0.0)

        table = pa.table(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "episode_index": episodes,
                "frame_index": frames,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file-000.parquet"
            pq.write_table(table, path)
            splits, metadata = load_aloha_insertion_dataset(
                path,
                horizon=6,
                stride=3,
                seed=4,
                train_frac=1 / 3,
                val_frac=1 / 3,
                pseudo_process_uncertainty=True,
                disturbance_dim=2,
            )
        train = splits["train"]
        # Consecutive windows overlap by three frames and must share the same
        # episode-level latent realization.
        if train["x"].shape[0] >= 2:
            np.testing.assert_allclose(
                np.asarray(train["x"][0, 3:]), np.asarray(train["x"][1, :3])
            )
        self.assertIsNotNone(metadata.disturbance_matrix)
        self.assertEqual(metadata.disturbance_matrix.shape, (dim, 2))


if __name__ == "__main__":
    unittest.main()
