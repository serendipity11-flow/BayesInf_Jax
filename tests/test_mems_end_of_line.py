import numpy as np

from experiments.MEMS_end_of_line.data import load_mems


def test_mems_split_and_train_only_normalization(tmp_path):
    rng = np.random.default_rng(2)
    np.save(tmp_path / "sig_c1_300.npy", rng.normal(size=(300, 12)))
    np.save(tmp_path / "theta_300.npy", rng.normal(size=(300, 6)))
    data = load_mems(tmp_path, seed=4)
    assert data.train_x.shape == (200, 12, 1)
    assert data.val_x.shape == (50, 12, 1)
    assert data.test_x.shape == (50, 12, 1)
    assert data.train_theta.shape == (200, 6)
    np.testing.assert_allclose(data.train_x.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(data.train_x.std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(data.train_theta.mean(axis=0), 0.0, atol=2e-6)
    np.testing.assert_allclose(data.train_theta.std(axis=0), 1.0, atol=2e-6)
