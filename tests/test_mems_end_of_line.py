import numpy as np

from experiments.MEMS_end_of_line.data import load_mems
from experiments.MEMS_end_of_line.train_inference import warmup_cosine_learning_rate


def test_mems_split_signal_shape_and_unnormalized_parameters(tmp_path):
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(300, 384))
    theta = rng.normal(loc=3.0, scale=2.0, size=(300, 6))
    np.save(tmp_path / "sig_c1_300.npy", signal)
    np.save(tmp_path / "theta_300.npy", theta)
    data = load_mems(tmp_path, seed=4)
    assert data.train_x.shape == (200, 96, 4)
    assert data.val_x.shape == (50, 96, 4)
    assert data.test_x.shape == (50, 96, 4)
    assert data.train_theta.shape == (200, 6)
    np.testing.assert_allclose(data.train_x.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(data.train_x.std(), 1.0, atol=1e-6)
    indices = np.random.default_rng(4).permutation(300)
    np.testing.assert_array_equal(data.train_theta, theta[indices[:200]].astype(np.float32))
    np.testing.assert_array_equal(data.val_theta, theta[indices[200:250]].astype(np.float32))
    np.testing.assert_array_equal(data.test_theta, theta[indices[250:300]].astype(np.float32))
    np.testing.assert_array_equal(data.theta_mean, np.zeros((1, 6), np.float32))
    np.testing.assert_array_equal(data.theta_std, np.ones((1, 6), np.float32))


def test_warmup_cosine_schedule_uses_explicit_decay_horizon():
    schedule = lambda epoch: float(warmup_cosine_learning_rate(
        epoch * 2, steps_per_epoch=2, start_lr=1e-4,
        warmup_epochs=10, decay_epochs=100, min_learning_rate_ratio=0.01))
    assert np.isclose(schedule(0), 1e-5)
    assert np.isclose(schedule(10), 1e-4)
    assert np.isclose(schedule(60), 5.05e-5)
    assert np.isclose(schedule(110), 1e-6)
    assert np.isclose(schedule(1000), 1e-6)
