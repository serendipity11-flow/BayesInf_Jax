from pathlib import Path

import numpy as np
import pytest

from bayesinf.applications.model_calibration import (
    DEFAULT_IDENTIFICATION_GROUPS,
    CalibrationConfig,
    IterativeModelCalibrator,
    load_fmu_dataset,
)
from bayesinf.applications.model_calibration.data import (
    CalibrationData, CalibrationNormalization, DataSplit,
)


def test_fmu_loader_split_and_preprocessing(tmp_path: Path):
    theta = np.arange(20 * 19, dtype=np.float32).reshape(20, 19)
    signals = np.full((20, 37, 16), 300.0, dtype=np.float32)
    np.save(tmp_path / "data_parameter_HeatUp_CoolDown_20%.npy", theta)
    np.save(tmp_path / "data_T_HeatUp_CoolDown_20%.npy", signals)
    data = load_fmu_dataset(tmp_path, train_fraction=.6, validation_fraction=.2,
                            real_test_size=2, shuffle=False, stride=18)
    assert data.train.signals.shape == (10, 3, 8)
    assert data.validation.parameters.shape == (3, 19)
    assert data.test.parameters.shape == (5, 19)
    assert data.real_test.parameters.shape == (2, 19)
    np.testing.assert_allclose(data.train.signals, 27.0)
    np.testing.assert_array_equal(data.train.parameters, theta[:10])
    np.testing.assert_array_equal(data.real_test.parameters, theta[-2:])
    assert not np.all(data.real_test.signals == 27.0)


def test_split_seed_never_changes_fixed_test_set(tmp_path: Path):
    theta = np.arange(40 * 19, dtype=np.float32).reshape(40, 19)
    signals = np.zeros((40, 19, 16), dtype=np.float32)
    np.save(tmp_path / "data_parameter_HeatUp_CoolDown_20%.npy", theta)
    np.save(tmp_path / "data_T_HeatUp_CoolDown_20%.npy", signals)
    first = load_fmu_dataset(tmp_path, seed=1, real_test_size=4)
    second = load_fmu_dataset(tmp_path, seed=99, real_test_size=4)
    np.testing.assert_array_equal(first.test.parameters, second.test.parameters)
    np.testing.assert_array_equal(first.real_test.signals, second.real_test.signals)
    assert not np.array_equal(first.train.parameters, second.train.parameters)


def test_calibration_normalization_is_global_for_signals_and_per_parameter():
    split = DataSplit(
        np.array([[[0., 2.]], [[4., 6.]]], dtype=np.float32),
        np.array([[10., 100.], [20., 300.]], dtype=np.float32),
    )
    normalization = CalibrationNormalization.fit(split)
    np.testing.assert_allclose(
        normalization.normalize_signals(split.signals),
        [[[-1., -1 / 3]], [[1 / 3, 1.]]], rtol=1e-6,
    )
    np.testing.assert_allclose(
        normalization.normalize_parameters(split.parameters),
        [[-1., -1.], [1., 1.]],
    )


def test_iterative_groups_must_partition_parameters(tmp_path: Path):
    split = type("Split", (), {
        "signals": np.zeros((2, 3, 2), np.float32),
        "parameters": np.zeros((2, 2), np.float32),
    })()
    data = type("Data", (), {
        "train": split, "validation": split, "test": split, "real_test": split
    })()
    config = CalibrationConfig(groups=((0,), (0,)), epochs=1)
    with pytest.raises(ValueError, match="every parameter exactly once"):
        IterativeModelCalibrator(config, tmp_path).run(data, lambda x, key: x)


def test_complete_five_stage_iterative_smoke(tmp_path: Path):
    rng = np.random.default_rng(11)

    def make_split(size):
        parameters = rng.normal(size=(size, 19)).astype(np.float32)
        # A cheap, deterministic stand-in for the pretrained forward surrogate.
        base = parameters[:, :3] @ np.array(
            [[.4, -.1], [.2, .3], [-.2, .5]], dtype=np.float32
        )
        time = np.linspace(0, 1, 6, dtype=np.float32)[None, :, None]
        signals = base[:, None, :] + time * parameters[:, 3:5, None].transpose(0, 2, 1)
        return DataSplit(signals.astype(np.float32), parameters)

    data = CalibrationData(make_split(10), make_split(4), make_split(2), make_split(2))

    def surrogate(parameters, key):
        del key
        base = parameters[:, :3] @ np.array(
            [[.4, -.1], [.2, .3], [-.2, .5]], dtype=np.float32
        )
        time = np.linspace(0, 1, 6, dtype=np.float32)[None, :, None]
        return (base[:, None, :] +
                time * parameters[:, 3:5, None].transpose(0, 2, 1)).astype(np.float32)

    config = CalibrationConfig(
        groups=DEFAULT_IDENTIFICATION_GROUPS, epochs=1, patience=1,
        batch_size=5, posterior_samples=3, intermediate_posterior_samples=2,
        intermediate_eval_every=1, n_coupling=1, width=4, summary_dim=4,
        rnn_hidden=4, recurrent_layers=1,
    )
    result = IterativeModelCalibrator(config, tmp_path).run(
        data, surrogate, real_test_index=0
    )

    assert result.estimate.shape == (19,)
    assert len(result.posterior_samples) == 5
    assert [sample.shape for sample in result.posterior_samples] == [
        (3, len(group)) for group in DEFAULT_IDENTIFICATION_GROUPS
    ]
    assert (tmp_path / "calibration_result.npz").exists()
    assert (tmp_path / "timing_summary.json").exists()
    assert (tmp_path / "timings.csv").exists()
    for stage in range(1, 6):
        stage_dir = tmp_path / f"stage_{stage:02d}"
        assert (stage_dir / "bayesflow.eqx").exists()
        assert (stage_dir / "bayesflow_state.eqx").exists()
        assert (stage_dir / "latest_intermediate_metrics.json").exists()
        assert (stage_dir / "posterior_samples.npy").exists()
        assert (stage_dir / "timing.json").exists()
