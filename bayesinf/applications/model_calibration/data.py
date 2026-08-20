"""Loading and splitting FMU-generated calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DataSplit:
    signals: np.ndarray
    parameters: np.ndarray


@dataclass(frozen=True)
class CalibrationData:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    real_test: DataSplit


@dataclass(frozen=True)
class CalibrationNormalization:
    """Training-only global signal and dimension-wise parameter min-max scales."""
    signal_min: np.ndarray
    signal_max: np.ndarray
    parameter_min: np.ndarray
    parameter_max: np.ndarray

    @classmethod
    def fit(cls, split: DataSplit) -> CalibrationNormalization:
        return cls(
            np.asarray(split.signals.min(), dtype=np.float32),
            np.asarray(split.signals.max(), dtype=np.float32),
            split.parameters.min(axis=0, keepdims=True).astype(np.float32),
            split.parameters.max(axis=0, keepdims=True).astype(np.float32),
        )

    @staticmethod
    def _scale(value: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        width = np.maximum(maximum - minimum, np.float32(1e-6))
        return (2.0 * (value - minimum) / width - 1.0).astype(np.float32)

    def normalize_signals(self, value: np.ndarray) -> np.ndarray:
        return self._scale(value, self.signal_min, self.signal_max)

    def denormalize_signals(self, value: np.ndarray) -> np.ndarray:
        return (0.5 * (value + 1.0) * (self.signal_max - self.signal_min)
                + self.signal_min).astype(np.float32)

    def normalize_parameters(self, value: np.ndarray) -> np.ndarray:
        return self._scale(value, self.parameter_min, self.parameter_max)

    def save(self, path: Path | str) -> None:
        np.savez(path, signal_min=self.signal_min, signal_max=self.signal_max,
                 parameter_min=self.parameter_min, parameter_max=self.parameter_max,
                 signal_transform="global_minmax_-1_1",
                 parameter_transform="dimensionwise_minmax_-1_1")

    @classmethod
    def load(cls, path: Path | str) -> CalibrationNormalization:
        values = np.load(path)
        return cls(*(values[name] for name in (
            "signal_min", "signal_max", "parameter_min", "parameter_max")))


def load_fmu_dataset(
    data_dir: Path | str,
    *,
    parameter_file: str = "data_parameter_HeatUp_CoolDown_20%.npy",
    signal_file: str = "data_T_HeatUp_CoolDown_20%.npy",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    real_test_size: int = 50,
    real_test_noise_std: float = 0.05,
    real_test_seed: int = 1729,
    seed: int = 0,
    shuffle: bool = True,
    stride: int = 18,
    channels: tuple[int, ...] = tuple(range(0, 16, 2)),
    kelvin_offset: float = 273.0,
) -> CalibrationData:
    """Load FMU arrays and make reproducible train/validation/test/real-test splits.

    Signals are downsampled and converted from Kelvin exactly as in the legacy
    PyTorch project. No normalization is done here; the iterative estimator fits
    stage-specific statistics using only its training partition.  The final
    ``real_test_size`` trajectories are excluded before the development split
    and receive small, deterministic Gaussian observation noise.  This proxy
    real-test partition is reserved for the complete calibration procedure;
    the clean test partition remains available for trained-model evaluation.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below one")
    if stride < 1:
        raise ValueError("stride must be positive")
    if real_test_size < 1:
        raise ValueError("real_test_size must be positive")
    if real_test_noise_std < 0:
        raise ValueError("real_test_noise_std must be non-negative")

    root = Path(data_dir).expanduser()
    theta = np.asarray(np.load(root / parameter_file), dtype=np.float32)
    signals = np.asarray(np.load(root / signal_file), dtype=np.float32)
    if signals.ndim != 3 or theta.ndim != 2 or len(signals) != len(theta):
        raise ValueError(f"Expected signals=(N,T,C), parameters=(N,P); got {signals.shape}, {theta.shape}")
    if not channels or max(channels) >= signals.shape[-1]:
        raise ValueError(f"Invalid channels {channels} for {signals.shape[-1]} signal channels")
    signals = signals[:, ::stride, channels] - np.float32(kelvin_offset)

    if real_test_size >= len(theta):
        raise ValueError("real_test_size must leave data for train/validation/test")

    # Reserve proxy real observations before constructing the clean model-
    # evaluation partitions. Neither held-out partition depends on the seed
    # used to shuffle train and validation data.
    n_development = len(theta) - real_test_size
    n_train = int(n_development * train_fraction)
    n_val = int(n_development * validation_fraction)
    n_test = n_development - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("The requested split leaves an empty partition")
    development_indices = np.arange(n_train + n_val)
    test_indices = np.arange(n_train + n_val, n_development)
    real_test_indices = np.arange(n_development, len(theta))
    if shuffle:
        development_indices = np.random.default_rng(seed).permutation(development_indices)

    def take(idx: np.ndarray) -> DataSplit:
        return DataSplit(signals[idx].copy(), theta[idx].copy())

    real_test = take(real_test_indices)
    noise = np.random.default_rng(real_test_seed).normal(
        loc=0.0, scale=real_test_noise_std, size=real_test.signals.shape
    ).astype(np.float32)
    real_test = DataSplit(real_test.signals + noise, real_test.parameters)

    return CalibrationData(
        take(development_indices[:n_train]),
        take(development_indices[n_train:]),
        take(test_indices),
        real_test,
    )
