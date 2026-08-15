"""Dataset loading and leakage-free normalization for the MEMS benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path("/home/zhangyi/data/MEMS/MEMS_simulated")


@dataclass(frozen=True)
class MEMSSplits:
    train_x: np.ndarray
    train_theta: np.ndarray
    val_x: np.ndarray
    val_theta: np.ndarray
    test_x: np.ndarray
    test_theta: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    theta_mean: np.ndarray
    theta_std: np.ndarray


def load_bayesflow_reference(data_dir: Path | str = DEFAULT_DATA_DIR) -> MEMSSplits:
    """Reproduce the preprocessing/split in the legacy BayesFlow MEMS script.

    The first 384 signal values become 96 time steps with four channels. The
    signal is divided by its global channel range, and samples are split in
    original order as 0:200, 200:250, and 250:300. Parameters are left in their
    original scale, matching the reference implementation.
    """
    data_dir = Path(data_dir).expanduser()
    x = np.asarray(np.load(data_dir / "sig_c1_300.npy"), dtype=np.float32)
    theta = np.asarray(np.load(data_dir / "theta_300.npy"), dtype=np.float32)
    if x.shape[0] < 300 or x.shape[1] < 384 or theta.shape != (300, 6):
        raise ValueError(f"Expected signal >=(300,384), theta=(300,6); got {x.shape}, {theta.shape}")
    x = np.stack((x[:, :96], x[:, 96:192], x[:, 192:288], x[:, 288:384]), axis=-1)
    channel_range = (x.max(axis=(0, 1)) - x.min(axis=(0, 1))).clip(1e-6).reshape(1, 1, 4)
    x = (x / channel_range).astype(np.float32)
    zeros = np.zeros((1, 6), dtype=np.float32)
    ones = np.ones((1, 6), dtype=np.float32)
    return MEMSSplits(x[:200], theta[:200], x[200:250], theta[200:250],
                      x[250:300], theta[250:300], np.zeros((1, 1, 4), np.float32),
                      channel_range, zeros, ones)


def load_mems(data_dir: Path | str = DEFAULT_DATA_DIR, seed: int = 0) -> MEMSSplits:
    """Load ``sig_c1``/``theta`` and make deterministic 200/50/50 splits.

    Statistics are fitted exclusively on the 200 training examples. A singleton
    feature axis is appended to the signal for the recurrent condition encoder.
    """
    data_dir = Path(data_dir).expanduser()
    x_path = data_dir / "sig_c1_300.npy"
    theta_path = data_dir / "theta_300.npy"
    if not x_path.exists():  # Other MEMS folders use un-suffixed signal names.
        x_path = data_dir / "sig_c1.npy"
    if not theta_path.exists():
        candidates = sorted(data_dir.glob("theta_*.npy"))
        if not candidates:
            raise FileNotFoundError(f"No theta_*.npy found in {data_dir}")
        theta_path = candidates[0]

    x = np.asarray(np.load(x_path), dtype=np.float32)
    theta = np.asarray(np.load(theta_path), dtype=np.float32)
    if x.ndim != 2 or theta.ndim != 2 or len(x) != len(theta):
        raise ValueError(f"Expected x=(N,T), theta=(N,P); got {x.shape}, {theta.shape}")
    if len(x) < 300:
        raise ValueError(f"The experiment requires 300 samples, found {len(x)}")

    indices = np.random.default_rng(seed).permutation(len(x))[:300]
    train_i, val_i, test_i = indices[:200], indices[200:250], indices[250:300]
    x_mean = x[train_i].mean(axis=(0, 1), keepdims=True)
    x_std = x[train_i].std(axis=(0, 1), keepdims=True).clip(1e-6)
    theta_mean = theta[train_i].mean(axis=0, keepdims=True)
    theta_std = theta[train_i].std(axis=0, keepdims=True).clip(1e-6)

    norm_x = lambda a: ((a - x_mean) / x_std)[..., None].astype(np.float32)
    norm_t = lambda a: ((a - theta_mean) / theta_std).astype(np.float32)
    return MEMSSplits(
        norm_x(x[train_i]), norm_t(theta[train_i]),
        norm_x(x[val_i]), norm_t(theta[val_i]),
        norm_x(x[test_i]), theta[test_i].astype(np.float32),
        x_mean, x_std, theta_mean, theta_std,
    )
