"""Dataset loading for the MEMS end-of-line benchmark."""

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


def load_mems(data_dir: Path | str = DEFAULT_DATA_DIR, seed: int = 0,
              standardize_signals: bool = True) -> MEMSSplits:
    """Load ``sig_c1``/``theta`` and make deterministic 200/50/50 splits.

    The 384 signal values are interpreted as four contiguous 96-value channels,
    giving each condition the shape ``(96, 4)``. Signal statistics are fitted
    exclusively on the 200 training examples. Parameters are already standardized
    in the source data and are therefore kept unchanged; in particular, parameter
    three (the fourth parameter, directly identifiable offset) remains unchanged.
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
    if theta.ndim != 2 or theta.shape[1] != 6 or len(x) != len(theta):
        raise ValueError(f"Expected theta=(N,6) aligned with x; got {x.shape}, {theta.shape}")
    if x.ndim == 2 and x.shape[1] >= 384:
        x = np.stack(
            (x[:, :96], x[:, 96:192], x[:, 192:288], x[:, 288:384]),
            axis=-1,
        )
    elif x.ndim != 3 or x.shape[1:] != (96, 4):
        raise ValueError(f"Expected x=(N,>=384) or x=(N,96,4); got {x.shape}")
    if len(x) < 300:
        raise ValueError(f"The experiment requires 300 samples, found {len(x)}")

    indices = np.random.default_rng(seed).permutation(len(x))[:300]
    train_i, val_i, test_i = indices[:200], indices[200:250], indices[250:300]
    x_mean = x[train_i].mean(axis=(0, 1), keepdims=True)
    x_std = x[train_i].std(axis=(0, 1), keepdims=True).clip(1e-6)
    # Identity metadata keeps the common sampling/evaluation path in physical
    # (source) parameter units without applying a second normalization.
    theta_mean = np.zeros((1, 6), dtype=np.float32)
    theta_std = np.ones((1, 6), dtype=np.float32)

    if standardize_signals:
        prepare_x = lambda a: ((a - x_mean) / x_std).astype(np.float32)
    else:
        # Retain statistics as metadata, but pass the source signals through
        # unchanged for raw-input encoder comparisons.
        prepare_x = lambda a: a.astype(np.float32)
    return MEMSSplits(
        prepare_x(x[train_i]), theta[train_i].astype(np.float32),
        prepare_x(x[val_i]), theta[val_i].astype(np.float32),
        prepare_x(x[test_i]), theta[test_i].astype(np.float32),
        x_mean, x_std, theta_mean, theta_std,
    )
