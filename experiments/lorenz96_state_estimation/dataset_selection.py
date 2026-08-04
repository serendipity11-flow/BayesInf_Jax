"""Dataset selection for Lorenz-96 training and optional ALOHA transfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import jax.random as jr

from experiments.aloha_state_estimation.utils.dataset import load_aloha_insertion_dataset
from .utils.dataset import generate_lorenz96_dataset, train_val_test_split


def add_dataset_arguments(parser: Any) -> None:
    """Add arguments common to the Lorenz-96 and ALOHA insertion datasets."""

    parser.add_argument(
        "--dataset",
        choices=("lorenz96", "aloha_insertion"),
        default="lorenz96",
        help="State-estimation dataset.",
    )
    parser.add_argument(
        "--data_path",
        default="data/aloha_sim_insertion_human",
        help="LeRobot snapshot directory or parquet file for ALOHA insertion.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=25,
        help="Window stride for ALOHA insertion trajectories.",
    )
    parser.add_argument("--pseudo_process_uncertainty", action="store_true")
    parser.add_argument("--disturbance_dim", type=int, default=4)
    parser.add_argument("--rho_d", type=float, default=0.98)
    parser.add_argument("--sigma_d", type=float, default=0.05)
    parser.add_argument("--regime_gain", type=float, default=2.0)
    parser.add_argument("--mixing_scale", type=float, default=0.35)


def load_selected_dataset(
    args: Any,
    key: jr.PRNGKey,
    *,
    obs_noise_type: str = "gaussian",
    obs_df: float | None = None,
    gmm_weights: Tuple[float, ...] | None = None,
    gmm_offsets: Tuple[float, ...] | None = None,
    gmm_stds: Tuple[float, ...] | None = None,
    clean_initial_state: bool = False,
) -> Tuple[Dict, Dict, Dict, int, int, int, str]:
    """Return train/val/test data, dimensions, and a checkpoint-safe tag."""

    if args.dataset == "aloha_insertion":
        splits, _ = load_aloha_insertion_dataset(
            Path(args.data_path),
            horizon=args.horizon,
            stride=args.window_stride,
            obs_std=args.obs_std,
            obs_noise_type=obs_noise_type,
            obs_df=obs_df if obs_df is not None else 4.0,
            gmm_weights=gmm_weights or (0.70, 0.15, 0.15),
            gmm_offsets=gmm_offsets or (0.0, 2.5, -2.5),
            gmm_stds=gmm_stds or (0.50, 1.50, 1.50),
            dropout_prob=0.0,
            include_mask=False,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            normalize=True,
            clean_initial_state=clean_initial_state,
            pseudo_process_uncertainty=args.pseudo_process_uncertainty,
            disturbance_dim=args.disturbance_dim,
            rho_d=args.rho_d,
            sigma_d=args.sigma_d,
            regime_gain=args.regime_gain,
            mixing_scale=args.mixing_scale,
            dt=args.dt,
        )
        train, val, test = splits["train"], splits["val"], splits["test"]
        d_x = int(train["x"].shape[-1])
        d_y = int(train["y"].shape[-1])
        d_u = int(train["u"].shape[-1])
        tag = f"aloha_insertion_h{args.horizon}_s{args.window_stride}_std{args.obs_std:g}"
        if obs_noise_type != "gaussian":
            tag += f"_{obs_noise_type}"
        if args.pseudo_process_uncertainty:
            tag += (
                f"_ppurho{args.rho_d:g}_sig{args.sigma_d:g}"
                f"_rg{args.regime_gain:g}_b{args.mixing_scale:g}"
            )
        return train, val, test, d_x, d_y, d_u, tag

    data = generate_lorenz96_dataset(
        key,
        n_samples=args.n_samples,
        horizon=args.horizon,
        N=args.N,
        obs_interval=args.obs_interval,
        obs_std=args.obs_std,
        obs_noise_type=obs_noise_type,
        obs_df=obs_df,
        gmm_weights=gmm_weights,
        gmm_offsets=gmm_offsets,
        gmm_stds=gmm_stds,
        dt=args.dt,
        F=getattr(args, "F_lorenz", getattr(args, "F", 8.0)),
        spinup_steps=args.spinup_steps,
        d_u=1,
    )
    train, val, test = train_val_test_split(
        data,
        jr.fold_in(key, 1),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    d_x = args.N
    d_y = args.N // args.obs_interval
    tag = f"lorenz96_N{d_x}_h{args.horizon}_obs{args.obs_interval}_std{args.obs_std:g}"
    return train, val, test, d_x, d_y, 1, tag
