"""Offline rollout analysis for saved ALOHA MPPI episodes.

This script replays the exact action sequences that MPPI executed in a saved
comparison run, then compares:

- the true MuJoCo plant rollouts saved in `task*_exact_mujoco.npz`
- the FlowSDE rollouts saved in `task*_flow_sde.npz`
- the fitted MuJoCo rollout model reconstructed from training data

The analysis is completely offline and does not modify MPPI or retrain models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aloha_data import load_episodes, split_episode_indices
from .train_aloha_flow_sde import build_model


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _rollout_prefix(run_dir: Path) -> Path:
    comp_dir = run_dir / "mppi_comparison"
    if not comp_dir.exists():
        raise FileNotFoundError(f"missing comparison directory: {comp_dir}")
    return comp_dir


def _predict_flow(model, initial_state, actions, start_step=0):
    pred = np.asarray(model.deterministic_rollout(jnp.asarray(initial_state), jnp.asarray(actions), start_step=start_step))
    return pred


def _nearest_neighbor_distance(train_xu, xu, batch_size=256):
    # Brute force in chunks to avoid a large memory spike.
    out = np.empty((xu.shape[0],), dtype=np.float32)
    train_xu = np.asarray(train_xu, dtype=np.float32)
    xu = np.asarray(xu, dtype=np.float32)
    for i in range(0, len(xu), batch_size):
        chunk = xu[i:i + batch_size]
        d2 = ((chunk[:, None, :] - train_xu[None, :, :]) ** 2).sum(axis=-1)
        out[i:i + len(chunk)] = np.sqrt(d2.min(axis=1))
    return out


def analyze(args):
    run_dir = args.run_dir
    comp_dir = _rollout_prefix(run_dir)
    config_path = run_dir / "aloha_flow_sde_heun_seg16_noise0p002_seed0_config.json"
    if not config_path.exists():
        candidates = sorted(run_dir.glob("*_config.json"))
        if not candidates:
            raise FileNotFoundError(f"no config json found in {run_dir}")
        config_path = candidates[0]
    with config_path.open() as handle:
        config = json.load(handle)

    episodes, actions, episode_ids, state_dim, control_dim = load_episodes(Path(config["data"]))
    train_i, _, test_i = split_episode_indices(len(episodes), config["seed"])
    stats = np.load(config["normalization"])
    x_mean = np.asarray(stats["x_mean"])
    x_std = np.asarray(stats["x_std"])
    u_mean = np.asarray(stats["u_mean"])
    u_std = np.asarray(stats["u_std"])

    model = build_model(type("Cfg", (), config), state_dim, control_dim)
    model = eqx.tree_deserialise_leaves(config["checkpoint"], model)

    rows = []
    episode_bundle = {}
    dist_rows = []

    train_xu = []
    for index in train_i:
        x = (episodes[index] - x_mean) / x_std
        u = (actions[index] - u_mean) / u_std
        train_xu.append(np.concatenate([x[:-1], u], axis=-1))
    train_xu = np.concatenate(train_xu, axis=0)

    representative = None
    for task_file in sorted(comp_dir.glob("task*_exact_mujoco.npz")):
        task_id = int(task_file.stem.split("_")[0][4:])
        truth_npz = _load_npz(task_file)
        truth = truth_npz["states"]
        flow = _load_npz(comp_dir / f"task{task_id}_flow_sde.npz")["states"]
        actions_seq = truth_npz["actions"]
        initial = truth_npz["initial"]
        episode_id = int(truth_npz["episode_id"])

        # FlowSDE prefix rollouts from every valid start state.
        T = len(truth)
        flow_horizon_errs = [[] for _ in range(T)]
        flow_horizon_norm_errs = [[] for _ in range(T)]
        flow_horizon_mae = [[] for _ in range(T)]
        flow_horizon_max = [[] for _ in range(T)]
        flow_prefix = np.full((T, T, truth.shape[1]), np.nan, dtype=np.float32)
        for start in range(T):
            suffix_actions = actions_seq[start:]
            pred = _predict_flow(model, truth[start], suffix_actions, start_step=start)
            for h in range(1, len(pred) + 1):
                err = pred[h - 1] - truth[start + h - 1]
                flow_horizon_errs[h - 1].append(err)
                flow_horizon_norm_errs[h - 1].append(err / x_std)
                flow_horizon_mae[h - 1].append(np.abs(err))
                flow_horizon_max[h - 1].append(np.abs(err))
                flow_prefix[start, h - 1] = pred[h - 1]
        # Exact MuJoCo baseline is the saved trajectory itself, so its error is zero.
        phys_prefix = np.broadcast_to(truth[:, None, :], (T, T, truth.shape[1])).copy()

        for h in range(1, T + 1):
            flow_errs = np.asarray(flow_horizon_errs[h - 1])
            flow_norm_errs = np.asarray(flow_horizon_norm_errs[h - 1])
            flow_mae = np.asarray(flow_horizon_mae[h - 1])
            flow_max = np.asarray(flow_horizon_max[h - 1])
            if flow_errs.size == 0:
                continue
            rows.append({
                "episode_id": episode_id,
                "task": task_id,
                "horizon": h,
                "model": "flow_sde",
                "rmse": float(np.sqrt(np.mean(flow_errs ** 2))),
                "normalized_rmse": float(np.sqrt(np.mean(flow_norm_errs ** 2))),
                "mae": float(np.mean(flow_mae)),
                "max_abs_error": float(np.max(flow_max)),
            })
            rows.append({
                "episode_id": episode_id,
                "task": task_id,
                "horizon": h,
                "model": "exact_mujoco",
                "rmse": 0.0,
                "normalized_rmse": 0.0,
                "mae": 0.0,
                "max_abs_error": 0.0,
            })

        # Distance to training data for each executed state-action pair.
        xu = np.concatenate([truth[:len(actions_seq)], actions_seq], axis=-1)
        dist = _nearest_neighbor_distance(train_xu, (xu - np.concatenate([x_mean, u_mean])) / np.concatenate([x_std, u_std]))
        for step, value in enumerate(dist, start=1):
            dist_rows.append({
                "episode_id": episode_id,
                "task": task_id,
                "step": step,
                "distance_to_dataset": float(value),
                "flow_prediction_error": float(np.sqrt(np.mean((flow_prefix[step - 1, 0] - truth[step - 1]) ** 2))),
                "physical_prediction_error": 0.0,
            })

        episode_bundle[f"episode_{episode_id}_truth"] = truth
        episode_bundle[f"episode_{episode_id}_flow_prefix"] = flow_prefix
        episode_bundle[f"episode_{episode_id}_physical_prefix"] = phys_prefix
        episode_bundle[f"episode_{episode_id}_actions"] = actions_seq
        episode_bundle[f"episode_{episode_id}_initial"] = initial
        episode_bundle[f"episode_{episode_id}_goal"] = _load_npz(task_file)["goal"]

        if representative is None:
            representative = (episode_id, truth, flow_prefix, actions_seq)

    out_dir = Path("results/rollout_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "rollout_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / "distance_vs_error.csv", index=False)

    # Save all episode-level arrays into one NPZ.
    np.savez_compressed(out_dir / "episode_rollouts.npz", **episode_bundle)

    # Aggregate horizon statistics.
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["horizon", "model"], as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_normalized_rmse=("normalized_rmse", "mean"),
    )
    summary.to_csv(out_dir / "rollout_horizon_summary.csv", index=False)

    # Plot 1: RMSE vs horizon.
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in [("flow_sde", "tab:orange"), ("physical_model", "tab:blue")]:
        sub = summary[summary["model"] == model_name]
        h = sub["horizon"].to_numpy()
        mean = sub["mean_rmse"].to_numpy()
        std = sub["std_rmse"].to_numpy()
        ax.plot(h, mean, label=model_name, color=color)
        ax.fill_between(h, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs prediction horizon")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_vs_horizon.png", dpi=180)
    plt.close(fig)

    # Plot 2: normalized RMSE vs horizon.
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in [("flow_sde", "tab:orange"), ("physical_model", "tab:blue")]:
        sub = summary[summary["model"] == model_name]
        ax.plot(sub["horizon"], sub["mean_normalized_rmse"], label=model_name, color=color)
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("normalized RMSE")
    ax.set_title("Normalized RMSE vs prediction horizon")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "normalized_rmse_vs_horizon.png", dpi=180)
    plt.close(fig)

    # Plot 3: per-joint RMSE heatmap on the FlowSDE model.
    # Simpler: use the first episode to render a compact joint-vs-horizon view.
    episode_id, truth, flow_prefix, actions_seq = representative
    flow_joint = np.abs(np.stack([
        flow_prefix[0, h - 1] - truth[h - 1] for h in range(1, truth.shape[0] + 1)
    ], axis=0)).T
    phys_joint = np.zeros_like(flow_joint)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    im0 = axes[0].imshow(flow_joint.T, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("FlowSDE joint RMSE")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("state dimension")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(phys_joint.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Physical model joint RMSE")
    axes[1].set_xlabel("time step")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(f"Joint RMSE heatmap — episode {episode_id}")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_rmse_heatmap.png", dpi=180)
    plt.close(fig)

    # Plot 4: representative trajectories for three joints.
    chosen = [0, 6, 13]
    seconds = np.arange(truth.shape[0]) * config["dt"]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 8), sharex=True)
    for ax, joint in zip(np.atleast_1d(axes), chosen):
        ax.plot(seconds, truth[:, joint], color="black", label="truth")
        ax.plot(seconds, flow_prefix[:, 0, joint], color="tab:orange", label="FlowSDE")
        ax.plot(seconds, truth[:, joint], color="tab:blue", label="exact MuJoCo")
        ax.set_ylabel(f"dim {joint}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle(f"Trajectory comparison — episode {episode_id}")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_comparison.png", dpi=180)
    plt.close(fig)

    # Plot 5: prediction error versus dataset distance.
    dist_df = pd.DataFrame(dist_rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, color in [("flow_sde", "tab:orange"), ("physical_model", "tab:blue")]:
        if model_name == "flow_sde":
            err = dist_df["flow_prediction_error"]
        else:
            err = dist_df["physical_prediction_error"]
        ax.scatter(dist_df["distance_to_dataset"], err, s=14, alpha=0.6, label=model_name, color=color)
    ax.set_xlabel("normalized distance to training data")
    ax.set_ylabel("prediction RMSE")
    ax.set_title("Prediction error vs dataset distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_error_vs_dataset_distance.png", dpi=180)
    plt.close(fig)

    # Short report.
    flow_ep = metrics[metrics["model"] == "flow_sde"].groupby("horizon")["rmse"].mean()
    phys_ep = metrics[metrics["model"] == "exact_mujoco"].groupby("horizon")["rmse"].mean()
    print(json.dumps({
        "output_dir": str(out_dir),
        "flow_one_step_rmse": float(flow_ep.iloc[0]),
        "flow_final_rmse": float(flow_ep.iloc[-1]),
        "physical_one_step_rmse": float(phys_ep.iloc[0]),
        "physical_final_rmse": float(phys_ep.iloc[-1]),
        "flow_error_growth_horizon": int(np.argmax(np.diff(flow_ep.to_numpy()) > 0.05) + 1),
    }, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing mppi_comparison/ and the FlowSDE config.",
    )
    return p


if __name__ == "__main__":
    analyze(parser().parse_args())
