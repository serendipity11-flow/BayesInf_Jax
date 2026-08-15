"""Compare FlowSDE and nonlinear physical-model open-loop state reconstruction."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import load_model
from .evaluate_nonlinear_physical_mppi import friction_at
from .extended_physical_model import ExtendedPhysicalModel


COMMON = ("p_x", "p_y", "theta", "v_forward", "v_lateral", "omega", "wheel_left", "wheel_right")


def physical_rollout(initial, actions, dt=.02, gain_v=.225, gain_omega=.36,
                     tau_v=.25, tau_omega=.18, drag_v=.08, drag_omega=.04,
                     turn_drag=.08):
    x, y, theta, v, omega = map(float, initial)
    path = [np.asarray((x, y, theta, v, omega))]
    for v_cmd, omega_cmd in actions:
        traction = friction_at(x, y) / .72
        v_dot = (traction * (gain_v * v_cmd - v) / tau_v
                 - drag_v * v * abs(v) - turn_drag * abs(omega) * v)
        omega_dot = (traction * (gain_omega * omega_cmd - omega) / tau_omega
                     - drag_omega * omega * abs(omega))
        v_next, omega_next = v + dt * v_dot, omega + dt * omega_dot
        v_mid, omega_mid = .5 * (v + v_next), .5 * (omega + omega_next)
        theta_mid = theta + .5 * dt * omega_mid
        x += dt * v_mid * np.cos(theta_mid); y += dt * v_mid * np.sin(theta_mid)
        theta = np.arctan2(np.sin(theta + dt * omega_mid), np.cos(theta + dt * omega_mid))
        v, omega = v_next, omega_next
        path.append(np.asarray((x, y, theta, v, omega)))
    return np.asarray(path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model, config = load_model(args.result)
    physical_model = ExtendedPhysicalModel.load(args.physical_model)
    stats = dict(np.load(args.result / "normalization.npz"))
    with h5py.File(args.data) as f:
        state, action, next_state, split, episode = (f[k][:] for k in
            ("state", "action", "next_state", "split", "trajectory_id"))
        names = tuple(json.loads(f.attrs["state_order"]))
    indices = np.asarray([names.index(name) for name in COMMON])
    dynamic_indices = np.asarray(config["dynamic_indices"])
    dt = float(config["dt"])

    def encode(x):
        raw = np.concatenate((x[:2], np.sin(x[2:3]), np.cos(x[2:3]), x[3:]))
        return (raw - stats["feature_mean"]) / stats["feature_std"]

    def flowsde_step(x, u):
        normalized_u = (u - stats["action_mean"]) / stats["action_std"]
        delta = np.asarray(model.mean_delta(jnp.asarray(encode(x)), jnp.asarray(normalized_u)))
        delta = delta * stats["delta_std"] + stats["delta_mean"]
        nxt = x.copy(); nxt[dynamic_indices] += delta
        vf, vl = .5 * (x[3] + nxt[3]), .5 * (x[4] + nxt[4])
        omega = .5 * (x[5] + nxt[5]); theta = x[2] + dt * omega
        nxt[0] = x[0] + dt * (vf * np.cos(theta) - vl * np.sin(theta))
        nxt[1] = x[1] + dt * (vf * np.sin(theta) + vl * np.cos(theta))
        nxt[2] = np.arctan2(np.sin(theta), np.cos(theta))
        return nxt

    truths, flow_paths, physical_paths = [], [], []
    test_episodes = np.unique(episode[split == 2])[:args.episodes]
    for ep in test_episodes:
        rows = np.flatnonzero(episode == ep)[:args.horizon]
        if len(rows) < args.horizon: continue
        truth = np.concatenate((state[rows[:1]], next_state[rows]), axis=0)[:, indices]
        flow = [state[rows[0]].copy()]
        for row in rows: flow.append(flowsde_step(flow[-1], action[row]))
        physical_initial = state[rows[0], indices]
        truths.append(truth); flow_paths.append(np.asarray(flow)[:, indices])
        path = [physical_initial.copy()]
        for row in rows: path.append(physical_model.step(path[-1], action[row]))
        physical_paths.append(np.asarray(path))

    truth = np.concatenate(truths); flow = np.concatenate(flow_paths); physical = np.concatenate(physical_paths)
    flow_error, physical_error = flow - truth, physical - truth
    for error in (flow_error, physical_error):
        error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))
    scale = np.maximum(np.std(truth, axis=0), 1e-8)
    flow_rmse = np.sqrt(np.mean(flow_error ** 2, axis=0))
    physical_rmse = np.sqrt(np.mean(physical_error ** 2, axis=0))
    flow_nrmse, physical_nrmse = flow_rmse / scale, physical_rmse / scale

    rows = []
    for i, name in enumerate(COMMON):
        rows.append({"state": name, "FlowSDE RMSE": float(flow_rmse[i]),
                     "nonlinear physical RMSE": float(physical_rmse[i]),
                     "FlowSDE NRMSE": float(flow_nrmse[i]),
                     "nonlinear physical NRMSE": float(physical_nrmse[i]),
                     "lower NRMSE": "FlowSDE" if flow_nrmse[i] < physical_nrmse[i] else "nonlinear physical"})
    payload = {"episodes": len(truths), "horizon": args.horizon,
               "normalization": "RMSE divided by held-out rollout target standard deviation",
               "states": rows}
    (args.output / "open_loop_state_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "open_loop_state_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

    labels = [[r["state"], f'{r["FlowSDE RMSE"]:.4f}', f'{r["nonlinear physical RMSE"]:.4f}',
               f'{r["FlowSDE NRMSE"]:.3f}', f'{r["nonlinear physical NRMSE"]:.3f}', r["lower NRMSE"]]
              for r in rows]
    figure, ax = plt.subplots(figsize=(15, 5.2)); ax.axis("off")
    table = ax.table(cellText=labels,
                     colLabels=("State", "FlowSDE\nRMSE", "Physical model\nRMSE",
                                "FlowSDE\nNRMSE", "Physical model\nNRMSE", "Lower NRMSE"),
                     cellLoc="center", loc="center", colWidths=(.14, .16, .18, .16, .18, .18))
    table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 2.2)
    for (row, _), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(weight="bold", fontsize=14); cell.set_facecolor("#dce6f1")
    ax.set_title(f"Open-loop state reconstruction ({len(truths)} test episodes, {args.horizon} steps)",
                 fontsize=19, pad=18)
    figure.tight_layout(); figure.savefig(args.output / "open_loop_state_comparison.pdf", bbox_inches="tight")
    figure.savefig(args.output / "open_loop_state_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    xloc = np.arange(len(COMMON)); width = .36
    figure, ax = plt.subplots(figsize=(11.5, 7.0))
    flow_bars = ax.bar(xloc - width / 2, flow_nrmse, width, label="NormFlowSDE", color="#4C78A8")
    physical_bars = ax.bar(xloc + width / 2, physical_nrmse, width,
                           label="Nonlinear physical model", color="#F58518")
    ax.set_xticks(xloc, COMMON, fontsize=18, rotation=18, ha="right")
    ax.set_ylabel("Normalized RMSE", fontsize=22)
    ax.set_title(f"Open-loop state reconstruction ({len(truths)} test episodes, {args.horizon} steps)",
                 fontsize=24, pad=14)
    ax.tick_params(axis="y", labelsize=18); ax.grid(axis="y", alpha=.25, linestyle="--")
    ax.legend(fontsize=18); ax.set_axisbelow(True)
    ax.bar_label(flow_bars, fmt="%.3f", padding=3, fontsize=14, rotation=90)
    ax.bar_label(physical_bars, fmt="%.3f", padding=3, fontsize=14, rotation=90)
    figure.tight_layout()
    figure.savefig(args.output / "open_loop_nrmse_comparison_barplot.pdf", bbox_inches="tight")
    figure.savefig(args.output / "open_loop_nrmse_comparison_barplot.png", dpi=220, bbox_inches="tight")
    plt.close(figure); print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True); parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--physical-model", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    run(parser.parse_args())
