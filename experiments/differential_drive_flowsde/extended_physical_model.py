"""Identified nonlinear 8-state differential-drive model with structured pose integration."""
from pathlib import Path

import h5py
import numpy as np

STATE_ORDER = ("p_x", "p_y", "theta", "v_forward", "v_lateral", "omega",
               "wheel_left", "wheel_right")
DYNAMIC = np.asarray((3, 4, 5, 6, 7))


def friction_at(x, y):
    variation = (.55 * np.sin(.72 * x) * np.cos(.51 * y)
                 + .30 * np.sin(1.31 * y + .37 * x)
                 + .15 * np.cos(1.73 * x - .44 * y))
    return np.clip(.72 + .16 * variation, .48, 1.0)


def feature_library(state, action):
    """Physics-guided nonlinear basis for actuator, wheel, slip, and body dynamics."""
    x = np.asarray(state); u = np.asarray(action)
    vf, vl, om, wl, wr = (x[..., i] for i in DYNAMIC)
    vc, oc = u[..., 0], u[..., 1]
    tr = friction_at(x[..., 0], x[..., 1]) / .72
    return np.stack((np.ones_like(vf), vf, vl, om, wl, wr, vc, oc, tr,
                     vf * np.abs(vf), vl * np.abs(vl), om * np.abs(om),
                     wl * np.abs(wl), wr * np.abs(wr), vf * om, vl * om,
                     wl + wr, wr - wl, tr * (wl + wr), tr * (wr - wl),
                     tr * vc, tr * oc, np.abs(om) * vf, vf * vl), axis=-1)


class ExtendedPhysicalModel:
    def __init__(self, coefficients, feature_mean, feature_std, lower, upper, dt=.02):
        self.coefficients = coefficients; self.feature_mean = feature_mean
        self.feature_std = feature_std; self.lower = lower; self.upper = upper; self.dt = dt

    def step(self, state, action):
        state = np.asarray(state); features = feature_library(state, action)
        normalized = (features - self.feature_mean) / self.feature_std
        delta = normalized @ self.coefficients
        nxt = state.copy(); nxt[..., DYNAMIC] += delta
        nxt[..., DYNAMIC] = np.clip(nxt[..., DYNAMIC], self.lower, self.upper)
        vf = .5 * (state[..., 3] + nxt[..., 3]); vl = .5 * (state[..., 4] + nxt[..., 4])
        om = .5 * (state[..., 5] + nxt[..., 5]); theta = state[..., 2] + self.dt * om
        c, s = np.cos(theta), np.sin(theta)
        nxt[..., 0] = state[..., 0] + self.dt * (vf * c - vl * s)
        nxt[..., 1] = state[..., 1] + self.dt * (vf * s + vl * c)
        nxt[..., 2] = np.arctan2(s, c)
        return nxt

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, coefficients=self.coefficients, feature_mean=self.feature_mean,
                 feature_std=self.feature_std, lower=self.lower, upper=self.upper, dt=self.dt,
                 state_order=np.asarray(STATE_ORDER))

    @classmethod
    def load(cls, path: Path):
        values = np.load(path)
        return cls(values["coefficients"], values["feature_mean"], values["feature_std"],
                   values["lower"], values["upper"], float(values["dt"]))


def fit(data: Path, output: Path, ridge=1e-3):
    with h5py.File(data) as f:
        state, action, nxt, split = (f[k][:] for k in ("state", "action", "next_state", "split"))
        dt = float(f.attrs["dt"])
    train = split == 0; features = feature_library(state[train], action[train])
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-6)
    design = (features - mean) / std; targets = nxt[train][:, DYNAMIC] - state[train][:, DYNAMIC]
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ targets)
    values = state[train][:, DYNAMIC]; margin = .1 * np.maximum(np.ptp(values, axis=0), 1e-3)
    model = ExtendedPhysicalModel(coefficients, mean, std, values.min(0) - margin,
                                  values.max(0) + margin, dt)
    model.save(output); return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    fit(**vars(parser.parse_args()))
